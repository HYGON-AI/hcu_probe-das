# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import re

import torch
import torch.distributed as dist
import torch.nn as nn

from msprobe.core.common.const import MonitorConst
from msprobe.core.common.file_utils import load_yaml
from msprobe.pytorch.monitor.module_metric import get_metrics, get_summary_writer_tag_name
from msprobe.pytorch.common.log import logger

try:
    import torch_npu
except ImportError:
    pass

RANK = None

OpsPath = os.path.join(os.path.dirname(__file__), "distributed_ops.yaml")
WrapDistributedOps = load_yaml(OpsPath).get("distributed", [])

StackBlackListPath = os.path.join(os.path.dirname(__file__), "stack_blacklist.yaml")
StackBlackList = load_yaml(StackBlackListPath).get("stack", [])
MAX_STRING_LENGTH = 1000

distributed_func = {}
for f in dir(dist):
    distributed_func[f] = getattr(dist, f)

ORIGIN_WAIT = getattr(dist.Work, 'wait')
PENDING_ASYNC_CC_BY_HANDLE = {}


def get_distributed_ops():
    global WrapDistributedOps
    _all_distributed_ops = dir(dist)
    return set(WrapDistributedOps) & set(_all_distributed_ops)


class DistributedOPTemplate(nn.Module):
    def __init__(self, op_name, pre_hooks, post_hooks):
        super(DistributedOPTemplate, self).__init__()
        self.op_name_ = str(op_name)
        self.__name__ = self.op_name_
        self.cc_hooks = []
        for pre_hook in pre_hooks:
            handle = self.register_forward_pre_hook(pre_hook, with_kwargs=True)
            self.cc_hooks.append(handle)
        for hook in post_hooks:
            handle = self.register_forward_hook(hook, with_kwargs=True)
            self.cc_hooks.append(handle)

    def forward(self, *args, **kwargs):
        return distributed_func.get(self.op_name_)(*args, **kwargs)


class ApiRegistry:
    def __init__(self):
        self.distributed_attr_origin = {}
        self.distributed_attr_hooked = {}

    @staticmethod
    def store_ori_attr(ori_api_group, api_list, api_ori_attr):
        for api in api_list:
            if '.' in api:
                sub_module_name, sub_op = api.rsplit('.', 1)
                sub_module = getattr(ori_api_group, sub_module_name)
                api_ori_attr[api] = getattr(sub_module, sub_op)
            else:
                api_ori_attr[api] = getattr(ori_api_group, api)

    @staticmethod
    def set_api_attr(api_group, attr_dict):
        for cc_api_name, cc_api_entry_func in attr_dict.items():
            if '.' in cc_api_name:
                sub_module_name, sub_op = cc_api_name.rsplit('.', 1)
                sub_module = getattr(api_group, sub_module_name, None)
                if sub_module is not None:
                    setattr(sub_module, sub_op, cc_api_entry_func)
            else:
                setattr(api_group, cc_api_name, cc_api_entry_func)

    @staticmethod
    def redirect_wait():
        global ORIGIN_WAIT
        global PENDING_ASYNC_CC_BY_HANDLE

        def wrapped_wait(work):
            def wrapped_wait(*args, **kwargs):
                ORIGIN_WAIT(*args, **kwargs)
                if args[0] in PENDING_ASYNC_CC_BY_HANDLE:
                    store_func = PENDING_ASYNC_CC_BY_HANDLE.pop(args[0])
                    store_func()

            return wrapped_wait

        dist.Work.wait = wrapped_wait(dist.Work)

    def redirect_api(self):
        self.set_api_attr(dist, self.distributed_attr_hooked)
        self.set_api_attr(dist.distributed_c10d, self.distributed_attr_hooked)
        self.redirect_wait()

    def restore_api(self):
        self.set_api_attr(dist, self.distributed_attr_origin)
        self.set_api_attr(dist.distributed_c10d, self.distributed_attr_origin)
        setattr(dist.Work, 'wait', ORIGIN_WAIT)

    def initialize_hook(self, pre_hooks, post_hooks):
        self.store_ori_attr(dist, get_distributed_ops(), self.distributed_attr_origin)
        cc_hooks = []
        for op_name in get_distributed_ops():
            self.distributed_attr_hooked[op_name] = DistributedOPTemplate(op_name, pre_hooks, post_hooks)
            cc_hooks.extend(self.distributed_attr_hooked[op_name].cc_hooks)
        return cc_hooks


def get_process_group(process_group):
    return (
        process_group
        if isinstance(process_group, dist.ProcessGroup)
        else dist.GroupMember.WORLD
    )


def stack_filter(stack):
    if len(stack) > MAX_STRING_LENGTH:
        logger.warning(f'The character string contains more than {MAX_STRING_LENGTH}. re match is skipped.')
    for pattern in StackBlackList:
        if re.search(pattern, stack):
            return False
    return True


def get_callstack():
    callstack = []
    for (_, path, line, func, _, _) in inspect.stack():
        stack_line = f'{path}[{line}]'
        if stack_filter(stack_line):
            callstack.append(stack_line + '   ' + func)
    return callstack


@torch.no_grad()
def op_aggregate(op, tensorlist):
    if isinstance(tensorlist, torch.Tensor):
        return tensorlist
    if not tensorlist:
        return torch.tensor(torch.nan)
    if op == 'min':
        return min(tensorlist)
    if op == 'max':
        return max(tensorlist)
    if op == 'norm':
        return sum(tensorlist)
    if op == 'zeros':
        return sum(tensorlist) / len(tensorlist)
    if op == 'nans':
        return sum(tensorlist)
    if op == 'mean':
        return sum(tensorlist) / len(tensorlist)
    return torch.tensor(torch.nan)


def update_data(old, new):
    for tag, op2tensor in new.items():
        if tag not in old:
            old[tag] = {}
        for op, tensor in op2tensor.items():
            if op not in old[tag]:
                old[tag][op] = [tensor]
            else:
                old[tag][op].append(tensor)
    return old


def is_target_line(codeline):
    if codeline == []:
        return True
    stack = get_callstack()
    whole_stack = ';'.join(stack)
    if len(whole_stack) > MAX_STRING_LENGTH:
        logger.warning(f'The character string contains more than {MAX_STRING_LENGTH}. re match is skipped.')
    for pattern in codeline:
        if re.search(pattern, whole_stack):
            return True
    return False


@torch.no_grad()
def catch_data(cc_context, cc_name, ops, args, prefix):
    tensor_args = {}
    for arg in args:
        if isinstance(arg, torch.Tensor):
            key = get_summary_writer_tag_name(cc_name, f'{prefix}_{len(tensor_args)}', RANK)
            tensor_args[key] = arg
        elif isinstance(arg, list):
            if isinstance(arg[0], torch.Tensor):
                stacked_arg = torch.stack(arg)
            elif isinstance(arg[0], dist.P2POp):
                stacked_arg = torch.stack([op.tensor for op in arg])
            key = get_summary_writer_tag_name(cc_name, f'{prefix}_{len(tensor_args)}', RANK)
            tensor_args[key] = stacked_arg

    new_data = get_metrics(ops, tensor_args, 1e-8)
    cc_context.data = update_data(cc_context.data, new_data)


def create_async_callback_func(context, cc_name, ops, args, prefix):
    def store_data():
        catch_data(context, cc_name, ops, args, prefix)

    return store_data


def create_hooks(context, monitor):
    def cc_log_hook(module, args, kwargs):
        stack = ';'.join(get_callstack())
        monitor.cc_logged_stack[module.op_name_].add(stack)
        return

    def cc_pre_hook(module, args, kwargs):
        if not is_target_line(monitor.cc_codeline):
            return
        args = args + tuple(kwargs.values())
        catch_data(context[module.op_name_], module.op_name_, monitor.ops, args, MonitorConst.PREFIX_PRE)
        return

    def cc_hook(module, args, kwargs, out=None):
        if not is_target_line(monitor.cc_codeline):
            return out
        args = args + tuple(kwargs.values())
        if out:  # async
            if isinstance(out, dist.Work):
                PENDING_ASYNC_CC_BY_HANDLE[out] = create_async_callback_func(
                    context[module.op_name_],
                    module.op_name_,
                    monitor.ops, args,
                    MonitorConst.PREFIX_POST
                )
            elif isinstance(out, list):  # batch_isend_irecv
                for out_element in out:
                    PENDING_ASYNC_CC_BY_HANDLE[out_element] = create_async_callback_func(
                        context[module.op_name_],
                        module.op_name_,
                        monitor.ops, args,
                        MonitorConst.PREFIX_POST
                    )
            return out
        catch_data(context[module.op_name_], module.op_name_, monitor.ops, args, MonitorConst.PREFIX_POST)
        return out

    global RANK
    pre_hooks = []
    hooks = []
    RANK = dist.get_rank()
    if dist.is_initialized() and RANK not in monitor.module_rank_list and monitor.module_rank_list != []:
        return [pre_hooks, hooks]

    if monitor.cc_log_only:
        pre_hooks.append(cc_log_hook)
        return [pre_hooks, hooks]

    if monitor.cc_pre_hook:
        pre_hooks.append(cc_pre_hook)

    hooks.append(cc_hook)

    return [pre_hooks, hooks]


api_register = ApiRegistry()
