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

import numpy as np

from mindspore import nn, Tensor, ops, _no_grad
from mindspore import communication
from mindspore.communication import comm_func, get_rank

from msprobe.core.common.const import MonitorConst, Const
from msprobe.core.common.file_utils import load_yaml
from msprobe.mindspore.monitor.utils import get_metrics, get_summary_writer_tag_name

enable_communication = True
try:
    from mindspore._c_expression import CommHandle as CommHandle_
except ImportError:
    enable_communication = False


RANK = None

OpsPath = os.path.join(os.path.dirname(__file__), "distributed_ops.yaml")
WrapDistributedOps = load_yaml(OpsPath).get("communication.comm_func", [])

StackBlackListPath = os.path.join(os.path.dirname(__file__), "stack_blacklist.yaml")
StackBlackList = load_yaml(StackBlackListPath).get("stack", [])

distributed_func = {}
for f in dir(comm_func):
    distributed_func[f] = getattr(comm_func, f)

ORIGIN_WAIT = CommHandle_.wait if enable_communication else None
PENDING_ASYNC_CC_BY_HANDLE = {}


def get_distributed_ops():
    global WrapDistributedOps
    _all_distributed_ops = dir(comm_func)
    return set(WrapDistributedOps) & set(_all_distributed_ops)


class DistributedOPTemplate(nn.Cell):
    def __init__(self, op_name, pre_hooks, post_hooks):
        super(DistributedOPTemplate, self).__init__()
        self.op_name_ = str(op_name)
        self.__name__ = self.op_name_
        self.cc_hooks = []
        for pre_hook in pre_hooks:
            handle = self.register_forward_pre_hook(pre_hook)
            self.cc_hooks.append(handle)
        for hook in post_hooks:
            handle = self.register_forward_hook(hook)
            self.cc_hooks.append(handle)

    def construct(self, *args, **kwargs):
        return distributed_func.get(self.op_name_)(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return distributed_func.get(self.op_name_)(*args, **kwargs)


class ApiRegistry:
    def __init__(self):
        self.distributed_attr_origin = {}
        self.distributed_attr_hooked = {}

    @staticmethod
    def store_ori_attr(ori_api_group, api_list, api_ori_attr):
        for api in api_list:
            if Const.SEP in api:
                sub_module_name, sub_op = api.rsplit(Const.SEP, 1)
                sub_module = getattr(ori_api_group, sub_module_name)
                api_ori_attr[api] = getattr(sub_module, sub_op)
            else:
                api_ori_attr[api] = getattr(ori_api_group, api)

    @staticmethod
    def set_api_attr(api_group, attr_dict):
        for cc_api_name, cc_api_entry_func in attr_dict.items():
            if Const.SEP in cc_api_name:
                sub_module_name, sub_op = cc_api_name.rsplit(Const.SEP, 1)
                sub_module = getattr(api_group, sub_module_name, None)
                if sub_module is not None:
                    setattr(sub_module, sub_op, cc_api_entry_func)
            else:
                setattr(api_group, cc_api_name, cc_api_entry_func)

    @staticmethod
    def redirect_wait():
        global ORIGIN_WAIT
        global PENDING_ASYNC_CC_BY_HANDLE
        if not ORIGIN_WAIT:
            return

        def wrapped_wait(work):
            def wrapped_wait(*args, **kwargs):
                ORIGIN_WAIT(*args, **kwargs)
                if args[0] in PENDING_ASYNC_CC_BY_HANDLE:
                    store_func = PENDING_ASYNC_CC_BY_HANDLE.pop(args[0])
                    store_func()

            return wrapped_wait

        CommHandle_.wait = wrapped_wait(CommHandle_)

    def redirect_api(self):
        self.set_api_attr(comm_func, self.distributed_attr_hooked)
        self.redirect_wait()

    def restore_api(self):
        if not ORIGIN_WAIT:
            return
        self.set_api_attr(comm_func, self.distributed_attr_origin)
        setattr(CommHandle_, 'wait', ORIGIN_WAIT)

    def initialize_hook(self, pre_hooks, post_hooks):
        self.store_ori_attr(comm_func, get_distributed_ops(), self.distributed_attr_origin)
        cc_hooks = []
        for op_name in get_distributed_ops():
            self.distributed_attr_hooked[op_name] = DistributedOPTemplate(op_name, pre_hooks, post_hooks)
            cc_hooks.extend(self.distributed_attr_hooked[op_name].cc_hooks)
        return cc_hooks


def get_process_group(process_group):
    return (
        process_group
        if process_group
        else comm_func.HCCL_WORLD_GROUP
    )


def stack_filter(stack):
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


@_no_grad()
def op_aggregate(op, tensorlist):
    if isinstance(tensorlist, Tensor):
        return tensorlist
    if not tensorlist:
        return Tensor(float('nan'))
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
    return Tensor(float('nan'))


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
    stack = get_callstack()
    whole_stack = ';'.join(stack)
    if codeline == []:
        return True
    for pattern in codeline:
        if re.search(pattern, whole_stack):
            return True
    return False


@_no_grad()
def catch_data(cc_context, cc_name, ops_list, args, prefix):
    tensor_args = {}
    for arg in args:
        if isinstance(arg, Tensor):
            key = get_summary_writer_tag_name(cc_name, f'{prefix}_{len(tensor_args)}', RANK)
            tensor_args[key] = arg
        elif isinstance(arg, list):
            if isinstance(arg[0], Tensor):
                stacked_arg = ops.stack(arg)
            elif isinstance(arg[0], comm_func.P2POp):
                stacked_arg = ops.stack([op.tensor for op in arg])
            key = get_summary_writer_tag_name(cc_name, f'{prefix}_{len(tensor_args)}', RANK)
            tensor_args[key] = stacked_arg

    new_data = get_metrics(ops_list, tensor_args, 1e-8)
    cc_context.data = update_data(cc_context.data, new_data)


def create_async_callback_func(context, cc_name, ops_list, args, prefix):
    def store_data():
        catch_data(context, cc_name, ops_list, args, prefix)

    return store_data


def create_hooks(context, monitor):
    def cc_log_hook(module, inputs):
        stack = ';'.join(get_callstack())
        monitor.cc_logged_stack[module.op_name_].add(stack)
        return

    def cc_pre_hook(module, inputs):
        if not is_target_line(monitor.cc_codeline):
            return
        catch_data(context[module.op_name_], module.op_name_, monitor.ops, inputs, MonitorConst.PREFIX_PRE)
        return

    def cc_hook(module, inputs, out=None):
        if not is_target_line(monitor.cc_codeline):
            return out
        if out and enable_communication:  # async
            if isinstance(out, CommHandle_):
                PENDING_ASYNC_CC_BY_HANDLE[out] = create_async_callback_func(
                    context[module.op_name_],
                    module.op_name_,
                    monitor.ops, inputs,
                    MonitorConst.PREFIX_POST
                )
            elif isinstance(out, list):  # batch_isend_irecv
                for out_element in out:
                    if isinstance(out_element, comm_func.P2POp):
                        PENDING_ASYNC_CC_BY_HANDLE[out_element] = create_async_callback_func(
                            context[module.op_name_],
                            module.op_name_,
                            monitor.ops, inputs,
                            MonitorConst.PREFIX_POST
                        )
            elif isinstance(out, tuple):
                if len(out) == 2 and isinstance(out[1], CommHandle_):
                    PENDING_ASYNC_CC_BY_HANDLE[out[1]] = create_async_callback_func(
                            context[module.op_name_],
                            module.op_name_,
                            monitor.ops, inputs,
                            MonitorConst.PREFIX_POST
                    )

            return out
        catch_data(context[module.op_name_], module.op_name_, monitor.ops, inputs, MonitorConst.PREFIX_POST)
        return out

    global RANK
    pre_hooks = []
    hooks = []
    RANK = get_rank()
    if communication.GlobalComm.INITED and RANK not in monitor.module_rank_list and monitor.module_rank_list != []:
        return [pre_hooks, hooks]

    if monitor.cc_log_only:
        pre_hooks.append(cc_log_hook)
        return [pre_hooks, hooks]

    if monitor.cc_pre_hook:
        pre_hooks.append(cc_pre_hook)

    hooks.append(cc_hook)

    return [pre_hooks, hooks]


api_register = ApiRegistry()
