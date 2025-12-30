# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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

from functools import wraps
from typing import Any, Callable

import torch
from torch.utils.hooks import BackwardHook

from msprobe.core.common.const import Const
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import is_float8_tensor
from msprobe.pytorch.dump.api_dump.api_register import get_api_register

torch_version_above_or_equal_2 = torch.__version__.split('+')[0] >= '2.0'


def wrap_setup_backward_hook(func):
    def requires_clone(tensor, need_check_leaf=False):
        if not isinstance(tensor, torch.Tensor):
            return False
        need_clone = torch.is_grad_enabled() and tensor.requires_grad and not is_float8_tensor(tensor)
        if need_check_leaf:
            need_clone &= tensor.grad_fn is not None
        return need_clone

    @recursion_depth_decorator("Dump: wrap_setup_backward_hook.parse_tensor", max_depth=Const.DUMP_MAX_DEPTH)
    def parse_tensor(item, tensor_list):
        if requires_clone(item):
            tensor_list.append(item)
        elif isinstance(item, (list, tuple)):
            for value in item:
                parse_tensor(value, tensor_list)
        elif isinstance(item, dict):
            for value in item.values():
                parse_tensor(value, tensor_list)

    @recursion_depth_decorator("Dump: wrap_setup_backward_hook.rebuild_args", max_depth=Const.DUMP_MAX_DEPTH)
    def rebuild_args(item, tensor_iter, need_check_leaf=False):
        if requires_clone(item, need_check_leaf):
            result = next(tensor_iter)
            if hasattr(result, "_base") and result._base is not None:
                if torch._C._autograd._get_creation_meta(result) != torch._C._autograd.CreationMeta(0):
                    torch._C._autograd._set_creation_meta(result, torch._C._autograd.CreationMeta(0))
            return result
        if isinstance(item, list):
            for index, value in enumerate(item):
                item[index] = rebuild_args(value, tensor_iter, need_check_leaf=True)
            return item
        if isinstance(item, dict):
            for key, value in item.items():
                item[key] = rebuild_args(value, tensor_iter, need_check_leaf=True)
            return item
        if isinstance(item, tuple):
            if hasattr(item, '_fields'):
                return type(item)(*[rebuild_args(i, tensor_iter) for i in item])
            return type(item)([rebuild_args(i, tensor_iter) for i in item])
        return item

    @wraps(func)
    def wrap_setup_hook_func(*args, **kwargs):
        if len(args) < 2:
            return func(*args, **kwargs)

        actual_args = args[1]

        tensor_list = []

        parse_tensor(actual_args, tensor_list)

        new_args = args[0], tuple(tensor_list)
        hooked_tensors = func(*new_args, **kwargs)

        tensor_iter = iter(hooked_tensors)
        try:
            new_data = rebuild_args(actual_args, tensor_iter)
        except Exception as e:
            logger.debug(f"Unsupported data in setup input/output hook. The detail info: {e}")
            new_data = actual_args

        return new_data

    return wrap_setup_hook_func


def wrap_setup_input_output_hook():
    BackwardHook.setup_input_hook = wrap_setup_backward_hook(BackwardHook.setup_input_hook)
    BackwardHook.setup_output_hook = wrap_setup_backward_hook(BackwardHook.setup_output_hook)


def get_apply_func_wrapper(original_func: Callable) -> Callable:
    @wraps(original_func)
    def wrapped_apply(*args, **kwargs) -> Any:
        api_register = get_api_register()
        if api_register:
            api_register.restore_inner_used_api()
        result = original_func(*args, **kwargs)
        if api_register:
            api_register.register_inner_used_api()
        return result

    return wrapped_apply


def wrap_backward_hook_function_apply():
    if torch_version_above_or_equal_2:
        original_apply = torch.nn.modules._functions.BackwardHookFunction.apply
        torch.nn.modules._functions.BackwardHookFunction.apply = get_apply_func_wrapper(original_apply)
