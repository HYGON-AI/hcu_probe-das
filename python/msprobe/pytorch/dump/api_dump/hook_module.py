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

import functools
from collections import defaultdict

import torch
import torch.nn as nn
import torch.utils.hooks as full_hooks

from msprobe.pytorch.common.utils import register_forward_pre_hook, is_float8_tensor


class HOOKModule(nn.Module):
    module_count = defaultdict(int)

    def __init__(self, hook_build_func) -> None:
        super(HOOKModule, self).__init__()
        prefix = self.prefix_api_name if hasattr(self, "prefix_api_name") else ""
        op_is_distributed = self.op_is_distributed if hasattr(self, "op_is_distributed") else False
        if callable(hook_build_func):
            hook_set = hook_build_func(prefix)
            register_forward_pre_hook(self, hook_set.forward_pre_hook)
            if op_is_distributed:
                self.distributed_forward_hook = hook_set.distributed_forward_hook

    def __call__(self, *args, **kwargs):
        return self._call_func(*args, **kwargs)

    @staticmethod
    def reset_module_stats():
        HOOKModule.module_count = defaultdict(int)

    @staticmethod
    def add_module_count(name):
        HOOKModule.module_count[name] += 1

    @staticmethod
    def get_module_count(name):
        return HOOKModule.module_count[name]

    def _call_func(self, *args, **kwargs):
        full_backward_hooks, non_full_backward_hooks = [], []
        if len(self._backward_hooks) > 0:
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()
        for hook in self._forward_pre_hooks.values():
            hook(self, args, kwargs)
        bw_hook = None
        if len(full_backward_hooks) > 0:
            bw_hook = full_hooks.BackwardHook(self, full_backward_hooks)
            args = bw_hook.setup_input_hook(args)
        if torch._C._get_tracing_state():
            result = self._slow_forward(*args, **kwargs)
        else:
            result = self.forward(*args, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, args, kwargs, result)
            if hook_result is not None:
                result = hook_result
        if bw_hook:
            result = bw_hook.setup_output_hook(result)
        if len(non_full_backward_hooks) > 0:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                elif isinstance(var, (list, tuple)):
                    if var:
                        var = var[0]
                    else:
                        return result
                else:
                    return result

            if is_float8_tensor(var) or not (var.requires_grad and torch.is_grad_enabled()):
                return result

            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in non_full_backward_hooks:
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
                self._maybe_warn_non_full_backward_hook(args, result, grad_fn)
        return result
