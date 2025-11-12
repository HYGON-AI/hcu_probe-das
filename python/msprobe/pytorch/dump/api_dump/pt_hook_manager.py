# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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
import threading
from contextlib import nullcontext

import torch

from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.core.common.runtime import Runtime
from msprobe.core.common.utils import replace_last_occurrence, ThreadSafe
from msprobe.core.dump.data_dump.data_processor.base import ModuleForwardInputsOutputs
from msprobe.core.dump.hook_manager import BaseHookManager, HookSet
from msprobe.pytorch.common.utils import (
    torch_version_above_or_equal_2,
    register_forward_hook,
    Const as PtConst
)
from msprobe.pytorch.dump.api_dump.hook_module import HOOKModule


class PytorchHookManager(BaseHookManager):
    @staticmethod
    def _no_grad_context():
        return nullcontext()

    @staticmethod
    def _add_count(name):
        HOOKModule.add_module_count(name)

    @staticmethod
    def _get_count(name):
        return HOOKModule.get_module_count(name)

    @staticmethod
    def _process_kwargs_and_output(module, tid, hook_type, kwargs_or_output, output_or_kwargs):
        if hook_type == Const.API:
            kwargs = kwargs_or_output
            output = output_or_kwargs
        else:
            kwargs = kwargs_or_output if torch_version_above_or_equal_2 else {}
            output = output_or_kwargs if torch_version_above_or_equal_2 else kwargs_or_output
        return kwargs, output

    def build_hook(self, hook_type, name):
        if hook_type == Const.API:
            hook_set = HookSet(
                forward_pre_hook=self._build_forward_pre_hook(hook_type, name),
                distributed_forward_hook=self._build_distributed_forward_hook()
            )
        else:
            full_backward_name = replace_last_occurrence(name, Const.FORWARD, Const.BACKWARD)
            hook_set = HookSet(
                forward_hook=self._build_forward_hook(hook_type, name),
                backward_hook=self._build_backward_hook(hook_type, full_backward_name)
            )
        return hook_set

    def _register_forward_hook(self, module, api_name):
        if not hasattr(module, 'msprobe_forward_hook'):
            register_forward_hook(module, self._build_forward_hook(Const.API, api_name))
            setattr(module, 'msprobe_forward_hook', True)

    def _register_backward_hook(self, module, full_backward_name, args):
        pass

    def _register_backward_pre_hook(self, module, full_backward_name, args, kwargs, output):
        if module.prefix_api_name in PtConst.DROPOUT_API_LIST:
            p = kwargs.get('p', 0.5)
            if len(args) >= 2:
                p = args[1]
            if p == 0:
                logger.debug(f"The parameter 'p' is set to 0, so the {full_backward_name} data will not be collected.")
                return output

        var = output
        while not isinstance(var, torch.Tensor):
            if isinstance(var, dict):
                var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
            elif isinstance(var, (list, tuple)):
                if var:
                    var = var[0]
                else:
                    return output
            else:
                return output

        if not (var.requires_grad and torch.is_grad_enabled()):
            return output

        grad_fn = var.grad_fn
        if grad_fn is not None:
            backward_hook = self._build_backward_hook(Const.API, full_backward_name)
            wrapper = functools.partial(backward_hook, module)
            functools.update_wrapper(wrapper, backward_hook)
            grad_fn.register_hook(wrapper)

        return output

    def _need_exchange(self, module):
        return True

    def _get_params_dict(self, module):
        params_dict = {}
        if self.config.task != Const.STRUCTURE:
            params_dict = {
                key.split(Const.SEP)[-1]: value
                for key, value in module.named_parameters(recurse=False)
            }
        return params_dict

    def _build_distributed_forward_hook(self):
        def distributed_forward_hook(module, full_name, args, kwargs, output):
            if not full_name or not Runtime.is_running:
                return

            tid = threading.get_ident()
            with ThreadSafe():
                BaseHookManager.inner_switch[tid] = True
                self.data_collector.update_api_or_module_name(full_name)
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=output)
                with self._no_grad_context():
                    self.data_collector.forward_output_data_collect(
                        full_name,
                        module,
                        self._pid,
                        module_input_output
                    )
                BaseHookManager.inner_switch[tid] = False

        return distributed_forward_hook
