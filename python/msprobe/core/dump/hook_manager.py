# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import gc
import os
import threading
from abc import ABC, abstractmethod
from collections import defaultdict

from msprobe.core.common.runtime import Runtime
from msprobe.core.common.utils import Const, ThreadSafe
from msprobe.core.dump.data_dump.data_processor.base import (ModuleBackwardInputsOutputs, ModuleForwardInputsOutputs)


class HookSet:
    def __init__(
            self,
            forward_pre_hook=None,
            forward_hook=None,
            backward_pre_hook=None,
            backward_hook=None,
            distributed_forward_hook=None
    ):
        self.forward_pre_hook = forward_pre_hook
        self.forward_hook = forward_hook
        self.backward_pre_hook = backward_pre_hook
        self.backward_hook = backward_hook
        self.distributed_forward_hook = distributed_forward_hook


class BaseHookManager(ABC):
    inner_switch = defaultdict(bool)
    inner_api_count = defaultdict(int)
    hook_handle_dict = {}
    params_grad_info = {}
    grad_hook_call = {}

    def __init__(self, data_collector, config):
        self.data_collector = data_collector
        self.config = config
        self.current_pid = self._pid

    @property
    def _pid(self):
        return os.getpid()

    @staticmethod
    def reset_status():
        BaseHookManager.inner_switch = defaultdict(bool)
        BaseHookManager.inner_api_count = defaultdict(int)
        BaseHookManager.params_grad_info.clear()

    @staticmethod
    def ensure_gc_enabled():
        is_gc_disabled = not gc.isenabled()
        if is_gc_disabled:
            gc.enable()
        return is_gc_disabled

    @staticmethod
    def restore_gc_state(original_state):
        if original_state:
            gc.disable()

    @staticmethod
    def _clear_input_kwargs(module, tid):
        if hasattr(module, 'msprobe_input_kwargs') and tid in module.msprobe_input_kwargs:
            del module.msprobe_input_kwargs[tid]

    @staticmethod
    def _get_grad_hook_call_index(ori_name, param_name):
        if ori_name not in BaseHookManager.grad_hook_call:
            BaseHookManager.grad_hook_call[ori_name] = [0, param_name]
        else:
            if BaseHookManager.grad_hook_call.get(ori_name)[1] == param_name:
                BaseHookManager.grad_hook_call[ori_name][0] += 1
        return BaseHookManager.grad_hook_call.get(ori_name)[0]

    @staticmethod
    @abstractmethod
    def _no_grad_context():
        pass

    @staticmethod
    @abstractmethod
    def _add_count(name):
        pass

    @staticmethod
    @abstractmethod
    def _get_count(name):
        pass

    @staticmethod
    @abstractmethod
    def _process_kwargs_and_output(module, tid, hook_type, kwargs_or_output, output_or_kwargs):
        pass

    @abstractmethod
    def build_hook(self, hook_type, name):
        pass

    @abstractmethod
    def _register_forward_hook(self, module, api_name):
        pass

    @abstractmethod
    def _register_backward_hook(self, module, full_backward_name, args):
        pass

    @abstractmethod
    def _register_backward_pre_hook(self, module, full_backward_name, args, kwargs, output):
        pass

    @abstractmethod
    def _get_params_dict(self, module):
        pass

    @abstractmethod
    def _need_exchange(self, module):
        pass

    @abstractmethod
    def _register_param_hook(self, name, module, params_dict):
        pass

    def is_child_process(self):
        return self.current_pid != self._pid

    def _should_execute_hook(self, hook_type, tid, is_forward=True):
        is_api_hook = hook_type == Const.API
        if self.is_child_process():
            return False
        if BaseHookManager.inner_switch[tid]:
            return False
        if not is_api_hook and not Runtime.is_running:
            return False
        elif is_api_hook and is_forward and not Runtime.is_running:
            return False
        if not self.data_collector or self.data_collector.data_processor.is_terminated:
            return False
        return True

    def _build_forward_pre_hook(self, hook_type, api_name):
        def forward_pre_hook(module, args, kwargs=None):
            if hook_type == Const.MODULE:
                return None

            tid = threading.get_ident()
            if not self._should_execute_hook(hook_type, tid):
                return None

            with ThreadSafe():
                original_state = self.ensure_gc_enabled()
                self._register_forward_hook(module, api_name)
                BaseHookManager.inner_api_count[tid] += 1
                if BaseHookManager.inner_api_count[tid] != 1:
                    return None

                full_forward_name = api_name + str(self._get_count(api_name)) + Const.SEP + Const.FORWARD
                full_backward_name = api_name + str(self._get_count(api_name)) + Const.SEP + Const.BACKWARD
                module.full_forward_name = full_forward_name
                if kwargs is None:
                    kwargs = module.msprobe_input_kwargs.get(tid, {}) if hasattr(module, 'msprobe_input_kwargs') else {}
                BaseHookManager.inner_switch[tid] = True
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=None)

                args = self._register_backward_hook(module, full_backward_name, args)
                with self._no_grad_context():
                    self.data_collector.update_api_or_module_name(full_forward_name)
                    self.data_collector.forward_input_data_collect(
                        full_forward_name,
                        module,
                        self._pid,
                        module_input_output
                    )
                BaseHookManager.inner_switch[tid] = False
                self.restore_gc_state(original_state)
                return args

        return forward_pre_hook

    def _build_forward_hook(self, hook_type, api_name):
        def forward_hook(module, args, kwargs_or_output, output_or_kwargs=None):
            tid = threading.get_ident()
            if not self._should_execute_hook(hook_type, tid):
                self._clear_input_kwargs(module, tid)
                return None

            with ThreadSafe():
                original_state = self.ensure_gc_enabled()
                if hook_type == Const.API:
                    if BaseHookManager.inner_api_count[tid] != 1:
                        if BaseHookManager.inner_api_count[tid] > 1:
                            BaseHookManager.inner_api_count[tid] -= 1
                        self._clear_input_kwargs(module, tid)
                        return None

                kwargs, output = self._process_kwargs_and_output(
                    module,
                    tid,
                    hook_type,
                    kwargs_or_output,
                    output_or_kwargs
                )
                BaseHookManager.inner_switch[tid] = True
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=output)
                if hook_type == Const.API:
                    full_forward_name = api_name + str(self._get_count(api_name)) + Const.SEP + Const.FORWARD
                    full_backward_name = api_name + str(self._get_count(api_name)) + Const.SEP + Const.BACKWARD
                    output = self._register_backward_pre_hook(module, full_backward_name, args, kwargs, output)

                with self._no_grad_context():
                    if hook_type == Const.MODULE:
                        params_dict = self._get_params_dict(module)
                        setattr(module_input_output, Const.PARAMS, params_dict)
                        if params_dict:
                            self._register_param_hook(api_name, module, params_dict)
                        self.data_collector.update_api_or_module_name(api_name)
                        self.data_collector.forward_data_collect(
                            api_name,
                            module,
                            self._pid,
                            module_input_output
                        )
                    else:
                        self.data_collector.update_api_or_module_name(full_forward_name)
                        self.data_collector.forward_output_data_collect(
                            full_forward_name,
                            module,
                            self._pid,
                            module_input_output
                        )
                        self._add_count(api_name)
                        BaseHookManager.inner_api_count[tid] -= 1
                    self._clear_input_kwargs(module, tid)
                BaseHookManager.inner_switch[tid] = False
                self.restore_gc_state(original_state)
                return output

        return forward_hook

    def _build_backward_hook(self, hook_type, full_name):
        def backward_hook(module, grad_input, grad_output):
            tid = threading.get_ident()
            if not self._should_execute_hook(hook_type, tid, is_forward=False):
                return

            with ThreadSafe():
                original_state = self.ensure_gc_enabled()
                BaseHookManager.inner_switch[tid] = True
                self.data_collector.update_api_or_module_name(full_name)

                need_exchange = self._need_exchange(module) if hook_type == Const.MODULE else True
                if need_exchange:
                    module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_output, grad_output=grad_input)
                else:
                    module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_input, grad_output=grad_output)
                self.data_collector.backward_data_collect(
                    full_name,
                    module,
                    self._pid,
                    module_input_output
                )
                BaseHookManager.inner_switch[tid] = False
                self.restore_gc_state(original_state)

        return backward_hook
