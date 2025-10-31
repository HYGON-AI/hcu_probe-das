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

import atexit
import os
import threading
import traceback

from msprobe.core.data_dump.scope import ScopeFactory
from msprobe.core.data_dump.json_writer import DataWriter
from msprobe.core.common.log import logger
from msprobe.core.common.const import Const
from msprobe.core.data_dump.data_processor.factory import DataProcessorFactory
from msprobe.core.common.megatron_utils import MegatronStepInfo, get_micro_step, is_megatron


def build_data_collector(config):
    return DataCollector(config)


class DataCollector:
    tasks_need_tensor_data = [Const.OVERFLOW_CHECK, Const.TENSOR, Const.FREE_BENCHMARK]
    level_without_construct = [Const.LEVEL_L1, Const.LEVEL_L2]

    def __init__(self, config):
        self.config = config
        self.data_writer = DataWriter()
        self.data_processor = DataProcessorFactory.create_processor(self.config, self.data_writer)
        self.module_processor = DataProcessorFactory.get_module_processor(self.config.framework)
        self.module_count = {}
        self.scope = ScopeFactory(self.config).build_scope()
        self.backward_module_names = {}
        self.params_grad_record = {}
        self.optimizer_status = ""
        self.optimizer_status_first_start = {Const.OPTIMIZER: True, Const.CLIP_GRAD: True}
        atexit.register(self.write_json_at_exit)

    @property
    def dump_data_dir(self):
        return self.data_writer.dump_tensor_data_dir

    @property
    def dump_file_path(self):
        return self.data_writer.dump_file_path

    @staticmethod
    def check_scope_and_pid(scope, name, pid):
        return (not scope or scope.check(name)) and pid == os.getpid()

    @staticmethod
    def set_is_recomputable(data_info, is_recompute):
        if data_info and len(data_info) == 1 and is_recompute is not None: # 正常情况下data_info的长度应改为1
            data_info[list(data_info.keys())[0]]["is_recompute"] = is_recompute

    def reset_status(self):
        self.optimizer_status = ""
        self.optimizer_status_first_start = {Const.OPTIMIZER: True, Const.CLIP_GRAD: True}
        self.data_writer.reset_cache()
        self.backward_module_names.clear()

    def if_return_forward_new_output(self):
        return self.data_processor.if_return_forward_new_output()

    def get_forward_new_output(self):
        return self.data_processor.get_forward_new_output()

    def update_api_or_module_name(self, api_or_module_name):
        self.data_processor.update_api_or_module_name(api_or_module_name)

    def write_json(self):
        self.data_writer.write_json()

    def write_json_at_exit(self):
        if self.config.async_dump and self.config.task == Const.TENSOR:
            self.data_processor.dump_async_data()
        self.data_writer.write_json()

    def update_data(self, name, data_info):
        msg = f"msprobe is collecting data on {name}."
        if self.config.task == Const.OVERFLOW_CHECK:
            if self.data_processor.has_overflow:
                msg += " Overflow detected."
                logger.warning(msg)
                self.data_writer.update_data(data_info)
            return
        logger.debug(msg)
        self.data_writer.update_data(data_info)

    def call_stack_collect(self, data_info, name):
        stack_info, is_recompute = self.data_processor.analyze_api_call_stack(name)
        self.data_writer.update_stack(name, stack_info)
        self.set_is_recomputable(data_info, is_recompute)

    def forward_input_data_collect(self, name, module, pid, module_input_output):
        try:
            if self.config.task == Const.FREE_BENCHMARK:
                backward_name = name.replace(Const.FORWARD, Const.BACKWARD)
                if self.check_scope_and_pid(self.scope, backward_name, pid):
                    self.data_processor.analyze_forward_input(backward_name, module, module_input_output)
                return

            if not self.check_scope_and_pid(self.scope, name, pid):
                return

            data_info = {}
            if self.config.task != Const.STRUCTURE:
                data_info = self.data_processor.analyze_forward_input(name, module, module_input_output)
            if self.config.level == Const.LEVEL_L2:
                return
            self.call_stack_collect(data_info, name)
            self.handle_data(name, data_info, flush=self.data_processor.is_terminated)

        except Exception as e:
            # 取异常类名作为“类型”做去重
            error_type = type(e).__name__
            tb = traceback.format_exc()
            self.data_writer.write_error_log(
                f"[ERROR] forward_input_data_collect failed: name={name}, pid={pid}\n{tb}",
                error_type=error_type
            )

    def forward_output_data_collect(self, name, module, pid, module_input_output):
        try:
            self.update_construct(name)
            if not self.check_scope_and_pid(self.scope, name, pid):
                return

            data_info = {}
            if self.config.task != Const.STRUCTURE:
                data_info = self.data_processor.analyze_forward_output(name, module, module_input_output)
            is_recompute = self.data_processor.is_recompute()
            self.set_is_recomputable(data_info, is_recompute)
            if self.config.level == Const.LEVEL_L2:
                return
            self.handle_data(name, data_info, flush=self.data_processor.is_terminated)

        except Exception as e:
            # 取异常类名作为“类型”做去重
            error_type = type(e).__name__
            tb = traceback.format_exc()
            self.data_writer.write_error_log(
                f"[ERROR] forward_output_data_collect failed: name={name}, pid={pid}\n{tb}",
                error_type=error_type
            )

    def forward_data_collect_only_tensor(self, name, module, pid, module_input_output):
        try:
            if not self.check_scope_and_pid(self.scope, name, pid):
                return
            self.data_processor.analyze_forward(name, module, module_input_output)

        except Exception as e:
            # 取异常类名作为“类型”做去重
            error_type = type(e).__name__
            tb = traceback.format_exc()
            self.data_writer.write_error_log(
                f"[ERROR] forward_data_collect_only_tensor failed: name={name}, pid={pid}\n{tb}",
                error_type=error_type
            )

    def forward_data_collect(self, name, module, pid, module_input_output):
        try:

            self.update_construct(name)
            if not self.check_scope_and_pid(self.scope, name, pid):
                return
            data_info = {}
            if self.config.task != Const.STRUCTURE:
                data_info = self.data_processor.analyze_forward(name, module, module_input_output)
            self.call_stack_collect(data_info, name)
            self.handle_data(name, data_info, flush=self.data_processor.is_terminated)

        except Exception as e:
            error_type = type(e).__name__
            tb = traceback.format_exc()
            self.data_writer.write_error_log(
                f"[ERROR] forward_data_collect failed: name={name}, pid={pid}\n{tb}",
                error_type=error_type
            )

    def backward_data_collect_only_tensor(self, name, module, pid, module_input_output):
        try:
            if not self.check_scope_and_pid(self.scope, name, pid):
                return
            self.data_processor.analyze_backward(name, module, module_input_output)

        except Exception as e:
            error_type = type(e).__name__
            tb = traceback.format_exc()
            self.data_writer.write_error_log(
                f"[ERROR] backward_data_collect_only_tensor failed: name={name}, pid={pid}\n{tb}",
                error_type=error_type
            )

    def backward_data_collect(self, name, module, pid, module_input_output):
        try:
            self.update_construct(name)
            if not self.check_scope_and_pid(self.scope, name, pid):
                return
            data_info = {}
            if self.config.task != Const.STRUCTURE:
                data_info = self.data_processor.analyze_backward(name, module, module_input_output)
            if self.config.level == Const.LEVEL_L2:
                return
            if data_info and name.split(Const.SEP)[0] in Const.MODULE_PREFIX:
                module_name = name.rsplit(Const.SEP, 2)[0]
                self.backward_module_names[module_name] = True
            self.handle_data(name, data_info, flush=self.data_processor.is_terminated)

        except Exception as e:
            error_type = type(e).__name__
            tb = traceback.format_exc()
            self.data_writer.write_error_log(
                f"[ERROR] backward_data_collect failed: name={name}, pid={pid}\n{tb}",
                error_type=error_type
            )

    def backward_input_data_collect(self, name, module, pid, module_input_output):
        try:
            self.update_construct(name)
            if not self.check_scope_and_pid(self.scope, name, pid):
                return
            data_info = {}
            if self.config.task != Const.STRUCTURE:
                data_info = self.data_processor.analyze_backward_input(name, module, module_input_output)
            self.handle_data(name, data_info)

        except Exception as e:
            error_type = type(e).__name__
            tb = traceback.format_exc()
            self.data_writer.write_error_log(
                f"[ERROR] backward_input_data_collect failed: name={name}, pid={pid}\n{tb}",
                error_type=error_type
            )

    def backward_output_data_collect(self, name, module, pid, module_input_output):
        try:
            self.update_construct(name)
            if not self.check_scope_and_pid(self.scope, name, pid):
                return
            data_info = {}
            if self.config.task != Const.STRUCTURE:
                data_info = self.data_processor.analyze_backward_output(name, module, module_input_output)
            self.handle_data(name, data_info)

        except Exception as e:
            error_type = type(e).__name__
            tb = traceback.format_exc()
            self.data_writer.write_error_log(
                f"[ERROR] backward_output_data_collect failed: name={name}, pid={pid}\n{tb}",
                error_type=error_type
            )

    def update_construct(self, name):
        if self.config.level not in DataCollector.level_without_construct:
            if self.optimizer_status in [Const.OPTIMIZER, Const.CLIP_GRAD]:
                if self.optimizer_status_first_start[self.optimizer_status]:
                    self.data_writer.update_construct(
                        {self.optimizer_status: None if not is_megatron() else [None, get_micro_step()]})
                    self.optimizer_status_first_start[self.optimizer_status] = False
                self.data_writer.update_construct(
                    {name: self.optimizer_status if not is_megatron() else [self.optimizer_status, get_micro_step()]})
            else:
                if self.config.level == Const.LEVEL_MIX and \
                  not (name.startswith(Const.MODULE) or name.startswith(Const.CELL)):
                    self.data_writer.update_construct(
                        {name: self.module_processor.api_parent_node.get(threading.get_ident())}
                    )
            if MegatronStepInfo.is_megatron:
                micro_step_number = max(MegatronStepInfo.forward_micro_step, MegatronStepInfo.backward_micro_step)
                self.data_writer.update_construct({Const.MEGATRON_MICRO_STEP_NUMBER: micro_step_number})

            self.data_writer.update_construct(self.module_processor.module_node)

    def handle_data(self, name, data_info, flush=False):
        if data_info:
            self.update_data(name, data_info)
        if self.config.async_dump:
            return
        if not flush:
            self.data_writer.flush_data_periodically()
        else:
            self.write_json()

    def update_dump_paths(self, *args):
        self.data_writer.update_dump_paths(*args)

    def initialize_json_file(self, framework=Const.UNKNOWN_FRAMEWORK):
        self.data_writer.initialize_json_file(task=self.config.task, level=self.config.level, framework=framework)

    def update_iter(self, current_iter):
        self.data_processor.update_iter(current_iter)

    def params_data_collect(self, name, param_name, pid, data):
        grad_name = name + Const.SEP + Const.PARAMS_GRAD
        self.update_api_or_module_name(grad_name)
        if not self.check_scope_and_pid(self.scope, name, pid) and not self.backward_module_names.get(name):
            if self.data_writer.cache_data.get("data"):
                self.data_writer.cache_data.get("data").pop(grad_name, None)
                self.params_grad_record[grad_name] = False
            return
        data_info = self.data_processor.analyze_params(grad_name, param_name, data)
        self.handle_data(grad_name, data_info, flush=self.data_processor.is_terminated)
        self.params_grad_record[grad_name] = False

    def params_data_collect_in_bw_hook(self, params_dict, name):
        try:
            if not params_dict:
                return
            ori_name = name.rsplit(Const.SEP, 2)[0]
            for param_name, param in params_dict.items():
                grad_name = ori_name + Const.SEP + Const.PARAMS_GRAD
                self.update_api_or_module_name(grad_name)
                if self.params_grad_record.get(grad_name, False):
                    grad = param.grad if hasattr(param, "grad") else None
                    data_info = self.data_processor.analyze_params(grad_name, param_name, grad)
                    self.handle_data(grad_name, data_info, flush=self.data_processor.is_terminated)
        except Exception as e:
            error_type = type(e).__name__
            tb = traceback.format_exc()
            self.data_writer.write_error_log(
                f"[ERROR] params_data_collect_in_bw_hook failed: "
                f"name={name}",
                error_type=error_type
            )

    def debug_data_collect_forward(self, variable, name_with_count):
        data_info = self.data_processor.analyze_debug_forward(variable, name_with_count)
        name_with_count_category = name_with_count + Const.SEP + Const.DEBUG
        self.data_writer.update_debug({name_with_count_category: data_info})

    def debug_data_collect_backward(self, variable, grad_name_with_count):
        # prepare all None nested data structure
        all_none_data_info = self.data_processor.analyze_element_to_all_none(variable)
        grad_name_with_count_category = grad_name_with_count + Const.SEP + Const.DEBUG
        self.data_writer.update_debug({grad_name_with_count_category: all_none_data_info})

        # register tensor backward hook
        self.data_processor.analyze_debug_backward(variable, grad_name_with_count_category,
                                                   self.data_writer.cache_debug['data'])
