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

import copy
import functools
import os
from abc import ABC, abstractmethod
from collections import defaultdict

from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.core.common.file_utils import create_directory
from msprobe.core.common.runtime import Runtime
from msprobe.core.common.utils import Const, print_tools_ends_info, DumpPathAggregation
from msprobe.core.data_dump.api_registry import ApiRegistry
from msprobe.core.data_dump.data_collector import build_data_collector
from msprobe.core.kernel_dump.kernel_config import create_kernel_config_json
from msprobe.core.common.megatron_utils import MegatronStepInfo


class BaseService(ABC):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.config.level = getattr(config, 'level_ori', config.level)  # 兼容MindSpore配置
        self.model = None
        self.data_collector = build_data_collector(self.config)
        self.current_iter = 0
        self.loop = 0
        self.init_step = 0
        self.cur_token_id = 0
        self.first_start = True
        self.primitive_switch = False
        self.current_rank = None
        self.dump_iter_dir = None
        self.should_stop_service = False
        self.hooked_modules = []
        self.ori_customer_func = {}
        self.debug_variable_counter = None
        self.currrent_step_first_debug_save = True
        self.logger = None  # 子类中注入
        self.api_register = None  # 子类中注入
        self.api_template = None  # 子类中注入
        self.hook_manager = None  # 子类中注入
        self._init_specific_components()
        self._register_api_hook()

    @property
    def _is_debug_level(self):
        return self.config.level == Const.LEVEL_DEBUG

    @property
    def _is_l2_level(self):
        return self.config.level == Const.LEVEL_L2

    @property
    def _is_mix_level(self):
        return self.config.level == Const.LEVEL_MIX

    @property
    def _is_need_module_hook(self):
        return self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L0]

    @property
    def _is_need_api_hook(self):
        return self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L1, Const.LEVEL_L2]

    @property
    def _is_no_dump_step(self):
        return (self.config.step and self.current_iter not in self.config.step)

    @property
    def _is_no_dump_rank(self):
        return (self.config.rank and self.current_rank not in self.config.rank)

    @property
    def _need_tensor_data(self):
        """判断是否需要采集tensor数据"""
        return bool(
            self.config.task in self.data_collector.tasks_need_tensor_data or
            (self.config.task == Const.STATISTICS and self.config.tensor_list)
        )

    @property
    @abstractmethod
    def _get_framework_type(self):
        """获取框架类型"""
        pass

    @staticmethod
    @abstractmethod
    def _get_current_rank():
        """获取当前rank_id"""
        pass

    @staticmethod
    def _change_jit_switch(status):
        """修改JitDump开关，mindspore子类重写"""
        pass

    def start(self, model=None, token_range=None):
        """通用start模板"""
        self._process_iteration()
        if self._is_debug_level:
            return
        if model:
            self.model = model
        if self._is_need_module_hook and self.model not in self.hooked_modules:
            self._register_module_hook()
            self.hooked_modules.append(self.model)
        if self._need_stop_service():
            return
        Runtime.is_running = True
        self.cur_token_id = 0
        if self.first_start:
            try:
                self.current_rank = self._get_current_rank()
            except DistributedNotInitializedError:
                self.current_rank = None
            Runtime.current_rank = self.current_rank
            if self._is_no_dump_rank:
                Runtime.is_running = False
                return
            self._register_hook()
            self.first_start = False

            if token_range:
                self._register_infer_count_hook(self.model, token_range)
        self.logger.info(f"{Const.TOOL_NAME}: debugger.start() is set successfully")
        if token_range is None:
            self.primitive_switch = True
            self._change_jit_switch(True)
            self.logger.info(f"Dump switch is turned on at step {self.current_iter}. ")

        self.create_dirs()
        self.logger.info(f"Dump data will be saved in {self.dump_iter_dir}.")

    def stop(self):
        """通用stop模板"""
        if self._is_debug_level or self.should_stop_service:
            return
        if self._is_no_dump_step or self._is_no_dump_rank:
            return
        self.logger.info(f"{Const.TOOL_NAME}: debugger.stop() is set successfully. "
                         "Please set debugger.start() to turn on the dump switch again. ")
        Runtime.is_running = False
        self.primitive_switch = False
        self._change_jit_switch(False)
        if self._is_l2_level:
            return

        self._process_async_dump()
        self.data_collector.write_json()

    def step(self):
        """通用step处理"""
        if self.should_stop_service:
            return
        self._process_async_dump()
        self.data_collector.write_json()
        self.currrent_step_first_debug_save = True
        self.loop += 1
        self._reset_status()
        MegatronStepInfo.reset()

    def save(self, variable, name, save_backward):
        '''
        Args:
            variable: Union[List[variable], dict{str: variable}, mindspore.tensor, str, float, int]
            name: str
            save_backward: boolean
        Return:
            void
        '''
        if not self._is_debug_level:
            return
        self.current_iter = self.loop + self.init_step
        if self._is_no_dump_step:
            return

        if self.currrent_step_first_debug_save:
            try:
                self.current_rank = self._get_current_rank()
            except DistributedNotInitializedError:
                self.current_rank = None

            self.create_dirs()
            self.debug_variable_counter = defaultdict(int)
            self.currrent_step_first_debug_save = False

        count = self.debug_variable_counter[name]
        self.debug_variable_counter[name] += 1

        name_with_count = f"{name}.{count}"
        grad_name_with_count = f"{name}_grad.{count}"

        # forward save
        self.data_collector.debug_data_collect_forward(variable, name_with_count)

        # backward save
        if save_backward:
            self.data_collector.debug_data_collect_backward(variable, grad_name_with_count)

    def register_custom_api(self, module, api_name, api_prefix):
        self.ori_customer_func[str(module) + Const.SEP + api_name] = getattr(module, api_name)
        ApiRegistry.register_custom_api(module, api_name, api_prefix,
                                        functools.partial(self.build_hook, Const.API), self.api_template)

    def restore_custom_api(self, module, api):
        ori_func = self.ori_customer_func.get(str(module) + Const.SEP + api)
        if ori_func:
            setattr(module, api, ori_func)

    def build_hook(self, hook_type, name):
        return self.hook_manager.build_hook(hook_type, name)

    def create_dirs(self):
        """统一目录创建逻辑"""
        create_directory(self.config.dump_path)
        if Runtime.run_mode == Const.PYNATIVE_GRAPH_MODE:
            self.dump_iter_dir = os.path.join(self.config.dump_path, Const.PYNATIVE_MODE, f"step{self.current_iter}")
        else:
            self.dump_iter_dir = os.path.join(self.config.dump_path, f"step{self.current_iter}")

        cur_rank = self.current_rank if self.current_rank is not None else ''
        if self._is_l2_level:
            self._create_l2_dirs(cur_rank)
        else:
            self._create_default_dirs(cur_rank)

    @abstractmethod
    def _init_specific_components(self):
        """初始化框架特定组件"""
        pass

    @abstractmethod
    def _register_hook(self):
        """注册hook函数"""
        pass

    @abstractmethod
    def _register_module_hook(self):
        """注册模块级别的hook函数"""

    def _need_stop_service(self):
        if self.should_stop_service:
            return True
        end_service = self.config.step and self.current_iter > max(self.config.step) or \
            self.data_collector and self.data_collector.data_processor.is_terminated
        if end_service:
            self.primitive_switch = False
            self._change_jit_switch(False)
            Runtime.is_running = False
            self.should_stop_service = True
            print_tools_ends_info()
            return True
        if self._is_no_dump_step:
            return True
        return False

    def _register_api_hook(self):
        if self._is_need_api_hook:
            self.api_register.initialize_hook(functools.partial(self.build_hook, Const.API))
            self.api_register.register_all_api()
            self.logger.info(f"The api {self.config.task} hook function is successfully mounted to the model.")

    def _register_infer_count_hook(self, root_model, token_range):
        """
        通过root_model执行的轮次来判断当前在第几个token
        param root_model: 需要采集的推理模型
        param token_range: [start, end], 采集infer的token循环范围，左右皆包含在内
        return: None
        """
        def infer_hook(model, args):
            if self.cur_token_id == token_range[0]:
                Runtime.is_running = True
                self.primitive_switch = True
                self._change_jit_switch(True)
                self.logger.info(f"Current token id: {self.cur_token_id}, start dump infer token.")
            elif token_range[0] < self.cur_token_id <= token_range[1]:
                self.logger.debug(f"Current token id: {self.cur_token_id}.")
            elif self.cur_token_id == token_range[1] + 1:
                Runtime.is_running = False
                self.primitive_switch = False
                self._change_jit_switch(False)
                self.logger.info(
                    f"Current token id: {self.cur_token_id}, exceed token_range, early stop dump infer token.")
            self.cur_token_id += 1
        # 此处root_model可以保证为 Module/Cell类型 或 [Module/Cell]类型
        if root_model and isinstance(root_model, list):
            root_model = root_model[0]
            self.logger.warning("Infer model can only input one to support token_range, choose the first one.")

        root_model.register_forward_pre_hook(infer_hook)

    def _create_l2_dirs(self, cur_rank):
        create_directory(self.dump_iter_dir)
        kernel_config_path = create_kernel_config_json(self.dump_iter_dir, cur_rank)
        self.config.kernel_config_path = kernel_config_path

    def _create_default_dirs(self, cur_rank):
        dump_dir = os.path.join(self.dump_iter_dir, f"rank{cur_rank}")
        create_directory(dump_dir)

        dump_data_dir = None
        if self._need_tensor_data:
            dump_data_dir = os.path.join(dump_dir, "dump_tensor_data")
            create_directory(dump_data_dir)

        self._configure_dump_paths(dump_dir, dump_data_dir)

    def _configure_dump_paths(self, dump_dir, dump_data_dir):
        dump_path_aggregation = DumpPathAggregation()
        dump_path_aggregation.dump_file_path = os.path.join(dump_dir, "dump.json")
        dump_path_aggregation.stack_file_path = os.path.join(dump_dir, "stack.json")
        dump_path_aggregation.construct_file_path = os.path.join(dump_dir, "construct.json")
        dump_path_aggregation.dump_error_info_path = os.path.join(dump_dir, "dump_error_info.log")
        dump_path_aggregation.dump_tensor_data_dir = dump_data_dir
        dump_path_aggregation.debug_file_path = os.path.join(dump_dir, "debug.json")
        dump_path_aggregation.free_benchmark_file_path = os.path.join(dump_dir, "free_benchmark.csv")
        self.data_collector.update_dump_paths(dump_path_aggregation)
        self.data_collector.initialize_json_file(self._get_framework_type)

    def _process_iteration(self):
        """处理迭代计数"""
        self.current_iter = self.loop + self.init_step
        self.data_collector.update_iter(self.current_iter)
        Runtime.current_iter = self.current_iter

    def _process_async_dump(self):
        """处理异步dump逻辑"""
        if self.config.async_dump and self.config.task in [Const.STATISTICS, Const.TENSOR]:
            self.data_collector.data_processor.dump_async_data()

    def _reset_status(self):
        """通用状态重置"""
        self.data_collector.reset_status()
        self.hook_manager.reset_status()
        if self._is_l2_level:
            self.data_collector.data_processor.reset_status()
