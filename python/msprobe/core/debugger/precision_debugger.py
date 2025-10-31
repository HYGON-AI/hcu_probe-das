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

import os

from msprobe.core.common.const import Const, FileCheckConst, MsgConst
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.file_utils import FileChecker, load_json
from msprobe.core.common.utils import get_real_step_or_rank, check_init_step, ThreadSafe
from msprobe.core.common_config import CommonConfig


class BasePrecisionDebugger:
    _instance = None
    tasks_not_need_debugger = [Const.GRAD_PROBE]

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with ThreadSafe():
                if not cls._instance:
                    cls._instance = super(BasePrecisionDebugger, cls).__new__(cls)
                    cls._instance.config = None
                    cls._instance.initialized = False
                    cls.service = None
                    cls.first_start = False
        return cls._instance

    def __init__(
            self,
            config_path=None,
            task=None,
            dump_path=None,
            level=None,
            step=None
    ):
        if self.initialized:
            return
        self.initialized = True
        self._check_input_params(config_path, task, dump_path, level)
        self.common_config, self.task_config = self._parse_config_path(config_path, task)
        self.task = self.common_config.task
        if step is not None:
            self.common_config.step = get_real_step_or_rank(step, Const.STEP)

    @staticmethod
    def _check_input_params(config_path, task, dump_path, level):
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), "../../config.json")
        if config_path is not None:
            if not isinstance(config_path, str):
                raise MsprobeException(
                    MsprobeException.INVALID_PARAM_ERROR, f"config_path must be a string")
            file_checker = FileChecker(
                file_path=config_path, path_type=FileCheckConst.FILE, file_type=FileCheckConst.JSON_SUFFIX)
            file_checker.common_check()

        if task is not None and task not in Const.TASK_LIST:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, f"task must be one of {Const.TASK_LIST}")

        if dump_path is not None:
            if not isinstance(dump_path, str):
                raise MsprobeException(
                    MsprobeException.INVALID_PARAM_ERROR, f"dump_path must be a string")

        if level is not None and level not in Const.LEVEL_LIST:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, f"level must be one of {Const.LEVEL_LIST}")

    @staticmethod
    def _get_task_config(task, json_config):
        raise NotImplementedError("Subclass must implement _get_task_config")

    @classmethod
    @ThreadSafe.synchronized
    def forward_backward_dump_end(cls):
        instance = cls._instance
        instance.stop()

    @classmethod
    @ThreadSafe.synchronized
    def set_init_step(cls, step):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        check_init_step(step)
        instance.service.init_step = step
        instance.service.loop = 0

    @classmethod
    @ThreadSafe.synchronized
    def register_custom_api(cls, module, api, api_prefix=None):
        if not api_prefix:
            api_prefix = getattr(module, "__name__", "Custom")
        if not isinstance(api_prefix, str):
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, "api_prefix must be string")
        if not hasattr(module, api):
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, f"module {str(module)} does not have {api}")
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        instance.service.register_custom_api(module, api, api_prefix)

    @classmethod
    @ThreadSafe.synchronized
    def restore_custom_api(cls, module, api):
        if not hasattr(module, api):
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, f"module {str(module)} does not have {api}")
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        instance.service.restore_custom_api(module, api)

    @classmethod
    def _get_instance(cls):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if instance.task in BasePrecisionDebugger.tasks_not_need_debugger:
            instance = None
        return instance

    def _parse_config_path(self, json_file_path, task):
        if not json_file_path:
            json_file_path = os.path.join(os.path.dirname(__file__), "../../config.json")
        json_config = load_json(json_file_path)
        common_config = CommonConfig(json_config)
        if task:
            task_config = self._get_task_config(task, json_config)
        else:
            if not common_config.task:
                common_config.task = Const.STATISTICS
            task_config = self._get_task_config(common_config.task, json_config)
        return common_config, task_config
