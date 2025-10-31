# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import re

from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.utils import get_real_step_or_rank


class CommonConfig:
    def __init__(self, json_config):
        self.task = json_config.get('task')
        self.dump_path = json_config.get('dump_path')
        self.rank = get_real_step_or_rank(json_config.get('rank'), Const.RANK)
        self.step = get_real_step_or_rank(json_config.get('step'), Const.STEP)
        self.level = json_config.get('level')
        self.enable_dataloader = json_config.get('enable_dataloader', False)
        self.async_dump = json_config.get("async_dump", False)
        self.precision = json_config.get("precision", Const.DUMP_PRECISION_LOW)
        self._check_config()

    def _check_config(self):
        if self.task and self.task not in Const.TASK_LIST:
            logger.error_log_with_exp("task is invalid, it should be one of {}".format(Const.TASK_LIST),
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if self.dump_path is not None and not isinstance(self.dump_path, str):
            logger.error_log_with_exp("dump_path is invalid, it should be a string",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if self.level and self.level not in Const.LEVEL_LIST:
            logger.error_log_with_exp("level is invalid, it should be one of {}".format(Const.LEVEL_LIST),
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if not isinstance(self.enable_dataloader, bool):
            logger.error_log_with_exp("enable_dataloader is invalid, it should be a boolean",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if not isinstance(self.async_dump, bool):
            logger.error_log_with_exp("async_dump is invalid, it should be a boolean",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        elif self.async_dump:
            logger.warning("async_dump is True, it may cause OOM when dumping large tensor.")

        if self.precision not in Const.DUMP_PRECISION_LIST:
            logger.error_log_with_exp("precision is invalid, it should be one of {}".format(Const.DUMP_PRECISION_LIST),
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))


class BaseConfig:
    def __init__(self, json_config):
        self.scope = json_config.get('scope')
        self.list = json_config.get('list')
        self.data_mode = json_config.get('data_mode')
        self.file_format = json_config.get("file_format")
        self.summary_mode = json_config.get("summary_mode")
        self.overflow_nums = json_config.get("overflow_nums")
        self.check_mode = json_config.get("check_mode")
        self.fuzz_device = json_config.get("fuzz_device")
        self.pert_mode = json_config.get("pert_mode")
        self.handler_type = json_config.get("handler_type")
        self.fuzz_level = json_config.get("fuzz_level")
        self.fuzz_stage = json_config.get("fuzz_stage")
        self.if_preheat = json_config.get("if_preheat")
        self.preheat_step = json_config.get("preheat_step")
        self.max_sample = json_config.get("max_sample")
        self.is_regex_valid = True

    @staticmethod
    def _check_str_list_config(config_item, config_name):
        if config_item is not None:
            if not isinstance(config_item, list):
                logger.error_log_with_exp(f"{config_name} is invalid, it should be a list[str]",
                                          MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
            for name in config_item:
                if not isinstance(name, str):
                    logger.error_log_with_exp(f"{config_name} is invalid, it should be a list[str]",
                                              MsprobeException(MsprobeException.INVALID_PARAM_ERROR))

    def check_config(self):
        self._check_str_list_config(self.scope, "scope")
        self._check_str_list_config(self.list, "list")
        self._check_data_mode()
        self._check_regex_in_list()

    def _check_data_mode(self):
        if self.data_mode is not None:
            if not isinstance(self.data_mode, list):
                logger.error_log_with_exp("data_mode is invalid, it should be a list[str]",
                                          MsprobeException(MsprobeException.INVALID_PARAM_ERROR))

            if Const.ALL in self.data_mode and len(self.data_mode) != 1:
                logger.error_log_with_exp(
                    "'all' cannot be combined with other options in data_mode.",
                    MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
                )

            if len(self.data_mode) >= len(Const.DUMP_DATA_MODE_LIST):
                logger.error_log_with_exp(
                    f"The number of elements in the data_made cannot exceed {len(Const.DUMP_DATA_MODE_LIST) - 1}.",
                    MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
                )

            for mode in self.data_mode:
                if not isinstance(mode, str):
                    logger.error_log_with_exp("data_mode is invalid, it should be a list[str]",
                                              MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
                if mode not in Const.DUMP_DATA_MODE_LIST:
                    logger.error_log_with_exp(
                        f"The element '{mode}' of data_mode {self.data_mode} is not in {Const.DUMP_DATA_MODE_LIST}.",
                        MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
                    )

    def _check_summary_mode(self):
        if self.summary_mode and self.summary_mode not in Const.SUMMARY_MODE:
            logger.error_log_with_exp(
                        f"summary_mode is invalid, summary_mode is not in {Const.SUMMARY_MODE}.",
                        MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
                    )

    def _check_regex_in_list(self):
        if self.list:
            for name in self.list:
                if name.startswith('name-regex(') and name.endswith(')'):
                    try:
                        re.compile(name[len('name-regex('):-1])
                    except re.error:
                        self.is_regex_valid = False
                        break
