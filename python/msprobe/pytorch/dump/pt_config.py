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

import os

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_json
from msprobe.core.dump.common_config import BaseConfig, CommonConfig
from msprobe.pytorch.dump.api_dump.utils import get_ops


class TensorConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.check_config()
        self._check_summary_mode()
        self._check_file_format()

    def _check_file_format(self):
        if self.file_format is not None and self.file_format not in ["npy", "bin"]:
            raise Exception("file_format is invalid")


class StatisticsConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.check_config()
        self._check_summary_mode()

        self.tensor_list = json_config.get("tensor_list", [])
        self._check_str_list_config(self.tensor_list, "tensor_list")


class StructureConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)


TaskDict = {
    Const.TENSOR: TensorConfig,
    Const.STATISTICS: StatisticsConfig,
    Const.STRUCTURE: StructureConfig
}


def parse_task_config(task, json_config):
    task_map = json_config.get(task, dict())
    return TaskDict.get(task)(task_map)


def parse_json_config(json_file_path, task):
    if not json_file_path:
        config_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        json_file_path = os.path.join(config_dir, "../../config.json")
    json_config = load_json(json_file_path)
    common_config = CommonConfig(json_config)
    if task:
        task_config = parse_task_config(task, json_config)
    else:
        task_config = parse_task_config(common_config.task, json_config)
    return common_config, task_config


class RunUTConfig(BaseConfig):
    WrapApi = get_ops()

    def __init__(self, json_config):
        super().__init__(json_config)
        self.white_list = json_config.get("white_list", Const.DEFAULT_LIST)
        self.black_list = json_config.get("black_list", Const.DEFAULT_LIST)
        self.error_data_path = json_config.get("error_data_path", Const.DEFAULT_PATH)

        self.check_run_ut_config()

    @classmethod
    def check_filter_list_config(cls, key, filter_list):
        if not isinstance(filter_list, list):
            raise Exception("%s must be a list type" % key)
        if not all(isinstance(item, str) for item in filter_list):
            raise Exception("All elements in %s must be string type" % key)
        invalid_api = [item for item in filter_list if item not in cls.WrapApi]
        if invalid_api:
            raise Exception("Invalid api in %s: %s" % (key, invalid_api))

    @classmethod
    def check_error_data_path_config(cls, error_data_path):
        if not os.path.exists(error_data_path):
            raise Exception("error_data_path: %s does not exist" % error_data_path)

    def check_run_ut_config(self):
        RunUTConfig.check_filter_list_config(Const.WHITE_LIST, self.white_list)
        RunUTConfig.check_filter_list_config(Const.BLACK_LIST, self.black_list)
        RunUTConfig.check_error_data_path_config(self.error_data_path)

