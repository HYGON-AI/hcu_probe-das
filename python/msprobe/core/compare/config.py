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

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.file_utils import load_yaml


class ModeConfig:
    def __init__(self, **kwargs):
        self.stack_mode = kwargs.get('stack_mode', False)
        self.auto_analyze = kwargs.get('auto_analyze', True)
        self.fuzzy_match = kwargs.get('fuzzy_match', False)
        self.highlight = kwargs.get('highlight', False)
        self.dump_mode = kwargs.get('dump_mode', Const.SUMMARY)
        self.first_diff_analyze = kwargs.get('first_diff_analyze', False)
        self.diff_analyze = kwargs.get('diff_analyze', False)
        self.compared_file_type = kwargs.get('compared_file_type', Const.DUMP_JSON_FILE)


class MappingConfig:
    def __init__(self, cell_mapping=None, api_mapping=None, data_mapping=None):
        self.cell_mapping = cell_mapping
        self.api_mapping = api_mapping
        self.data_mapping = data_mapping


class MappingDict:
    def __init__(self, mapping_config: MappingConfig):
        self.cell_mapping_dict = self.load_mapping_file(mapping_config.cell_mapping)
        self.api_mapping_dict = self.load_mapping_file(mapping_config.api_mapping)
        if mapping_config.api_mapping is not None:
            self.ms_to_pt_mapping = self.load_internal_api()
        self.data_mapping_dict = self.init_data_mapping(mapping_config.data_mapping)

    @staticmethod
    def load_internal_api():
        cur_path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.abspath(os.path.join(cur_path, CompareConst.INTERNAL_API_MAPPING_FILE))
        return load_yaml(yaml_path)

    @staticmethod
    def load_mapping_file(mapping_file):
        if isinstance(mapping_file, str):
            mapping_dict = load_yaml(mapping_file)
        else:
            mapping_dict = {}
        return mapping_dict

    def init_data_mapping(self, data_mapping):
        """
        初始化data_mapping_dict
        """
        if isinstance(data_mapping, str) or data_mapping is None:
            data_mapping_dict = self.load_mapping_file(data_mapping)
        elif isinstance(data_mapping, dict):
            data_mapping_dict = data_mapping
        else:
            raise TypeError(f"The type of parameter `data_mapping` must be dict, str or None, but got "
                            f"{type(data_mapping)}")
        return data_mapping_dict
