#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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
from collections import namedtuple
from msprobe.core.common.file_utils import load_yaml, check_file_or_directory_path
from msprobe.core.common.utils import is_int
from msprobe.pytorch.dump.pt_config import ACCCheckConfig


AccCheckConfig = namedtuple('AccCheckConfig', ['forward_content', 'backward_content',
                                               'result_csv_path', 'details_csv_path',
                                         'save_error_data', 'is_continue_acc_check', 'real_data_path', 'white_list',
                                         'black_list', 'error_data_path'])


class Config:
    def __init__(self, yaml_file):
        check_file_or_directory_path(yaml_file, False)
        config = load_yaml(yaml_file)
        self.config = {key: self.validate(key, value) for key, value in config.items()}

    def __getattr__(self, item):
        return self.config[item]

    def __str__(self):
        return '\n'.join(f"{key}={value}" for key, value in self.config.items())

    @staticmethod
    def validate(key, value):
        validators = {
            'white_list': list,
            'black_list': list,
            'error_data_path': str,
            'precision': int
        }
        if key not in validators:
            raise ValueError(f"{key} must be one of {validators.keys()}")
        if not isinstance(value, validators.get(key)):
            raise ValueError(f"{key} must be {validators[key].__name__} type")
        if key == 'precision' and not is_int(value):
            raise ValueError("precision must be an integer")
        if key == 'precision' and (value < 0 or value > 20):
            raise ValueError("precision must be greater than or equal to 0 and less than 21")
        if key == 'white_list':
            ACCCheckConfig.check_filter_list_config(key, value)
        if key == 'black_list':
            ACCCheckConfig.check_filter_list_config(key, value)
        if key == 'error_data_path':
            ACCCheckConfig.check_error_data_path_config(value)
        return value


cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
yaml_path = os.path.join(cur_path, "config.yaml")
msCheckerConfig = Config(yaml_path)


class CheckerConfig:
    def __init__(self, task_config=None):
        self.white_list = msCheckerConfig.white_list
        self.black_list = msCheckerConfig.black_list
        self.error_data_path = msCheckerConfig.error_data_path

        if task_config:
            self.load_config(task_config)

    def load_config(self, task_config):
        self.white_list = task_config.white_list
        self.black_list = task_config.black_list
        self.error_data_path = task_config.error_data_path
    

    def get_acc_check_config(self, **config_params):
        return AccCheckConfig(
            forward_content=config_params.get('forward_content'),
            backward_content=config_params.get('backward_content'),
            result_csv_path=config_params.get('result_csv_path'),
            details_csv_path=config_params.get('details_csv_path'),
            save_error_data=config_params.get('save_error_data'),
            is_continue_acc_check=config_params.get('is_continue_acc_check'),
            real_data_path=config_params.get('real_data_path'),
            white_list=self.white_list.copy() if self.white_list else [],
            black_list=self.black_list.copy() if self.black_list else [],
            error_data_path=config_params.get('error_data_path')
        )
