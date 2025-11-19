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

import unittest
from unittest.mock import patch

from msprobe.core.dump.kernel_dump.kernel_config import create_kernel_config_json


class TestPtKernelConfig(unittest.TestCase):
    @patch("msprobe.core.dump.kernel_dump.kernel_config.save_json")
    def test_create_kernel_config_json_with_rank(self, mock_save_json):
        dump_path = "./step0"
        cur_rank = 0
        kernel_config_path = create_kernel_config_json(dump_path, cur_rank)
        self.assertEqual(kernel_config_path, "./step0/kernel_config_0.json")
        config_info = {
            "dump": {
                "dump_list": [],
                "dump_path": dump_path,
                "dump_mode": "all",
                "dump_op_switch": "on"
            }
        }
        mock_save_json.assert_called_once_with(kernel_config_path, config_info, indent=4)

    @patch("msprobe.core.dump.kernel_dump.kernel_config.save_json")
    def test_create_kernel_config_json_without_rank(self, mock_save_json):
        dump_path = "./step0"
        cur_rank = ''
        kernel_config_path = create_kernel_config_json(dump_path, cur_rank)
        self.assertEqual(kernel_config_path, "./step0/kernel_config.json")
        config_info = {
            "dump": {
                "dump_list": [],
                "dump_path": dump_path,
                "dump_mode": "all",
                "dump_op_switch": "on"
            }
        }
        mock_save_json.assert_called_once_with(kernel_config_path, config_info, indent=4)
