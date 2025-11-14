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

import unittest
from unittest.mock import patch

from msprobe.core.common.const import Const
from msprobe.mindspore.dump.ms_config import (parse_task_config,
                                         TensorConfig, StatisticsConfig, OverflowCheckConfig)


class TestMsConfig(unittest.TestCase):

    def test_parse_task_config(self):
        mock_json_config = {
            "tensor": None,
            "statistics": None,
            "overflow_check": None,
            "free_benchmark": None
        }

        task_config = parse_task_config("tensor", mock_json_config)
        self.assertTrue(isinstance(task_config, TensorConfig))

        task_config = parse_task_config("statistics", mock_json_config)
        self.assertTrue(isinstance(task_config, StatisticsConfig))

        task_config = parse_task_config("overflow_check", mock_json_config)
        self.assertTrue(isinstance(task_config, OverflowCheckConfig))

        mock_json_config.update({"overflow_check": {"overflow_nums": 1}})
        mock_json_config.update({"overflow_check": {"check_mode": "core"}})
        with self.assertRaises(Exception) as context:
            task_config = parse_task_config("overflow_check", mock_json_config)
        self.assertEqual(str(context.exception), "check_mode is invalid")

        with self.assertRaises(Exception) as context:
            parse_task_config("unsupported_task", mock_json_config)
        self.assertEqual(str(context.exception), "task is invalid.")
