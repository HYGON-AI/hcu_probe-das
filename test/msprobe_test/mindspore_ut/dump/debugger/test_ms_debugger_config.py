# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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
from msprobe.core.common.log import logger
from msprobe.core.dump.common_config import CommonConfig
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.ms_config import StatisticsConfig


class TestDebuggerConfig(unittest.TestCase):
    @patch.object(logger, "error")
    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_init(self, _, mock_logger_error):
        json_config = {
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L2"
        }
        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        debugger_config = DebuggerConfig(common_config, task_config)
        self.assertEqual(debugger_config.task, Const.STATISTICS)
        self.assertEqual(debugger_config.file_format, "npy")
        self.assertEqual(debugger_config.check_mode, "all")
        self.assertEqual(debugger_config.tensor_list, [])
