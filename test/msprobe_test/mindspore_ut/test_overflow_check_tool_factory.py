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

from unittest import TestCase
from unittest.mock import patch

from msprobe.core.common.log import logger
from msprobe.core.dump.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.common.const import Const
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.overflow_check.overflow_check_tool_factory import OverflowCheckToolFactory


class TestOverflowCheckToolFactory(TestCase):
    @patch.object(logger, "error")
    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_create(self, _, mock_logger_error):
        json_config = {
            "task": "overflow_check",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)

        config.level = "module"
        with self.assertRaises(Exception) as context:
            OverflowCheckToolFactory.create(config)
        self.assertEqual(str(context.exception), "Valid level is needed.")

        config.execution_mode = Const.GRAPH_GE_MODE
        config.level = "cell"
        with self.assertRaises(ValueError):
            OverflowCheckToolFactory.create(config)
        mock_logger_error.assert_called_with(f"Overflow check is not supported in {config.execution_mode} mode "
                                             f"when level is {config.level}.")

        config.level = "kernel"
        dumper = OverflowCheckToolFactory.create(config)[0]
        self.assertEqual(dumper.dump_json["common_dump_settings"]["file_format"], "npy")
