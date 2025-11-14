# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

import os
import sys

from unittest import TestCase
from unittest.mock import patch

from msprobe.core.dump.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.exception_dump.kernel_graph_exception_dump import KernelGraphExceptionDump
from msprobe.core.common.file_utils import move_file


class TestKernelGraphExceptionDump(TestCase):
    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle(self, _):
        json_config = {
            "task": "exception_dump",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        task_config.check_mode = "atomic"
        config = DebuggerConfig(common_config, task_config)
        checker = KernelGraphExceptionDump(config)
        self.assertEqual(checker.dump_json["common_dump_settings"]["op_debug_mode"], 4)

        _msprobe_c_existed = True
        try:
            from msprobe.lib import _msprobe_c
        except ImportError:
            _msprobe_c_existed = False

        with patch("msprobe.mindspore.dump.exception_dump.kernel_graph_exception_dump.create_directory"), \
             patch("msprobe.mindspore.dump.exception_dump.kernel_graph_exception_dump.logger.info"), \
             patch("msprobe.mindspore.dump.exception_dump.kernel_graph_exception_dump.save_json") as mock_save_json:

            checker.handle()
            self.assertIn("kernel_graph_exception_check.json", mock_save_json.call_args_list[0][0][0])
            self.assertIn("kernel_graph_exception_check.json", os.environ.get("MINDSPORE_DUMP_CONFIG"))

        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]