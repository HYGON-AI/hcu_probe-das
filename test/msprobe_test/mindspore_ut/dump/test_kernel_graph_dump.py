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
import sys

from unittest import TestCase
from unittest.mock import patch

from msprobe.core.dump.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_processor.kernel_graph_dump import KernelGraphDump
from msprobe.core.common.file_utils import move_file


class TestKernelGraphDump(TestCase):
    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        task_config.data_mode = ["output"]
        task_config.file_format = "bin"
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelGraphDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")
        self.assertEqual(dumper.dump_json["common_dump_settings"]["file_format"], "bin")
        self.assertEqual(dumper.dump_json["common_dump_settings"]["input_output"], 2)

        _msprobe_c_existed = True
        try:
            from msprobe.lib import _msprobe_c
        except ImportError:
            _msprobe_c_existed = False

        with patch("msprobe.mindspore.dump.dump_processor.kernel_graph_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_graph_dump.logger.info"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_graph_dump.save_json") as mock_save_json:

            if _msprobe_c_existed:
                dumper.handle()
                mock_save_json.assert_not_called()

                _msprobe_c_path = _msprobe_c.__file__
                _msprobe_c_test_path = _msprobe_c_path.replace('_msprobe_c.so', '_msprobe_c_test.so')
                move_file(_msprobe_c_path, _msprobe_c_test_path)
                sys.modules.pop('msprobe.lib')
                sys.modules.pop('msprobe.lib._msprobe_c')

            os.environ["GRAPH_OP_RUN"] = "1"
            with self.assertRaises(Exception) as context:
                dumper.handle()
            self.assertEqual(str(context.exception), "Must run in graph mode, not kbk mode")
            if "GRAPH_OP_RUN" in os.environ:
                del os.environ["GRAPH_OP_RUN"]

            dumper.handle()
            self.assertIn("kernel_graph_dump.json", mock_save_json.call_args_list[0][0][0])
            self.assertIn("kernel_graph_dump.json", os.environ.get("MS_ACL_DUMP_CFG_PATH"))

        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]
        if "MS_ACL_DUMP_CFG_PATH" in os.environ:
            del os.environ["MS_ACL_DUMP_CFG_PATH"]
        if _msprobe_c_existed:
            move_file(_msprobe_c_test_path, _msprobe_c_path)
