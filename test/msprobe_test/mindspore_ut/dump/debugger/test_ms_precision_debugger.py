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
from unittest.mock import patch, MagicMock

from msprobe.core.common.const import Const, MsgConst
from msprobe.core.common.runtime import Runtime
from msprobe.core.dump.common_config import CommonConfig
from msprobe.core.dump.debugger.precision_debugger import BasePrecisionDebugger
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.dump.cell_processor import CellProcessor
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.debugger.precision_debugger import PrecisionDebugger
from msprobe.mindspore.dump.dump_processor.hook_cell.hook_cell import HOOKCell
from msprobe.mindspore.dump.ms_config import StatisticsConfig


class TestPrecisionDebugger(unittest.TestCase):

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_start(self, _):
        PrecisionDebugger._instance = None

        class Handler:
            called = False

            def handle(self):
                Handler.called = True

        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L1",
            "async_dump": False
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        handler = Handler()

        mock_get_mode = MagicMock()
        mock_parse_json_config = MagicMock()
        with patch.object(BasePrecisionDebugger, "_parse_config_path", new=mock_parse_json_config), \
             patch.object(PrecisionDebugger, "_get_execution_mode", new=mock_get_mode), \
             patch("msprobe.mindspore.dump.debugger.precision_debugger.TaskHandlerFactory.create", return_value=handler), \
             patch("msprobe.mindspore.dump.debugger.precision_debugger.set_register_backward_hook_functions"):
            mock_get_mode.return_value = MsConst.GRAPH_GE_MODE
            mock_parse_json_config.return_value = [common_config, task_config]
            debugger = PrecisionDebugger()
            self.assertEqual(Runtime.step_count, 0)
            self.assertFalse(Runtime.is_running)
            debugger.start()
            self.assertTrue(Runtime.is_running)
            self.assertTrue(isinstance(debugger.config, DebuggerConfig))
            self.assertTrue(Handler.called)

            mock_get_mode.return_value = MsConst.PYNATIVE_MODE
            with patch("msprobe.mindspore.dump.debugger.precision_debugger.MindsporeService") as mock_Service, \
                 patch("msprobe.mindspore.dump.debugger.precision_debugger.set_register_backward_hook_functions"):
                debugger = PrecisionDebugger()
                debugger.start()
                service = mock_Service.return_value
                mock_Service.assert_called_with(debugger.config)
                service.start.assert_called_with(None, None)

        PrecisionDebugger._instance = None
        with self.assertRaises(Exception) as context:
            debugger.start()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)

    def test_stop_step(self):
        class MockPrecisionDebugger:
            def __init__(self):
                self.task = Const.TENSOR
                self.service = None
                self.config = MagicMock()
                self.config.level_ori = MagicMock()
                self.config.level_ori.return_value = Const.LEVEL_L1
        PrecisionDebugger._instance = None
        with self.assertRaises(Exception) as context:
            PrecisionDebugger.stop()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)
        with self.assertRaises(Exception) as context:
            PrecisionDebugger.step()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)
        PrecisionDebugger._instance = MockPrecisionDebugger()
        Runtime.is_running = True
        PrecisionDebugger.stop()
        self.assertFalse(Runtime.is_running)
        Runtime.step_count = 0
        PrecisionDebugger.step()
        self.assertEqual(Runtime.step_count, 1)
        Runtime.step_count = 0

        HOOKCell.cell_count["api"] = 1
        PrecisionDebugger.step()
        self.assertEqual(HOOKCell.cell_count["api"], 0)

        with patch.object(CellProcessor, "reset_cell_stats") as mock_reset_cell:
            PrecisionDebugger.step()
        mock_reset_cell.assert_called_once()

    def test_is_graph_dump_empty_list(self):
        config = MagicMock()
        config.level = MsConst.KERNEL
        config.list = []
        result = PrecisionDebugger._is_graph_dump(config)
        self.assertTrue(result)

    def test_is_graph_dump_multiple_items_in_list(self):
        config = MagicMock()
        config.level = MsConst.KERNEL
        config.list = ["item1", "item2"]
        result = PrecisionDebugger._is_graph_dump(config)
        self.assertTrue(result)

    def test_is_graph_dump_single_item_with_slash_or_dash(self):
        config = MagicMock()
        config.level = MsConst.KERNEL
        config.list = ["item/with/slash"]
        result = PrecisionDebugger._is_graph_dump(config)
        self.assertTrue(result)
        config.list = ["item-with-dash"]
        result = PrecisionDebugger._is_graph_dump(config)
        self.assertTrue(result)

    def test_is_graph_dump_single_item_without_dash_or_slash(self):
        config = MagicMock()
        config.level = MsConst.KERNEL
        config.list = ["Functional.relu.1.forward"]
        result = PrecisionDebugger._is_graph_dump(config)
        self.assertFalse(result)
