# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from unittest import TestCase
from unittest.mock import patch

from msprobe.core.common.log import logger
from msprobe.core.dump.common_config import CommonConfig, BaseConfig
from msprobe.core.common.const import Const as CoreConst
from msprobe.mindspore.common.const import Const
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_processor.dump_tool_factory import DumpToolFactory
from msprobe.mindspore.dump.ms_config import StatisticsConfig


class TestDumpToolFactory(TestCase):
    @patch.object(logger, "error")
    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_create(self, _, mock_logger_error):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L1"
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        config = DebuggerConfig(common_config, task_config)

        config.data_mode = [CoreConst.INPUT, CoreConst.OUTPUT]
        with self.assertRaises(Exception) as context:
            DumpToolFactory.create(config)
        self.assertEqual(str(context.exception), "data_mode must be one of all, input, output.")

        config.data_mode = [CoreConst.FORWARD]
        with self.assertRaises(Exception) as context:
            DumpToolFactory.create(config)
        self.assertEqual(str(context.exception), "data_mode must be one of all, input, output.")

        config.data_mode = [CoreConst.ALL]
        config.level = "module"
        with self.assertRaises(Exception) as context:
            DumpToolFactory.create(config)
        self.assertEqual(str(context.exception), "Valid level is needed.")

        config.execution_mode = Const.GRAPH_GE_MODE
        config.level = Const.CELL
        with patch('msprobe.mindspore.dump.dump_processor.dump_tool_factory.is_graph_mode_cell_dump_allowed') as \
             mock_is_cell_dump_allowed:
            mock_is_cell_dump_allowed.return_value = True
            with self.assertRaises(ValueError):
                DumpToolFactory.create(config)
            mock_logger_error.assert_called_with("Data dump is not supported in graph_ge mode when dump level is cell.")
            mock_logger_error.reset_mock()

            mock_is_cell_dump_allowed.return_value = False
            with self.assertRaises(Exception) as context:
                DumpToolFactory.create(config)
            self.assertEqual(str(context.exception), "Cell dump is not supported in graph mode.")

        config.level = Const.KERNEL
        dumper = DumpToolFactory.create(config)
        self.assertIsInstance(dumper, tuple)
