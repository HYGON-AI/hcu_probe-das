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
