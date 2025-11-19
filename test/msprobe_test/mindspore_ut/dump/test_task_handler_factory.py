#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
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
"""
from unittest import TestCase
from unittest.mock import patch

from msprobe.core.dump.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_processor.kernel_graph_dump import KernelGraphDump
from msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump import KernelKbykDump
from msprobe.mindspore.dump.task_handler_factory import TaskHandlerFactory
from msprobe.mindspore.common.const import Const


class TestTaskHandlerFactory(TestCase):
    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_create(self, _):
        class HandlerFactory:
            def create(self):
                return None

        tasks = {"statistics": HandlerFactory}

        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        config.execution_mode = Const.GRAPH_GE_MODE

        handler = TaskHandlerFactory.create(config)
        self.assertTrue(isinstance(handler, tuple))
        self.assertTrue(isinstance(handler[1], KernelKbykDump))
        self.assertTrue(isinstance(handler[0], KernelGraphDump))

        with patch("msprobe.mindspore.dump.task_handler_factory.TaskHandlerFactory.tasks", new=tasks):
            with self.assertRaises(Exception) as context:
                TaskHandlerFactory.create(config)
            self.assertEqual(str(context.exception), "Can not find task handler")

        config.task = "Free_benchmark"
        with self.assertRaises(Exception) as context:
            TaskHandlerFactory.create(config)
        self.assertEqual(str(context.exception), "Valid task is needed.")
