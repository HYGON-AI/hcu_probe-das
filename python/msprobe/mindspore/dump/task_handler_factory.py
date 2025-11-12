# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

from msprobe.core.common.const import Const
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_processor.dump_tool_factory import DumpToolFactory
from msprobe.mindspore.dump.overflow_check.overflow_check_tool_factory import OverflowCheckToolFactory
from msprobe.mindspore.dump.exception_dump.exception_dump_tool_factory import ExceptionDumpToolFactory


class TaskHandlerFactory:
    tasks = {
        Const.TENSOR: DumpToolFactory,
        Const.STATISTICS: DumpToolFactory,
        Const.OVERFLOW_CHECK: OverflowCheckToolFactory,
        Const.EXCEPTION_DUMP: ExceptionDumpToolFactory
    }

    @staticmethod
    def create(config: DebuggerConfig, model=None):
        task = TaskHandlerFactory.tasks.get(config.task)
        if not task:
            raise Exception("Valid task is needed.")
        if task == DumpToolFactory:
            handler = task.create(config, model)
        else:
            handler = task.create(config)
        if not handler:
            raise Exception("Can not find task handler")
        return handler
