# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

from msprobe.core.common.log import logger
from msprobe.mindspore.common.const import Const
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.exception_dump.kernel_graph_exception_dump import KernelGraphExceptionDump


class ExceptionDumpToolFactory:
    tools = {
        Const.CELL: {
            Const.GRAPH_KBYK_MODE: None,
            Const.GRAPH_GE_MODE: None,
            Const.PYNATIVE_MODE: None
        },
        Const.API: {
            Const.GRAPH_KBYK_MODE: None,
            Const.GRAPH_GE_MODE: None,
            Const.PYNATIVE_MODE: None
        },
        Const.KERNEL: {
            Const.GRAPH_KBYK_MODE: KernelGraphExceptionDump,
            Const.GRAPH_GE_MODE: None,
            Const.PYNATIVE_MODE: KernelGraphExceptionDump
        }
    }

    @staticmethod
    def create(config: DebuggerConfig):
        tool = ExceptionDumpToolFactory.tools.get(config.level)
        if not tool:
            raise Exception("Valid level is needed.")
        tool = tool.get(config.execution_mode)
        if not tool:
            logger.error(f"Exception dump is not supported in {config.execution_mode} mode "
                         f"when level is {config.level}.")
            raise ValueError
        return (tool(config),)
