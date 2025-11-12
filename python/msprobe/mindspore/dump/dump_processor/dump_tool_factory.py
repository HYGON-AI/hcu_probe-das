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

from msprobe.mindspore.common.const import Const
from msprobe.core.common.log import logger
from msprobe.mindspore.common.utils import is_graph_mode_cell_dump_allowed
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_processor.kernel_graph_dump import KernelGraphDump
from msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump import KernelKbykDump
from msprobe.mindspore.dump.dump_processor.graph_mode_cell_dump import GraphModeCellDump


class DumpToolFactory:
    tools = {
        Const.CELL: {
            Const.GRAPH_KBYK_MODE: GraphModeCellDump,
            Const.GRAPH_GE_MODE: None,
            Const.PYNATIVE_MODE: None
        },
        Const.API: {
            Const.GRAPH_KBYK_MODE: None,
            Const.GRAPH_GE_MODE: None,
            Const.PYNATIVE_MODE: None
        },
        Const.KERNEL: {
            Const.GRAPH_KBYK_MODE: KernelKbykDump,
            Const.GRAPH_GE_MODE: KernelGraphDump,
            Const.PYNATIVE_MODE: KernelKbykDump
        }
    }

    @staticmethod
    def create(config: DebuggerConfig, model=None):
        if config.level == Const.CELL:
            if not is_graph_mode_cell_dump_allowed(config):
                raise Exception("Cell dump is not supported in graph mode.")
            if len(config.data_mode) != 1 or config.data_mode[0] not in Const.GRAPH_CELL_DUMP_DATA_MODE_LIST:
                raise Exception("data_mode must be one of all, forward, backward.")
        else:
            if len(config.data_mode) != 1 or config.data_mode[0] not in Const.GRAPH_DATA_MODE_LIST:
                raise Exception("data_mode must be one of all, input, output.")
        if config.level == Const.KERNEL:
            return (KernelGraphDump(config), KernelKbykDump(config))
        tool = DumpToolFactory.tools.get(config.level)
        if not tool:
            raise Exception("Valid level is needed.")
        tool = tool.get(config.execution_mode)
        if not tool:
            logger.error(f"Data dump is not supported in {config.execution_mode} mode "
                         f"when dump level is {config.level}.")
            raise ValueError
        return tool(config, model) if tool == GraphModeCellDump else tool(config)
