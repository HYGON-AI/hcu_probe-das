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

from collections import defaultdict
import mindspore as ms
from mindspore.ops.primitive import Primitive

from msprobe.core.common.utils import Const
from msprobe.core.dump.service import BaseService
from msprobe.mindspore.dump.cell_processor import CellProcessor
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.common.utils import (
    get_rank_if_initialized,
    is_mindtorch,
    get_cells_and_names_with_index
)
from msprobe.mindspore.dump.dump_processor.hook_cell.api_register import get_api_register, ApiTemplate
from msprobe.mindspore.dump.dump_processor.hook_cell.ms_hook_manager import MindsporeHookManager
from msprobe.mindspore.dump.dump_processor.hook_cell.primitive_hooks import PrimitiveHookService
from msprobe.mindspore.dump.dump_processor.jit_dump import JitDump

try:
    from mindspore.common._pijit_context import PIJitCaptureContext
except ImportError:
    pijit_label = False
else:
    pijit_label = True


class MindsporeService(BaseService):
    @property
    def _get_framework_type(self):
        return Const.MT_FRAMEWORK if is_mindtorch() else Const.MS_FRAMEWORK

    @staticmethod
    def _get_current_rank():
        return get_rank_if_initialized()

    def empty(self, *args, **kwargs):
        pass

    def reset_status(self):
        self._reset_status()

    def _init_specific_components(self):
        self.logger = logger
        self.api_register = get_api_register()
        self.primitive_hook_service = PrimitiveHookService(self)
        self.cell_processor = CellProcessor(self.data_collector.scope)
        self.hook_manager = MindsporeHookManager(self.data_collector, self.config)
        self._setup_jit_context()
        self.api_template = ApiTemplate

    def _setup_jit_context(self):
        if self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L1]:
            JitDump.set_config(self.config)
            JitDump.set_data_collector(self.data_collector)
            if hasattr(ms.common.api, "_MindsporeFunctionExecutor"):
                ms.common.api._MindsporeFunctionExecutor = JitDump
            else:
                ms.common.api._JitExecutor = JitDump
            ms.common.api._PyNativeExecutor.grad = JitDump.grad
            if pijit_label:
                PIJitCaptureContext.__enter__ = self.empty
                PIJitCaptureContext.__exit__ = self.empty

    def _register_module_hook(self):
        self.cell_processor.register_cell_hook(self.model, self.build_hook, self.config)
        self.logger.info_on_rank_0(f"The module {self.config.task} hook function is successfully mounted to the model.")

    def _register_hook(self):
        self._register_primitive_hook()

    def _register_primitive_hook(self):
        if self.config.level not in [Const.LEVEL_MIX, Const.LEVEL_L1]:
            return
        if not self.model or self.config.task not in Const.DUMP_DATA_COLLECTION_LIST:
            return

        primitive_set = set()
        cells_and_names_with_index, _ = get_cells_and_names_with_index(self.model)
        for cells_and_names in cells_and_names_with_index.values():
            for _, cell in cells_and_names:
                for attribute, value in vars(cell).items():
                    if isinstance(value, Primitive):
                        primitive_set.add((attribute, value))

        for pname, primitive in primitive_set:
            primitive_class_name = primitive.__class__.__name__
            primitive_combined_name = pname + Const.SEP + primitive_class_name
            new_primitive = type('NewPrimitive', (primitive.__class__,),
                                    {'__call__': self.primitive_hook_service.wrap_primitive(primitive.__call__,
                                                                                            primitive_combined_name)})
            primitive.__class__ = new_primitive

    def _reset_status(self):
        super()._reset_status()
        self.primitive_hook_service.primitive_counters.clear()
        JitDump.jit_count = defaultdict(int)

    def _change_jit_switch(self, status):
        JitDump.jit_dump_switch = status
