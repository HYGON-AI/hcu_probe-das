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

from msprobe.pytorch.common.log import logger
from msprobe.pytorch.dump.module_dump.module_processer import ModuleProcesser
from msprobe.pytorch.dump.api_dump.api_register import get_api_register


class ModuleDumper:
    def __init__(self, service):
        self.service = service
        self.api_register = get_api_register()

    def start_module_dump(self, module, dump_name):
        if hasattr(module, 'msprobe_hook') and not hasattr(module, 'msprobe_module_dump'):
            logger.info_on_rank_0("The init dump is enabled, and the module dump function will not be available.")
            return

        ModuleProcesser.enable_module_dump = True
        self.api_register.restore_all_api()
        if not hasattr(module, 'msprobe_module_dump'):
            self.service.module_processor.register_module_hook(module, self.service.build_hook,
                                                               recursive=False, module_names=[dump_name])
            setattr(module, 'msprobe_module_dump', True)

    def stop_module_dump(self):
        ModuleProcesser.enable_module_dump = False
        self.api_register.register_all_api()
