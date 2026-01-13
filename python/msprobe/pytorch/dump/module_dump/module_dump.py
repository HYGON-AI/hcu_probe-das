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

from msprobe.pytorch.common.log import logger
from msprobe.pytorch.dump.module_dump.module_processor import ModuleProcessor
from msprobe.pytorch.dump.api_dump.api_register import get_api_register


class ModuleDumper:
    def __init__(self, service):
        self.service = service
        self.api_register = get_api_register()

    def start_module_dump(self, module, dump_name):
        if hasattr(module, 'msprobe_hook') and not hasattr(module, 'msprobe_module_dump'):
            logger.info_on_rank_0("The init dump is enabled, and the module dump function will not be available.")
            return

        ModuleProcessor.enable_module_dump = True
        self.api_register.restore_all_api()
        if not hasattr(module, 'msprobe_module_dump'):
            self.service.module_processor.register_module_hook(module, self.service.build_hook,
                                                               recursive=False, module_names=[dump_name])
            setattr(module, 'msprobe_module_dump', True)

    def stop_module_dump(self):
        ModuleProcessor.enable_module_dump = False
        self.api_register.register_all_api()
