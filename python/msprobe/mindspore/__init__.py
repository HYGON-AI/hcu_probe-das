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

import os

try:
    from msprobe.lib import _msprobe_c
    os.environ["HOOK_TOOL_PATH"] = _msprobe_c.__file__
except ImportError:
    from .common.log import logger
    logger.info("Module _msprobe_c has not been installed. L2-Dump may not work normally.")

from msprobe.mindspore.dump.debugger.precision_debugger import PrecisionDebugger
from msprobe.mindspore.common.utils import seed_all, MsprobeStep, MsprobeInitStep
from msprobe.mindspore.monitor.module_hook import TrainerMon
from msprobe.mindspore.dump.dump_processor.graph_tensor_dump import save, save_grad, step
