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


from __future__ import annotations

from typing import Any, Dict, Optional, Type

from msprobe.core.common.log import logger
from msprobe.core.monitor_v2.module import (
    MindSporeModuleMonitorV2,
    PyTorchModuleMonitorV2,
)
from msprobe.core.monitor_v2.cc import PyTorchCCMonitorV2, MindSporeCCMonitorV2
from msprobe.core.monitor_v2.optimizer import (
    MindSporeOptimizerMonitorV2,
    PyTorchOptimizerMonitorV2,
)
from msprobe.core.monitor_v2.param import (
    MindSporeParamMonitorV2,
    PyTorchParamMonitorV2,
)
from msprobe.core.monitor_v2.weight_grad import (
    MindSporeWeightGradMonitorV2,
    PyTorchWeightGradMonitorV2,
)

MonitorCls = Type


class MonitorFactory:
    """
    Registry-based factory that organizes monitors by framework, then by monitor name.
    """

    _REGISTRY: Dict[str, Dict[str, MonitorCls]] = {
        "pytorch": {
            "module": PyTorchModuleMonitorV2,
            "weight_grad": PyTorchWeightGradMonitorV2,
            "optimizer": PyTorchOptimizerMonitorV2,
            "param": PyTorchParamMonitorV2,
            "cc": PyTorchCCMonitorV2,
        },
        "mindspore": {
            "module": MindSporeModuleMonitorV2,
            "weight_grad": MindSporeWeightGradMonitorV2,
            "optimizer": MindSporeOptimizerMonitorV2,
            "param": MindSporeParamMonitorV2,
            "cc": MindSporeCCMonitorV2,
        },
    }

    @classmethod
    def create(cls, framework: str, name: str) -> Optional[Any]:
        fwk_key = str(framework).lower()
        name_key = str(name).lower()
        fwk = cls._REGISTRY.get(fwk_key)
        if not fwk:
            logger.warning(f"[monitor_v2] Unsupported framework '{framework}'.")
            return None
        mon_cls = fwk.get(name_key)
        if mon_cls is None:
            logger.warning(f"[monitor_v2] Unsupported monitor '{name}' for framework '{framework}'.")
        return mon_cls() if mon_cls else None

    @classmethod
    def available(cls, framework: str) -> Dict[str, MonitorCls]:
        return cls._REGISTRY.get(str(framework).lower(), {}).copy()
