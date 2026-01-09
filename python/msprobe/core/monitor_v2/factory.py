# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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
