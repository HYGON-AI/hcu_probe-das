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

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from msprobe.core.common.log import logger


class BaseMonitorV2(ABC):
    """
    Minimal base class for v2 monitors.

    Compared to core.monitor.BaseMonitor, this class is intentionally kept
    small so that higher-level policies live in dedicated components
    (framework, trainer, backends) instead of being mixed into the base.
    """

    DEFAULT_OPS = ["min", "max", "mean", "norm", "nans"]

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}
        self._context: Dict[str, Any] = {}
        self._ops: List[str] = []
        # Common per-step rows buffer; subclasses can populate this and
        # rely on the default collect() implementation.
        self._rows: List[Dict[str, Any]] = []

    def set_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._configure_ops(config.get("ops"))

    def set_context(self, **context: Any) -> None:
        self._context.update(context)

    def collect(self) -> Optional[Dict[str, Any]]:
        """
        Default collect implementation used by most monitors:
        return per-step rows and reset the internal buffer.
        """
        if not self._rows:
            return None
        data = {"rows": self._rows}
        self._rows = []
        return data

    @abstractmethod
    def start(self, *args: Any, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    def _configure_ops(self, ops: Optional[Iterable[str]] = None) -> None:
        if ops is None or not isinstance(ops, Iterable):
            self._ops = list(self.DEFAULT_OPS)
            return
        requested = [ops] if isinstance(ops, str) else list(ops)
        invalid_ops = [op for op in requested if op not in self.DEFAULT_OPS]
        if invalid_ops:
            logger.warning(
                f"[monitor_v2] Unsupported ops {invalid_ops}; supported: {self.DEFAULT_OPS}."
            )
        filtered = [op for op in requested if op in self.DEFAULT_OPS]
        if not filtered:
            logger.warning("[monitor_v2] No valid ops provided; fallback to default ops.")
        self._ops = filtered if filtered else list(self.DEFAULT_OPS)
