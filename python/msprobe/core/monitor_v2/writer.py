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

import csv
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from msprobe.core.common.const import Const
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.file_utils import make_dir
from msprobe.core.common.log import logger


class CSVWriterV2:
    """
    Minimal CSV writer for monitor_v2.

    - Each rank gets its own directory: <out_dir>/rank_<rank>/
    - Each monitor writes to <slug>_step<start>-<end>.csv
    - Row structure: step plus dynamic fields from rows (no ts/monitor/rank).
    """

    def __init__(self, out_dir: str = "monitor_v2_output", rank: int = 0, async_write: bool = False):
        self.out_dir = out_dir
        self.rank = rank
        self.async_write = async_write  # reserved, currently synchronous
        if self.async_write:
            logger.warning("[monitor_v2] async_write is not supported yet; falling back to sync writes.")

        self.rank_dir = os.path.join(self.out_dir, f"rank_{self.rank}")
        make_dir(self.rank_dir)
        self._known_fields: Dict[str, List[str]] = {}

    @staticmethod
    def close():
        return None

    @staticmethod
    def _stack_tensors(tensors: List[Any]) -> Any:
        if FmkAdp.fmk == Const.PT_FRAMEWORK:
            import torch

            return torch.stack(tensors, dim=0)
        from mindspore import ops

        return ops.stack(tensors, axis=0)

    def write_monitor_data(self, monitor_data: Dict[str, Any]):
        rows = monitor_data.get("rows") or []
        if not rows:
            return None
        rows = self._flatten_rows(rows)
        if not rows:
            return None

        monitor_name = monitor_data.get("monitor", "monitor")
        slug = self._safe_slug(monitor_data.get("slug", monitor_name))
        step = monitor_data.get("step", "unknown")
        interval = self._safe_int(monitor_data.get("step_count_per_record", 1), default=1)
        start_step = self._safe_int(monitor_data.get("start_step", 0), default=0)
        csv_path = self._resolve_csv_path(slug, step, interval, start_step)

        base_fields = ["vpp_stage", "step"]
        row_keys = list(dict.fromkeys(k for r in rows for k in r if k not in base_fields))
        fields = base_fields + row_keys

        file_exists = os.path.exists(csv_path)
        existing = self._known_fields.get(slug)
        if existing:
            if not file_exists:
                for k in fields:
                    if k not in existing:
                        existing.append(k)
            fields = existing
        else:
            self._known_fields[slug] = fields

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            for r in rows:
                out = dict(r)
                out.setdefault("step", step)
                writer.writerow(out)

        return csv_path

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_slug(slug: Any) -> str:
        text = str(slug) if slug is not None else "monitor"
        text = text.replace(os.sep, "_")
        if os.altsep:
            text = text.replace(os.altsep, "_")
        text = text.replace("..", "_")
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
        return text or "monitor"

    def _resolve_csv_path(self, slug: str, step: Any, interval: int, start_step: int) -> str:
        try:
            step_val = int(step)
        except (TypeError, ValueError):
            return os.path.join(self.rank_dir, f"{slug}.csv")
        interval = max(interval, 1)
        relative = step_val - start_step
        if relative < 0:
            start = step_val
        else:
            start = start_step + (relative // interval) * interval
        end = start + interval - 1
        return os.path.join(self.rank_dir, f"{slug}_step{start}-{end}.csv")

    def _flatten_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        stat_ops = list(dict.fromkeys(op for row in rows for op in (row.get("stats") or {})))
        stacked_values = self._stack_stats(rows, stat_ops)
        default_values = [float("nan")] * len(rows)
        flattened = []
        for idx, row in enumerate(rows):
            base = {k: v for k, v in row.items() if k != "stats"}
            for op in stat_ops:
                values = stacked_values.get(op, default_values)
                base[op] = values[idx] if idx < len(values) else float("nan")
            flattened.append(base)
        return flattened

    def _stack_stats(self, rows: List[Dict[str, Any]], stat_ops: List[str]) -> Dict[str, List[float]]:
        stacked: Dict[str, List[float]] = {}
        total = len(rows)
        for op in stat_ops:
            tensors = [(row.get("stats") or {}).get(op) for row in rows]
            if not tensors or any(tensor is None for tensor in tensors):
                stacked[op] = [float("nan")] * total
                continue
            stacked_tensor = self._stack_tensors(tensors)
            if FmkAdp.fmk == Const.PT_FRAMEWORK:
                stacked_tensor = stacked_tensor.cpu()
            numpy_vals = FmkAdp.asnumpy(stacked_tensor).reshape(-1)
            stacked[op] = [float(value) for value in numpy_vals.tolist()]
        return stacked
