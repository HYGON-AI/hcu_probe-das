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

from typing import Any, Dict, Optional

from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.file_utils import load_json
from msprobe.core.common.log import logger
from msprobe.core.monitor.utils import validate_int_arg, validate_step_count_per_record, validate_ranks
from msprobe.core.monitor_v2.base import BaseMonitorV2
from msprobe.core.monitor_v2.factory import MonitorFactory
from msprobe.core.monitor_v2.writer import CSVWriterV2


_FRAMEWORK_ALIASES = {
    "pytorch": "pytorch",
    "pt": "pytorch",
    "torch": "pytorch",
    "mindspore": "mindspore",
    "ms": "mindspore",
}


def _resolve_framework(config: Dict[str, Any], fr: Optional[str]) -> str:
    candidate = fr or config.get("framework", "pytorch")
    key = str(candidate).strip().lower()
    if key not in _FRAMEWORK_ALIASES:
        raise ValueError(f"[monitor_v2] Unsupported framework '{candidate}'")
    return _FRAMEWORK_ALIASES[key]


class TrainerMonitorV2:
    """
    Thin orchestrator for monitor_v2.

    Usage (PyTorch):
        mon = TrainerMonitorV2("config.json", fr="pytorch")
        mon.start(model=model)
        for step in range(num_steps):
            ...
            mon.step()
        mon.stop()

    Config (minimal example):
        {
          "framework": "pytorch",
          "output_dir": "./monitor_v2_out",
          "rank": [0, 1],                # optional, list of ranks to monitor
          "start_step": 0,               # optional, first step to write
          "step_interval": 1,            # optional, write every N steps
          "step_count_per_record": 1,    # optional, steps per output file
          "monitors": {
            "module": {
              "enabled": true,
              "targets": [],
              "ops": ["min", "max", "mean", "norm", "nans"]
            }
          }
        }
    """

    def __init__(self, config_path: str, fr: Optional[str] = None) -> None:
        self.config_path = config_path
        self.config: Dict[str, Any] = load_json(config_path)
        if not isinstance(self.config, dict):
            raise TypeError("[monitor_v2] Configuration must be a dictionary.")

        self.framework = _resolve_framework(self.config, fr)

        # Ensure framework adapter is initialised so that rank resolution and
        # tensor ops work correctly for the selected framework.
        try:
            FmkAdp.set_fmk(self.framework)
        except Exception as exc:
            # Best-effort; fall back to default adapter behaviour.
            logger.warning(f"[monitor_v2] Failed to set framework adapter to '{self.framework}': {exc}")

        # Current rank is always resolved from framework adapter.
        self.rank = FmkAdp.get_rank_id()

        # `rank` in config is a list (or single int) of ranks to monitor.
        self._target_ranks = self._parse_target_ranks(self.config.get("rank", []))

        # Step control: start_step / stop_step define active window,
        # step_interval controls how often we write.
        # step_count_per_record controls how many steps go into one output file.
        self.start_step = validate_int_arg(self.config.get("start_step"), "start_step", 0, 0)
        self.step_interval = validate_int_arg(self.config.get("step_interval"), "step_interval", 1, 1)
        try:
            validate_step_count_per_record(self.config.get("step_count_per_record", 1))
            self.step_count_per_record = int(self.config.get("step_count_per_record", 1))
        except Exception as exc:
            logger.warning(f"[monitor_v2] Validate step_count_per_record failed: {exc}; fallback to 1.")
            self.step_count_per_record = 1
        # collect_times: 最大采集次数，达到后停止写出；默认大值表示“几乎一直采集”
        # stop_step：优先使用显式配置，其次由 collect_times 推导，最后使用一个极大默认值
        self.collect_times = validate_int_arg(self.config.get("collect_times"), "collect_times", 1, 100_000_000)
        explicit_stop = validate_int_arg(self.config.get("stop_step"), "stop_step", self.start_step, None)
        if explicit_stop is not None:
            self.stop_step = int(explicit_stop)
        else:
            # start_step + collect_times * interval 对齐“采集次数”语义
            self.stop_step = self.start_step + self.collect_times * max(self.step_interval, 1)
        self.current_step = 0
        self._collected_steps = 0
        self.output_dir = self.config.get("output_dir", "./")

        self.monitors = self._build_monitors()
        fmt = str(self.config.get("format", "csv")).strip().lower()
        if fmt != "csv":
            raise ValueError(f"[monitor_v2] Unsupported format '{fmt}', only 'csv' is currently supported.")
        self.writer = CSVWriterV2(
            self.output_dir,
            rank=self.rank,
            async_write=bool(self.config.get("async_write", False)),
        )
        self._writer_closed = False
        self._optimizer = None
        self._orig_step = None
        self._step_patched = False
        self._patch_optimizer_step = bool(self.config.get("patch_optimizer_step", False))
        self._warned_manual_step_with_optimizer = False

    def start(self, model: Any = None, optimizer: Any = None, **context: Any) -> None:
        """
        Start all configured monitors.
        """
        if not self._is_target_rank():
            return
        static_ctx = {
            "rank": self.rank,
            "output_dir": self.output_dir,
            "step_provider": lambda: self.current_step,
        }
        static_ctx.update(context)
        for mon in self.monitors:
            mon.start(model=model, optimizer=optimizer, **static_ctx)
        if optimizer is not None:
            self._optimizer = optimizer
            if "patch_optimizer_step" not in self.config:
                self._patch_optimizer_step = True
        if self._optimizer is not None and self._patch_optimizer_step:
            self._patch_step_via_optimizer()

    def step(self, from_optimizer: bool = False) -> None:
        """
        Increment step counter and collect/write data from monitors.
        """
        if not self._is_target_rank():
            return
        if self._patch_optimizer_step and self._optimizer is not None and not from_optimizer:
            if not self._warned_manual_step_with_optimizer:
                logger.warning(
                    "[monitor_v2] step() called manually while optimizer patch is enabled; "
                    "skip to avoid double increment. Set patch_optimizer_step=false to "
                    "manage steps manually."
                )
                self._warned_manual_step_with_optimizer = True
            return
        step_val = self.current_step
        self.current_step += 1

        # step_interval 语义：每隔多少个 step 写一次。
        # step_count_per_record 语义：每多少个 step 写到一个 csv 文件。
        if not self._should_collect_step(step_val):
            for mon in self.monitors:
                collect_fn = getattr(mon, "collect", None)
                if not callable(collect_fn):
                    continue
                try:
                    collect_fn()
                except Exception as exc:
                    # 收集失败不影响训练流程，直接跳过该 monitor
                    logger.warning(f"[monitor_v2] Failed to collect data '{mon.__class__.__name__}': {exc}")
            return

        for mon in self.monitors:
            collect_fn = getattr(mon, "collect", None)
            if not callable(collect_fn):
                continue
            out = collect_fn()
            if not out:
                continue
            # Normalise monitor payload so that individual monitors
            # only need to care about producing "rows". Trainer is
            # responsible for adding common metadata.
            if not isinstance(out, dict):
                continue
            if not out.get("rows"):
                continue
            if "slug" not in out:
                slug = getattr(mon, "slug", None)
                if not isinstance(slug, str):
                    slug = mon.__class__.__name__.lower()
                out["slug"] = slug
            if "monitor" not in out:
                out["monitor"] = out.get("slug")
            if "rank" not in out:
                out["rank"] = self.rank
            if "step" not in out:
                out["step"] = step_val
            if "step_interval" not in out:
                out["step_interval"] = self.step_interval
            if "step_count_per_record" not in out:
                out["step_count_per_record"] = self.step_count_per_record
            if "start_step" not in out:
                out["start_step"] = self.start_step
            self.writer.write_monitor_data(out)
        # 仅在实际写出数据时增加采集次数
        self._collected_steps += 1

    def stop(self) -> None:
        """
        Stop all monitors and close writer.
        """
        if not self._is_target_rank():
            return
        for mon in self.monitors:
            mon.stop()
        self._restore_optimizer_step()
        self._close_writer()

    def _build_monitors(self):
        monitors_cfg = self.config.get("monitors", {})
        if not isinstance(monitors_cfg, dict):
            logger.warning("[monitor_v2] Monitors config should be a dict; using empty config.")
            monitors_cfg = {}
        monitors = []

        for name, cfg in monitors_cfg.items():
            if not isinstance(cfg, dict) or not cfg.get("enabled", name == "module"):
                continue
            mon = MonitorFactory.create(self.framework, name)
            if mon is None:
                continue
            mon_cfg = {k: v for k, v in cfg.items() if k != "enabled"}
            setattr(mon, "slug", name)
            mon.set_config(mon_cfg)
            monitors.append(mon)

        return monitors

    def _parse_target_ranks(self, rank_cfg: Any) -> set[int]:
        if isinstance(rank_cfg, int):
            return {rank_cfg}
        if isinstance(rank_cfg, list):
            try:
                validate_ranks(rank_cfg)
                return {int(r) for r in rank_cfg}
            except Exception as exc:
                logger.warning(f"[monitor_v2] Validate rank list failed: {exc}; fallback to all ranks.")
                return set()
        logger.warning("[monitor_v2] Rank config should be int or list; fallback to all ranks.")
        return set()

    def _should_collect_step(self, step: int) -> bool:
        """
        Global step filter for monitor_v2:
        - Only write when start_step <= step < stop_step
        - And (step - start_step) % step_interval == 0 when interval > 1
        - Respect collect_times: stop after N successful writes
        """
        if self.collect_times > 0 and self._collected_steps >= self.collect_times:
            return False
        if not (self.start_step <= step < self.stop_step):
            return False
        if self.step_interval <= 1:
            return True
        return (step - self.start_step) % self.step_interval == 0

    def _is_target_rank(self) -> bool:
        """
        Whether current rank should participate in monitor_v2 collection.
        Empty `rank` list in config means all ranks.
        """
        return not self._target_ranks or self.rank in self._target_ranks

    def _close_writer(self) -> None:
        if self._writer_closed:
            return
        try:
            self.writer.close()
        finally:
            self._writer_closed = True

    def _patch_step_via_optimizer(self) -> None:
        if self._step_patched or self._optimizer is None:
            return
        step_fn = getattr(self._optimizer, "step", None)
        if not callable(step_fn):
            return
        self._orig_step = step_fn

        def wrapped_step(*args: Any, **kwargs: Any):
            out = self._orig_step(*args, **kwargs)
            try:
                self.step(from_optimizer=True)
            except Exception as exc:
                logger.warning(f"[monitor_v2] Failed to run step from optimizer patch: {exc}")
            return out

        self._optimizer.step = wrapped_step  # type: ignore[assignment]
        self._step_patched = True

    def _restore_optimizer_step(self) -> None:
        if not self._step_patched or self._optimizer is None:
            return
        if self._orig_step is not None:
            self._optimizer.step = self._orig_step  # type: ignore[assignment]
        self._step_patched = False
        self._orig_step = None
