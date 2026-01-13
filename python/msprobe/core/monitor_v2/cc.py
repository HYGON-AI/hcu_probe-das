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

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from msprobe.core.common.log import logger
from msprobe.core.monitor_v2.base import BaseMonitorV2


class _PTCCContextV2:
    """
    Lightweight communication context for PyTorch cc_distribution monitor_v2.

    结构与 pytorch.monitor.module_hook.CommunicationContext 一致：
      - data: {tag: {op: aggregated_tensor_or_scalar}}
      - aggregate(): 对同一 tag/op 下的多次调用做聚合（min/max/mean/...）
    """

    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _agg(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        from msprobe.pytorch.monitor.distributed.wrap_distributed import op_aggregate

        aggregated_data: Dict[str, Dict[str, Any]] = {}
        for tag, op2tensorlist in data.items():
            aggregated_data[tag] = {}
            for op, tensorlist in op2tensorlist.items():
                aggregated_data[tag][op] = op_aggregate(op, tensorlist)
        return aggregated_data

    def reset(self) -> None:
        self.data = {}

    def aggregate(self) -> None:
        self.data = self._agg(self.data)


class _MSCCContextV2:
    """
    MindSpore 版本的通信 context，结构与 _PTCCContextV2 相同，
    但使用 mindspore.wrap_distributed.op_aggregate。
    """

    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _agg(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        from msprobe.mindspore.monitor.distributed.wrap_distributed import op_aggregate

        aggregated_data: Dict[str, Dict[str, Any]] = {}
        for tag, op2tensorlist in data.items():
            aggregated_data[tag] = {}
            for op, tensorlist in op2tensorlist.items():
                aggregated_data[tag][op] = op_aggregate(op, tensorlist)
        return aggregated_data

    def reset(self) -> None:
        self.data = {}

    def aggregate(self) -> None:
        self.data = self._agg(self.data)


class _BaseCCMonitorV2(BaseMonitorV2, ABC):
    """
    通信算子监控（cc_distribution）的通用 v2 基类。

    框架相关的细节（环境检查 / hook 注册 / API 恢复等）
    由子类实现，避免在子类中再做 FmkAdp.fmk 等框架判断。
    """

    def __init__(self) -> None:
        super().__init__()
        self._patched: bool = False
        self._cc_handles: List[Any] = []
        self.cc_context: Dict[str, Any] = defaultdict(self._create_context)

        # cc 配置
        self.cc_codeline: List[str] = []
        self.cc_log_only: bool = False
        self.cc_pre_hook: bool = False
        self.cc_logged_stack: Dict[str, set] = defaultdict(set)
        # 保持与 TrainerMon 同名，供 create_hooks 使用
        self.module_rank_list: List[int] = []

    @property
    def ops(self) -> List[str]:
        return self._ops

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        配置示例（monitors.cc）：
          {
            "enabled": true,
            "ops": ["min", "max", "mean", "norm", "nans"],
            "eps": 1e-8,
            "cc_codeline": [],
            "cc_log_only": false,
            "cc_pre_hook": false,
            "module_ranks": []
          }
        """
        super().set_config(config)
        self.config = config
        # 兼容 cc_distribution 结构：如果嵌套提供，则取内部 dict
        cc_cfg = config.get("cc_distribution", config)
        self.cc_codeline = cc_cfg.get("cc_codeline", [])
        self.cc_log_only = bool(cc_cfg.get("cc_log_only", False))
        self.cc_pre_hook = bool(cc_cfg.get("cc_pre_hook", False))
        self.module_rank_list = list(cc_cfg.get("module_ranks", []))

    def start(self, model: Any = None, optimizer: Any = None, **context: Any) -> None:
        self.set_context(**context)
        if self._patched:
            return

        if not self._is_env_ready():
            return

        api_register, create_hooks = self._get_wrap_module()
        pre_hooks, post_hooks = create_hooks(context=self.cc_context, monitor=self)
        self._cc_handles = api_register.initialize_hook(pre_hooks, post_hooks)
        api_register.redirect_api()
        self._patched = True

    def stop(self) -> None:
        if not self._patched:
            return
        try:
            self._restore_api()
        except Exception as exc:
            logger.warning(f"[monitor_v2] Failed to restore communication APIs: {exc}")
        self._patched = False
        self._cc_handles = []
        for _, ctx in self.cc_context.items():
            ctx.reset()
        self.cc_context.clear()
        self._rows = []

    def collect(self) -> Dict[str, Any] | None:
        """
        聚合各通信算子的监控数据，并转为 rows。
        """
        if not self._patched or not self.cc_context:
            return None

        metric_dict: Dict[str, Dict[str, Any]] = {}
        for ctx in self.cc_context.values():
            ctx.aggregate()
            # TrainerMon 中的行为一致：后写入的 op/tag 覆盖前面的同名项
            for tag, op2val in ctx.data.items():
                metric_dict.setdefault(tag, {}).update(op2val)
            ctx.reset()

        self._after_collect(metric_dict)

        if not metric_dict:
            return None

        self._rows = []
        for tag, op2val in metric_dict.items():
            row = {
                "module_name": tag,
                "scope": "comm",
                "stats": op2val,
            }
            self._rows.append(row)

        return super().collect()

    # ---- 子类需要实现的抽象方法 ----

    @abstractmethod
    def _create_context(self) -> Any:
        ...

    @abstractmethod
    def _is_env_ready(self) -> bool:
        ...

    @abstractmethod
    def _get_wrap_module(self) -> Tuple[Any, Any]:
        """
        返回 (api_register, create_hooks) 两个对象，供 start() 使用。
        """

    @abstractmethod
    def _restore_api(self) -> None:
        ...

    def _after_collect(self, metric_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        子类可在 collect 之后做一些额外操作（如 MindSpore 重置 idx）。
        """
        return


class PyTorchCCMonitorV2(_BaseCCMonitorV2):
    """
    PyTorch 通信算子监控（cc_distribution）的 v2 实现。
    """

    def _create_context(self) -> Any:
        return _PTCCContextV2()

    def _is_env_ready(self) -> bool:
        # 仅在分布式环境下挂通信 hook
        try:
            import torch.distributed as dist  # type: ignore[import]
        except Exception:
            return False
        return dist.is_initialized()

    def _get_wrap_module(self) -> Tuple[Any, Any]:
        from msprobe.pytorch.monitor.distributed.wrap_distributed import (
            api_register,
            create_hooks,
        )

        return api_register, create_hooks

    def _restore_api(self) -> None:
        from msprobe.pytorch.monitor.distributed.wrap_distributed import api_register

        api_register.restore_api()


class MindSporeCCMonitorV2(_BaseCCMonitorV2):
    """
    MindSpore 通信算子监控（cc_distribution）的 v2 实现。

    骨架对齐 mindspore/monitor.module_hook 中 cc_distribution 的逻辑。
    """

    def _create_context(self) -> Any:
        return _MSCCContextV2()

    def _is_env_ready(self) -> bool:
        try:
            from mindspore import communication  # type: ignore[import]
        except Exception:
            return False
        if not getattr(communication, "GlobalComm", None) or not communication.GlobalComm.INITED:
            return False
        return True

    def _get_wrap_module(self) -> Tuple[Any, Any]:
        from msprobe.mindspore.monitor.distributed.wrap_distributed import (
            api_register,
            create_hooks,
        )

        return api_register, create_hooks

    def _restore_api(self) -> None:
        from msprobe.mindspore.monitor.distributed.wrap_distributed import api_register

        api_register.restore_api()

    def _after_collect(self, metric_dict: Dict[str, Dict[str, Any]]) -> None:
        # reset communication op call index for next step
        try:
            from msprobe.mindspore.monitor.distributed.wrap_distributed import api_register

            api_register.reset_idx()
        except Exception as exc:
            logger.warning(f"[monitor_v2] Failed to reset MindSpore communication index: {exc}")
