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

from typing import Any, Dict

from msprobe.core.common.const import Const, MonitorConst
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.log import logger
from msprobe.core.monitor_v2.base import BaseMonitorV2
from msprobe.core.monitor_v2.utils import build_param2name, get_vpp_stage_from_tag


class OptimizerMonitorV2(BaseMonitorV2):
    """
    通用 Optimizer 监控基类（当前只实现 mv_distribution，后续可扩展 wg/param 等）。

    公共职责：
      - 解析 config（eps / mv_distribution / ops）
      - 记录 model / optimizer / context
      - 基于模型构建 param -> name 映射
      - 将 (exp_avg_dict, exp_avg_sq_dict) 转成 rows（module_name / hook_type / stats）

    框架差异由子类通过以下方法提供：
      - _build_backend
      - _get_mv
      - _compute_metrics
    """

    def __init__(self) -> None:
        super().__init__()
        self._optimizer: Any = None
        self._backend: Any = None
        self._param2name: Dict[Any, str] = {}
        self._eps: float = 1e-8

        self.mv_distribution: bool = True
        self.mg_direction: bool = False
        self.ur_distribution: bool = False

    def set_config(self, config: Dict[str, Any]) -> None:
        super().set_config(config)
        self.config = config
        self._eps = float(config.get("eps", 1e-8))
        self.mv_distribution = bool(config.get("mv_distribution", True))

    def start(self, model: Any = None, optimizer: Any = None, **context: Any) -> None:
        self.set_context(**context)
        if self._optimizer is not None:
            return
        optimizer = optimizer or self._context.get("optimizer")
        model = model or self._context.get("model")
        if optimizer is None or model is None:
            raise ValueError("Model and optimizer must be provided to start optimizer function")
        self._optimizer = optimizer
        self._register_params_v2(model)
        self._backend = self._build_backend(optimizer)

    def stop(self) -> None:
        self._optimizer = None
        self._backend = None
        self._param2name.clear()
        self._rows = []

    def collect(self) -> Dict[str, Any] | None:
        if not self.mv_distribution or not self._param2name:
            return None
        exp_avg_dict, exp_avg_sq_dict = self._get_mv()
        if not exp_avg_dict and not exp_avg_sq_dict:
            return None
        self._rows = []
        self._record_mv("exp_avg", exp_avg_dict)
        self._record_mv("exp_avg_sq", exp_avg_sq_dict)
        return super().collect()

    def _register_params_v2(self, model: Any) -> None:
        """
        使用统一工具函数建立 param->name 映射，支持
        单个 model 或 model 列表。
        """
        self._param2name = build_param2name(model=model, optimizer=self._optimizer)

    def _register_params(self, model: Any) -> None:
        """
        默认使用模型的 named_parameters 建立 param->name 映射。
        子类可在特殊场景中覆写。
        """
        self._param2name.clear()
        try:
            for name, param in FmkAdp.named_parameters(model):
                self._param2name[param] = name
        except Exception:
            self._param2name.clear()

    def _record_mv(self, metric_type: str, name2tensor: Dict[str, Any]) -> None:
        if not name2tensor:
            return
        tag2tensor: Dict[str, Any] = {}
        for name, tensor in name2tensor.items():
            if tensor is None or not FmkAdp.is_tensor(tensor):
                continue
            tag2tensor[f"{name}.{metric_type}"] = tensor
        if not tag2tensor:
            return
        stats = self._compute_metrics(tag2tensor)
        if not stats:
            return
        for tag, op_dict in stats.items():
            vpp_stage = get_vpp_stage_from_tag(tag)
            row = {
                "module_name": tag,
                "scope": metric_type,
                "stats": {op: self._detach_stat(val) for op, val in op_dict.items()},
            }
            if vpp_stage is not None:
                row["vpp_stage"] = vpp_stage
            else:
                row["vpp_stage"] = MonitorConst.DEFAULT_STAGE
            self._rows.append(row)

    def _detach_stat(self, value: Any) -> Any:
        if value is None or not FmkAdp.is_tensor(value):
            return value
        if FmkAdp.fmk == Const.PT_FRAMEWORK:
            return value.detach()
        return value

    # 由子类实现的框架特定逻辑
    def _build_backend(self, optimizer: Any) -> Any:  # pragma: no cover - abstract by convention
        raise NotImplementedError

    def _get_mv(self) -> tuple[Dict[str, Any], Dict[str, Any]]:  # pragma: no cover - abstract by convention
        raise NotImplementedError

    def _compute_metrics(self, tag2tensor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError


class PyTorchOptimizerMonitorV2(OptimizerMonitorV2):
    """
    PyTorch optimizer monitor，当前实现 mv_distribution 部分。
    """

    def _build_backend(self, optimizer: Any) -> Any:
        from msprobe.pytorch.monitor.optimizer_collect import OptimizerMonFactory

        return OptimizerMonFactory.create_optimizer_mon(optimizer)

    def _get_mv(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        from msprobe.core.monitor.utils import MVResult

        if self._backend is None:
            return {}, {}
        result = self._backend.fetch_mv(self, self._param2name)
        if isinstance(result, MVResult):
            return result.exp_avg, result.exp_avg_sq
        try:
            exp_avg_dict, exp_avg_sq_dict = result[0], result[1]
            return exp_avg_dict, exp_avg_sq_dict
        except Exception:
            return {}, {}

    def _compute_metrics(self, tag2tensor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        from msprobe.pytorch.monitor.module_metric import get_metrics

        return get_metrics(self._ops, tag2tensor, self._eps, {})


class MindSporeOptimizerMonitorV2(OptimizerMonitorV2):
    """
    MindSpore optimizer monitor，当前实现 mv_distribution 部分。

    对齐 mindspore/monitor 中的两条路径：
      - mindtorch 场景：使用 optimizer_collect.OptimizerMon.fetch_mv
      - 纯 MindSpore 场景：使用 get_mv_for_ms 从 optimizer.parameters 中抓取 m/v
    """

    def _build_backend(self, optimizer: Any) -> Any:
        from msprobe.mindspore.monitor.optimizer_collect import OptimizerMonFactory

        return OptimizerMonFactory.create_optimizer_mon(optimizer)

    def _get_mv(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        from msprobe.mindspore.common.utils import is_mindtorch

        if self._optimizer is None:
            return {}, {}
        if is_mindtorch():
            if self._backend is None:
                return {}, {}
            result = self._backend.fetch_mv(self, self._param2name)
            try:
                exp_avg_dict, exp_avg_sq_dict = result[0], result[1]
            except Exception:
                return {}, {}
            return exp_avg_dict, exp_avg_sq_dict
        # 纯 MindSpore 场景：使用原生 get_mv_for_ms 语义
        return self._get_mv_for_ms(self._optimizer)

    def _get_mv_for_ms(self, opt: Any) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        直接复用 mindspore/monitor.module_hook.get_mv_for_ms 的采集语义：
          - 若是纯 MindSpore，optimizer 可能包了一层，需要解出真正的 Cell
          - 从 get_parameters(common_opt) 中抓取名字带 EXP_AVG / EXP_AVG_SQ 的参数
        """
        from msprobe.mindspore.monitor.common_func import is_valid_instance, get_parameters

        if not self.mv_distribution:
            return {}, {}

        common_opt = opt
        if not is_valid_instance(opt):
            common_opt = getattr(opt, "optimizer", None)
            if not is_valid_instance(common_opt):
                logger.warning("MindSporeOptimizerMonitorV2: optimizer is not valid, please check usage")
                return {}, {}

        m_dict: Dict[str, Any] = {}
        v_dict: Dict[str, Any] = {}
        for name, param in get_parameters(common_opt):
            if MonitorConst.EXP_AVG_SQ in name:
                v_dict[name] = param
            elif MonitorConst.EXP_AVG in name:
                m_dict[name] = param

        return m_dict, v_dict

    def _compute_metrics(self, tag2tensor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        from msprobe.mindspore.monitor.utils import get_metrics

        return get_metrics(self._ops, tag2tensor, self._eps, {})
