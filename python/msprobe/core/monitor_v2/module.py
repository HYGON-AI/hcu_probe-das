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
from typing import Any, Dict, List, Optional, Tuple

from msprobe.core.common.const import Const, MonitorConst
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.log import logger
from msprobe.core.monitor_v2.base import BaseMonitorV2
from msprobe.core.monitor_v2.utils import get_vpp_stage_from_tag, iter_model_modules


def _iter_named_modules(model: Any):
    return iter_model_modules(model)


class ModuleMonitorV2(BaseMonitorV2, ABC):
    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[Any] = None
        self._hooks: List[Any] = []
        self._targets: List[str] = []
        self._eps: float = 1e-8

    @staticmethod
    def _try_register_hook(register_fn: Any, hook: Any) -> Any:
        if register_fn is None:
            return None
        try:
            return register_fn(hook, with_kwargs=True)
        except TypeError:
            return register_fn(hook)

    def set_config(self, config: Dict[str, Any]) -> None:
        super().set_config(config)
        self.config = config
        self._targets = config.get("targets", [])
        self._eps = float(config.get("eps", 1e-8))

    @abstractmethod
    def _register_backward_hook(self, module: Any, module_name: str) -> Any:
        ...

    @abstractmethod
    def _compute_metrics(self, tag2tensor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        ...

    def start(self, model: Any = None, optimizer: Any = None, **context: Any) -> None:
        self.set_context(**context)
        if self._model is not None:
            return
        model = model or self._context.get("model")
        if model is None:
            raise ValueError("Model must be provided to start ModuleMonitorV2")
        self._model = model
        self._register_hooks(model)

    def stop(self) -> None:
        if self._model is None:
            return
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception as exc:
                logger.warning(f"[monitor_v2] Failed to remove module hook: {exc}")
        self._hooks.clear()
        self._model = None

    def _register_hooks(self, model: Any) -> None:
        targets = self._targets
        for name, module in _iter_named_modules(model):
            if not targets or any(keyword in name for keyword in targets):
                self._register_module_hooks(module, name)

    def _register_module_hooks(self, module: Any, module_name: str) -> None:
        if not FmkAdp.is_nn_module(module):
            return
        self._add_hook(self._try_register_hook(
            module.register_forward_pre_hook,
            self._build_forward_pre_hook(module_name),
        ))
        self._add_hook(self._try_register_hook(
            module.register_forward_hook,
            self._build_forward_hook(module_name),
        ))
        backward_handle = self._register_backward_hook(module, module_name)
        self._add_hook(backward_handle)

    def _build_forward_pre_hook(self, module_name: str):
        def hook(module: Any, inputs: Any, kwargs: Optional[Dict[str, Any]] = None):
            tag2tensor, idx = self._collect_tagged_tensors(module_name, "input", inputs)
            if kwargs:
                extras, _ = self._collect_tagged_tensors(module_name, "input", tuple(kwargs.values()), idx)
                tag2tensor.update(extras)
            self._record_hook_data(module_name, "input", tag2tensor, "forward")
        return hook

    def _build_forward_hook(self, module_name: str):
        def hook(module: Any, *hook_args: Any, **hook_kwargs: Any):
            if len(hook_args) < 2:
                return
            outputs = hook_args[-1]
            tag2tensor, _ = self._collect_tagged_tensors(module_name, "output", outputs)
            self._record_hook_data(module_name, "output", tag2tensor, "forward")
        return hook

    def _build_backward_hook(self, module_name: str):
        def hook(module: Any, grad_inputs: Any, grad_outputs: Any):
            input_tags, _ = self._collect_tagged_tensors(module_name, "grad_input", grad_inputs)
            output_tags, _ = self._collect_tagged_tensors(module_name, "grad_output", grad_outputs)
            self._record_hook_data(module_name, "grad_input", input_tags, "backward")
            self._record_hook_data(module_name, "grad_output", output_tags, "backward")
        return hook

    def _record_hook_data(self, module_name: str, io_kind: str, tag2tensor: Dict[str, Any], hook_type: str) -> None:
        if not tag2tensor:
            return
        stats = self._compute_metrics(tag2tensor)
        if not stats:
            return
        for tag, op_dict in stats.items():
            vpp_stage = get_vpp_stage_from_tag(tag)
            row = {
                "module_name": tag,
                "scope": hook_type,
                "stats": {op: self._detach_stat(value) for op, value in op_dict.items()},
            }
            if vpp_stage is not None:
                row["vpp_stage"] = vpp_stage
            else:
                row["vpp_stage"] = MonitorConst.DEFAULT_STAGE
            self._rows.append(row)

    def _collect_tagged_tensors(
        self,
        module_name: str,
        io_kind: str,
        tensors: Any,
        start_idx: int = 0,
    ) -> Tuple[Dict[str, Any], int]:
        tagged: Dict[str, Any] = {}
        idx = start_idx
        if tensors is None:
            return tagged, idx
        if FmkAdp.is_tensor(tensors):
            tagged[f"{module_name}.{io_kind}.{idx}"] = tensors
            return tagged, idx + 1
        if isinstance(tensors, (list, tuple)):
            for tensor in tensors:
                if not FmkAdp.is_tensor(tensor):
                    continue
                tagged[f"{module_name}.{io_kind}.{idx}"] = tensor
                idx += 1
        return tagged, idx

    def _detach_stat(self, value: Any) -> Any:
        if value is None or not FmkAdp.is_tensor(value):
            return value
        if FmkAdp.fmk == Const.PT_FRAMEWORK:
            return value.detach()
        return value

    def _add_hook(self, handle: Any) -> None:
        if handle is not None:
            self._hooks.append(handle)


class PyTorchModuleMonitorV2(ModuleMonitorV2):
    def _register_backward_hook(self, module: Any, module_name: str) -> Any:
        return module.register_full_backward_hook(self._build_backward_hook(module_name))

    def _compute_metrics(self, tag2tensor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        from msprobe.pytorch.monitor.module_metric import get_metrics

        return get_metrics(self._ops, tag2tensor, self._eps, {})


class MindSporeModuleMonitorV2(ModuleMonitorV2):
    def _register_backward_hook(self, module: Any, module_name: str) -> Any:
        return module.register_backward_hook(self._build_backward_hook(module_name))

    def _compute_metrics(self, tag2tensor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        from msprobe.mindspore.monitor.utils import get_metrics

        return get_metrics(self._ops, tag2tensor, self._eps, {})
