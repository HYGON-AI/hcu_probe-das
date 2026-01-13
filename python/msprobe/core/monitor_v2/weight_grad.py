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
from typing import Any, Dict, List, Optional

from msprobe.core.common.const import Const, MonitorConst
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.log import logger
from msprobe.core.monitor_v2.base import BaseMonitorV2
from msprobe.core.monitor_v2.utils import build_param2name, get_vpp_stage_from_tag, iter_model_chunks


class WeightGradMonitorV2(BaseMonitorV2, ABC):
    """
    Lightweight weight-gradient monitor that patches optimizer.step to grab
    pre/post gradients, reusing per-framework metric computation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._optimizer: Any = None
        self._model: Any = None
        self._param2name: Dict[Any, str] = {}
        self._orig_step = None
        self._patched = False
        self._eps: float = 1e-8
        self._grad_hooks: List[Any] = []
        self.monitor_mbs_grad: bool = False
        self._micro_batch_number: int = 1
        self._param_micro_steps: Dict[Any, int] = {}

    @abstractmethod
    def _compute_metrics(self, tag2tensor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        ...

    def set_config(self, config: Dict[str, Any]) -> None:
        super().set_config(config)
        self._eps = self._parse_eps(self.config.get("eps", 1e-8))
        self.monitor_mbs_grad = self._parse_bool(self.config.get("monitor_mbs_grad", False))

    def start(self, model: Any = None, optimizer: Any = None, **context: Any) -> None:
        self.set_context(**context)
        if self._optimizer is not None:
            return
        optimizer = optimizer or self._context.get("optimizer")
        if optimizer is None:
            raise ValueError("Optimizer must be provided to start weight grad function")
        self._optimizer = optimizer
        self._model = model or self._context.get("model")
        self._micro_batch_number = self._resolve_micro_batch_number()
        self._register_params(self._model, optimizer)
        self._patch_optimizer()

    def stop(self) -> None:
        if self._optimizer is None:
            return
        self._restore_optimizer()
        self._optimizer = None
        self._model = None
        self._rows = []
        self._param2name.clear()
        self._clear_grad_hooks()

    # ---------------------------------------------------------------------
    #  Hook registration helpers
    # ---------------------------------------------------------------------

    def _register_params(self, model: Any, optimizer: Any) -> None:
        self._param2name = build_param2name(model=model, optimizer=optimizer)
        self._register_pre_grad_hooks()

    def _register_pre_grad_hooks(self) -> None:
        self._param_micro_steps.clear()
        for param, name in self._param2name.items():
            if not getattr(param, "requires_grad", True):
                continue
            handle = self._create_param_hook(param, name)
            if handle is not None:
                self._grad_hooks.append(handle)

    def _create_param_hook(self, param: Any, name: str):
        hook_fn = getattr(param, "register_hook", None)
        if not callable(hook_fn):
            return None
        self._param_micro_steps[param] = 0
        return hook_fn(lambda grad, *, _p=param, _n=name: self._on_param_grad(_p, _n, grad))

    def _on_param_grad(self, param: Any, name: str, grad: Any) -> None:
        if grad is None or not FmkAdp.is_tensor(grad):
            return
        current_idx = self._param_micro_steps.get(param, 0) + 1
        self._param_micro_steps[param] = current_idx

        should_record = self.monitor_mbs_grad or (
            self._micro_batch_number <= 0 or current_idx >= self._micro_batch_number
        )
        if should_record:
            micro_step_val = current_idx if self.monitor_mbs_grad else self._micro_batch_number
            self._record_grad("unreduced", name, grad, micro_step=micro_step_val)

        if self._micro_batch_number > 0 and current_idx >= self._micro_batch_number:
            self._param_micro_steps[param] = 0

    def _clear_grad_hooks(self) -> None:
        for handle in self._grad_hooks:
            try:
                handle.remove()
            except Exception as exc:
                logger.warning(f"[monitor_v2] Failed to remove grad hook: {exc}")
        self._grad_hooks.clear()
        self._param_micro_steps.clear()

    # ---------------------------------------------------------------------
    #  Optimizer patching
    # ---------------------------------------------------------------------

    def _patch_optimizer(self) -> None:
        if self._patched or not hasattr(self._optimizer, "step"):
            return
        self._orig_step = self._optimizer.step

        def patched_step(*args: Any, **kwargs: Any):
            self._capture_grads("reduced")
            return self._orig_step(*args, **kwargs)

        self._optimizer.step = patched_step  # type: ignore[assignment]
        self._patched = True

    def _restore_optimizer(self) -> None:
        if self._patched and self._orig_step is not None:
            self._optimizer.step = self._orig_step  # type: ignore[assignment]
        self._patched = False
        self._orig_step = None
        self._clear_grad_hooks()

    # ---------------------------------------------------------------------
    #  Recording helpers
    # ---------------------------------------------------------------------

    def _capture_grads(self, kind: str) -> None:
        tag2tensor: Dict[str, Any] = {}
        tag2micro: Dict[str, int] = {}
        micro_step_val = self._micro_batch_number
        for param, name in self._param2name.items():
            grad = self._fetch_param_grad(param)
            if grad is None or not FmkAdp.is_tensor(grad):
                continue
            tag = name
            tag2tensor[tag] = grad
            tag2micro[tag] = micro_step_val
        self._record_grad_dict(kind, tag2tensor, tag2micro)

    def _resolve_micro_batch_number(self) -> int:
        candidates = [
            self.config.get("micro_batch_number"),
            self.config.get("grad_acc_steps"),
            self._context.get("micro_batch_number"),
            self._context.get("grad_acc_steps"),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                value = int(candidate)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return 1

    def _record_grad(self, kind: str, name: str, grad: Any, micro_step: Optional[int] = None) -> None:
        if grad is None or not FmkAdp.is_tensor(grad):
            return
        tag = name
        micro_map = {tag: micro_step} if micro_step is not None else None
        self._record_grad_dict(kind, {tag: grad}, micro_map)

    def _record_grad_dict(
        self,
        kind: str,
        tag2tensor: Dict[str, Any],
        tag2micro: Optional[Dict[str, int]] = None,
    ) -> None:
        if not tag2tensor:
            return
        stats = self._compute_metrics(tag2tensor)
        for tag, op_dict in stats.items():
            vpp_stage = get_vpp_stage_from_tag(tag)
            row = {
                "module_name": tag,
                "scope": kind,
                "stats": {op: self._detach_stat(val) for op, val in op_dict.items()},
            }
            if tag2micro and tag in tag2micro:
                row["micro_step"] = tag2micro[tag]
            if vpp_stage is not None:
                row["vpp_stage"] = vpp_stage
            else:
                row["vpp_stage"] = MonitorConst.DEFAULT_STAGE
            self._rows.append(row)

    def _fetch_param_grad(self, param: Any) -> Any:
        return getattr(param, "grad", None)

    def _detach_stat(self, value: Any) -> Any:
        if value is None or not FmkAdp.is_tensor(value):
            return value
        if FmkAdp.fmk == Const.PT_FRAMEWORK:
            return value.detach()
        return value

    def _parse_eps(self, value: Any) -> float:
        try:
            eps = float(value)
        except (TypeError, ValueError):
            logger.warning(f"[monitor_v2] Invalid eps value: {value}, fallback to 1e-8.")
            return 1e-8
        if eps <= 0:
            logger.warning(f"[monitor_v2] eps must be positive, got {eps}, fallback to 1e-8.")
            return 1e-8
        return eps

    def _parse_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        logger.warning(f"[monitor_v2] Invalid boolean value: {value}, fallback to False.")
        return False




class PyTorchWeightGradMonitorV2(WeightGradMonitorV2):
    def __init__(self) -> None:
        super().__init__()
        self._fsdp_mode: bool = False
        self._orig_autograd_backward = None
        self._fsdp_backward_patched: bool = False
        self._fsdp_micro_step: int = 0

    def start(self, model: Any = None, optimizer: Any = None, **context: Any) -> None:
        self.set_context(**context)
        if self._optimizer is not None:
            return
        model = model or self._context.get("model")
        self._fsdp_mode = self._detect_fsdp(model)
        self._fsdp_micro_step = 0
        super().start(model=model, optimizer=optimizer, **context)
        if self._fsdp_mode:
            self._patch_autograd_backward()

    def stop(self) -> None:
        self._restore_autograd_backward()
        self._fsdp_mode = False
        self._fsdp_micro_step = 0
        super().stop()

    def _register_pre_grad_hooks(self) -> None:
        if self._fsdp_mode:
            self._param_micro_steps.clear()
            return
        super()._register_pre_grad_hooks()

    def _create_param_hook(self, param: Any, name: str):
        try:
            import torch

            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            if grad_acc is None:
                raise RuntimeError
        except Exception:
            return super()._create_param_hook(param, name)

        @torch.no_grad()
        def acc_hook(*_unused):
            grad = self._fetch_param_grad(param)
            self._on_param_grad(param, name, grad)

        self._param_micro_steps[param] = 0
        return grad_acc.register_hook(acc_hook)

    def _fetch_param_grad(self, param: Any) -> Any:
        grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = getattr(param, "grad", None)
        if grad is None:
            return None
        if grad.__class__.__name__ == "DTensor":
            try:
                grad = grad.to_local()
            except Exception as exc:
                logger.warning(f"[monitor_v2] Failed to convert DTensor grad to local: {exc}")
        try:
            from msprobe.pytorch.common.utils import is_float8_tensor

            if is_float8_tensor(grad):
                grad = grad.float()
        except Exception as exc:
            logger.warning(f"[monitor_v2] Failed to convert grad tensor from float8: {exc}")
        try:
            return grad.clone()
        except Exception:
            return grad

    def _compute_metrics(self, tag2tensor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        from msprobe.pytorch.monitor.module_metric import get_metrics

        return get_metrics(self._ops, tag2tensor, self._eps, {})

    def _detect_fsdp(self, model: Any) -> bool:
        if model is None:
            return False
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore[import]
        except Exception:
            FSDP = None
        for chunk, _prefix in iter_model_chunks(model):
            if FSDP is not None and isinstance(chunk, FSDP):
                if getattr(chunk, "_use_orig_params", False):
                    return True
            try:
                for _param_name, param in chunk.named_parameters():
                    if param.__class__.__name__ == "DTensor":
                        return True
            except Exception:
                continue
        return False

    def _patch_autograd_backward(self) -> None:
        if self._fsdp_backward_patched:
            return
        try:
            import torch.autograd as _autograd
        except Exception:
            logger.warning("[monitor_v2] Failed to import torch.autograd, skipping FSDP grad patch.")
            return

        orig = getattr(_autograd, "backward", None)
        if orig is None:
            return

        def wrapped_backward(*args: Any, **kwargs: Any):
            out = orig(*args, **kwargs)
            self._capture_fsdp_pre_grads()
            return out

        self._orig_autograd_backward = orig
        _autograd.backward = wrapped_backward
        self._fsdp_backward_patched = True

    def _restore_autograd_backward(self) -> None:
        if not self._fsdp_backward_patched or self._orig_autograd_backward is None:
            self._fsdp_backward_patched = False
            self._orig_autograd_backward = None
            return
        try:
            import torch.autograd as _autograd

            _autograd.backward = self._orig_autograd_backward
        except Exception as exc:
            logger.warning(f"[monitor_v2] Failed to restore torch.autograd.backward: {exc}")
        self._fsdp_backward_patched = False
        self._orig_autograd_backward = None

    def _capture_fsdp_pre_grads(self) -> None:
        if self._model is None:
            return
        current_idx = self._fsdp_micro_step + 1
        self._fsdp_micro_step = current_idx

        tag2tensor: Dict[str, Any] = {}
        tag2micro: Dict[str, int] = {}
        should_record = True
        if self.monitor_mbs_grad:
            micro_step_val = current_idx
        else:
            micro_step_val = self._micro_batch_number
            if self._micro_batch_number > 0 and current_idx < self._micro_batch_number:
                should_record = False

        if should_record:
            for param, name in self._param2name.items():
                if not getattr(param, "requires_grad", True):
                    continue
                grad = self._fetch_param_grad(param)
                if grad is None or not FmkAdp.is_tensor(grad):
                    continue
                clean_name = name.replace("_fsdp_wrapped_module.", "")
                tag = clean_name
                tag2tensor[tag] = grad
                tag2micro[tag] = micro_step_val

            self._record_grad_dict("unreduced", tag2tensor, tag2micro)

        if self._micro_batch_number > 0 and current_idx >= self._micro_batch_number:
            self._fsdp_micro_step = 0


class MindSporeWeightGradMonitorV2(WeightGradMonitorV2):
    def _compute_metrics(self, tag2tensor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        from msprobe.mindspore.monitor.utils import get_metrics

        return get_metrics(self._ops, tag2tensor, self._eps, {})
