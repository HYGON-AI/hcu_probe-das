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

from typing import Any, Dict, Iterable, Optional, Tuple

from msprobe.core.common.const import MonitorConst
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.log import logger


def iter_model_chunks(model: Any) -> Iterable[Tuple[Any, str]]:
    """
    Iterate over model or model_list, yielding (model_chunk, prefix).

    - Single model: yields (model, "")
    - List/tuple: yields (sub_model, "<idx><NAME_SEP>")
    """
    if model is None:
        # 保持与后续分支一致的生成器行为
        return
        # 不再显式 return ()

    if isinstance(model, (list, tuple)):
        for idx, sub in enumerate(model):
            if sub is None:
                continue
            prefix = f"{idx}{MonitorConst.NAME_SEP}"
            yield sub, prefix
        return
        # 同样去掉显式的 return ()

    yield model, ""


def iter_model_params(model: Any) -> Iterable[Tuple[str, Any]]:
    """
    Iterate (name, param) pairs for a model that may be:
      - a single nn.Module/nn.Cell
      - a list/tuple of models

    For list/tuple, each sub-model is prefixed with
    "<idx><NAME_SEP>" to keep names unique and consistent
    with other multi-model monitor behaviours.
    """
    for chunk, prefix in iter_model_chunks(model):
        try:
            for name, param in FmkAdp.named_parameters(chunk):
                yield prefix + name, param
        except Exception:
            # If a sub-element is not a valid module, skip it.
            continue


def iter_model_modules(model: Any) -> Iterable[Tuple[str, Any]]:
    """
    Iterate (name, module) pairs for a model or model_list, using
    framework-specific module traversal, with optional vpp_stage prefix.
    """
    for chunk, prefix in iter_model_chunks(model):
        try:
            for name, module in FmkAdp.iter_named_modules(chunk):
                yield prefix + name, module
        except Exception:
            logger.warning("Failed to iterate modules for model chunk, skipping.")
            continue


def build_param2name(model: Any = None, optimizer: Any = None) -> Dict[Any, str]:
    """
    Build a param->name mapping from model/optimizer.

    - Primary source: model (or list/tuple of models), using framework
      specific named_parameters().
    - Fallback: optimizer.param_groups when model mapping is empty.
    """
    param2name: Dict[Any, str] = {}

    for name, param in iter_model_params(model):
        param2name[param] = name

    if param2name or optimizer is None:
        return param2name

    idx = 0
    for group in getattr(optimizer, "param_groups", []):
        for param in group.get("params", []):
            if param not in param2name:
                param2name[param] = f"param_{idx}"
                idx += 1

    return param2name


def get_vpp_stage_from_tag(tag: str) -> Optional[int]:
    """
    Extract vpp_stage index from a tag or name that may start with
    "<idx><NAME_SEP>". Returns None if no valid stage prefix exists.
    """
    if not tag:
        return None
    prefix, sep, _ = tag.partition(MonitorConst.NAME_SEP)
    if not sep:
        return None
    try:
        return int(prefix)
    except (TypeError, ValueError):
        return None
