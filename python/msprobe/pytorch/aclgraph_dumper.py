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

import functools
import os
from collections.abc import Iterable

import torch

from msprobe.pytorch.aclgraph_dump import acl_stat, get_acl_stat_dict
from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.file_utils import create_directory, check_and_get_real_path, save_json

try:
    import torch_npu
except Exception:
    torch_npu = None


FORWARD_START_MARKER = "__msprobe_fwd_start__"


def _iter_tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, sub in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(sub, next_prefix)
        return
    if isinstance(value, tuple):
        for idx, sub in enumerate(value):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _iter_tensors(sub, next_prefix)
        return
    if isinstance(value, list):
        for idx, sub in enumerate(value):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _iter_tensors(sub, next_prefix)
        return
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        for idx, sub in enumerate(value):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _iter_tensors(sub, next_prefix)


def _is_collectable_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        return False
    if getattr(tensor, "is_meta", False):
        return False
    try:
        _ = tensor.device
    except Exception:
        return False
    return True


class AclGraphDumper:
    def __init__(self, dump_path="./debugger_dump"):
        self.dump_path = self._validate_dump_path(dump_path)
        self.rank_id = self._resolve_rank_id()
        self.model = None
        self.step_id = 0
        self._running = False

    @staticmethod
    def _validate_dump_path(dump_path):
        if not isinstance(dump_path, str):
            raise TypeError("dump_path must be a string")
        dump_path = check_and_get_real_path(dump_path, FileCheckConst.WRITE_ABLE, must_exist=False)
        create_directory(dump_path)
        return dump_path

    @staticmethod
    def _resolve_rank_id():
        dist = getattr(torch, "distributed", None)
        if dist is None or not dist.is_available() or not dist.is_initialized():
            return None
        try:
            return int(dist.get_rank())
        except Exception:
            return None

    def _step_rank_dir(self):
        rank_name = f"rank{self.rank_id}" if self.rank_id is not None else f"pid{os.getpid()}"
        path = os.path.join(self.dump_path, f"step{self.step_id}", rank_name)
        create_directory(path)
        return path

    def _module_scope(self, module_name):
        return module_name if module_name else "__root__"

    @staticmethod
    def _normalize_dtype(dtype):
        dtype_map = {
            "Bool": "torch.bool",
            "Byte": "torch.uint8",
            "Char": "torch.int8",
            "Short": "torch.int16",
            "Int": "torch.int32",
            "Long": "torch.int64",
            "Half": "torch.float16",
            "Float": "torch.float32",
            "Double": "torch.float64",
            "BFloat16": "torch.bfloat16",
            "ComplexFloat": "torch.complex64",
            "ComplexDouble": "torch.complex128",
        }
        return dtype_map.get(dtype, dtype)

    @classmethod
    def _build_tensor_record(cls, record):
        return {
            Const.TYPE: Const.TENSOR_TYPE,
            Const.DTYPE: cls._normalize_dtype(record.get("dtype")),
            Const.SHAPE: record.get("shape"),
            Const.MAX: record.get("max"),
            Const.MIN: record.get("min"),
            Const.MEAN: record.get("mean"),
            Const.NORM: record.get("norm"),
        }

    @classmethod
    def _assign_nested_value(cls, container, path_parts, value):
        current = container
        for idx, part in enumerate(path_parts):
            is_last = idx == len(path_parts) - 1
            if not isinstance(current, dict):
                raise TypeError(f"Expected dict when assigning key path, got {type(current)}")
            if is_last:
                current[part] = value
                return
            if part not in current or current[part] is None:
                current[part] = {}
            current = current[part]

    @classmethod
    def _compress_numeric_tree_to_list(cls, value):
        if isinstance(value, dict):
            converted = {key: cls._compress_numeric_tree_to_list(sub) for key, sub in value.items()}
            keys = list(converted.keys())
            if keys and all(part.isdigit() for part in keys):
                sorted_items = sorted(converted.items(), key=lambda kv: int(kv[0]))
                return [item for _, item in sorted_items]
            return converted
        return value

    @classmethod
    def _parse_stat_key(cls, key):
        marker = ".forward."
        marker_pos = key.find(marker)
        if marker_pos == -1:
            return None

        op_name = key[: marker_pos + len(".forward")]
        tail = key[marker_pos + len(marker):]
        io_candidates = ("input_kwargs", "input", "output")
        for io_name in io_candidates:
            if tail == io_name:
                return op_name, io_name, []
            prefix = io_name + "."
            if tail.startswith(prefix):
                suffix = tail[len(prefix):]
                return op_name, io_name, suffix.split(".") if suffix else []
        return None

    @classmethod
    def _convert_stats_to_dump_data(cls, stats):
        dump_data = {}
        for key, record in stats.items():
            parsed = cls._parse_stat_key(key)
            if parsed is None:
                continue
            op_name, io_name, path_parts = parsed
            op_entry = dump_data.setdefault(op_name, {})
            tensor_record = cls._build_tensor_record(record)

            if io_name == "input":
                container = op_entry.setdefault(Const.INPUT_ARGS, {})
                cls._assign_nested_value(container, path_parts or ["0"], tensor_record)
                continue

            if io_name == "input_kwargs":
                container = op_entry.setdefault(Const.INPUT_KWARGS, {})
                cls._assign_nested_value(container, path_parts or ["0"], tensor_record)
                continue

            if io_name == "output":
                container = op_entry.setdefault(Const.OUTPUT, {})
                cls._assign_nested_value(container, path_parts or ["0"], tensor_record)

        for op_entry in dump_data.values():
            if Const.INPUT_ARGS in op_entry:
                op_entry[Const.INPUT_ARGS] = cls._compress_numeric_tree_to_list(op_entry[Const.INPUT_ARGS])
            if Const.OUTPUT in op_entry:
                op_entry[Const.OUTPUT] = cls._compress_numeric_tree_to_list(op_entry[Const.OUTPUT])

        return dump_data

    def _collect(self, module_name, io_name, value, mark_forward_start=False):
        has_collected = False
        has_marked = False
        for suffix, tensor in _iter_tensors(value):
            if not _is_collectable_tensor(tensor):
                continue
            tag = f"{self._module_scope(module_name)}.{io_name}"
            effective_suffix = suffix
            if mark_forward_start and not has_marked:
                effective_suffix = (
                    FORWARD_START_MARKER if not suffix else f"{FORWARD_START_MARKER}.{suffix}"
                )
                has_marked = True
            if effective_suffix:
                tag = f"{tag}.{effective_suffix}"
            acl_stat(tensor, tag)
            has_collected = True
        return has_collected

    def _patch(self, model):
        self.model = model
        dumper = self

        for module_name, module in model.named_modules():
            if hasattr(module, "_msprobe_aclgraph_origin_forward"):
                continue
            origin = module.forward

            @functools.wraps(origin)
            def wrapped_forward(*args, __origin=origin, __module_name=module_name, **kwargs):
                started = False
                if dumper._running:
                    collected = dumper._collect(__module_name, "input", args, mark_forward_start=not started)
                    started = started or collected
                    if kwargs:
                        collected = dumper._collect(
                            __module_name, "input_kwargs", kwargs, mark_forward_start=not started
                        )
                        started = started or collected

                output = __origin(*args, **kwargs)

                if dumper._running:
                    dumper._collect(__module_name, "output", output, mark_forward_start=not started)
                return output

            module.forward = wrapped_forward
            module._msprobe_aclgraph_origin_forward = origin

    def _synchronize(self):
        if torch_npu is not None:
            try:
                torch_npu.npu.synchronize()
                return
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, model):
        self.rank_id = self._resolve_rank_id()
        self._patch(model)
        self._running = True

    def step(self):
        if not self._running:
            return

        self._synchronize()
        stats = dict(get_acl_stat_dict(clear=True))
        dump_json = {
            "task": Const.STATISTICS,
            "level": Const.LEVEL_L0,
            "framework": Const.PT_FRAMEWORK,
            "dump_data_dir": None,
            "data": self._convert_stats_to_dump_data(stats),
        }
        file_path = os.path.join(self._step_rank_dir(), "dump.json")
        save_json(file_path, dump_json, indent=2)
        self.step_id += 1
