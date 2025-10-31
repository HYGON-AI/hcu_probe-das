# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import ctypes
import os
import zlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import List
import inspect

import numpy as np
import torch
from torch import distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

from msprobe.core.common.const import Const
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.log import logger
from msprobe.core.common.utils import convert_tuple, is_int
from msprobe.core.data_dump.data_processor.base import BaseDataProcessor, ModuleBackwardInputsOutputs, \
    ModuleForwardInputsOutputs, TensorStatInfo
from msprobe.pytorch.common.utils import save_pt, is_recomputation
from msprobe.pytorch.free_benchmark import FreeBenchmarkCheck, UnequalRow

is_gpu = False
try:
    import torch_npu
except ImportError:
    is_gpu = True


class TensorHandler:
    def __init__(self):
        self.has_dtensor = hasattr(dist, "tensor") and hasattr(dist.tensor, "DTensor")
        self.has_fake_tensor = hasattr(torch, "_subclasses") and hasattr(torch._subclasses, "fake_tensor")
        self.has_async_collective_tensor = hasattr(dist, "_functional_collectives") and \
                                           hasattr(dist._functional_collectives, "AsyncCollectiveTensor")

    @staticmethod
    def free_tensor(tensor, tensor_name):
        try:
            tensor.untyped_storage().resize_(0)
        except Exception as e:
            logger.warning(f"Failed to free tensor: {tensor_name}, the detail info: {e}.")

    def is_dtensor(self, tensor):
        return self.has_dtensor and isinstance(tensor, dist.tensor.DTensor)

    def is_fake_tensor(self, tensor):
        return self.has_fake_tensor and isinstance(tensor, torch._subclasses.fake_tensor.FakeTensor)

    def is_async_collective_tensor(self, tensor):
        return self.has_async_collective_tensor and \
            isinstance(tensor, dist._functional_collectives.AsyncCollectiveTensor)

    def is_empty_data(self, tensor):
        return tensor.is_meta or self.is_fake_tensor(tensor) or self.is_async_collective_tensor(tensor)

    def convert_common_tensor(self, tensor):
        if self.is_dtensor(tensor):
            return tensor.to_local()
        if self.is_fake_tensor(tensor):
            logger.debug("FakeTensor cannot be converted to torch.Tensor type.")
            return tensor
        return tensor

    def get_tensor_type(self, tensor):
        if self.is_dtensor(tensor):
            return Const.DTENSOR_TYPE
        if self.is_fake_tensor(tensor):
            return Const.FAKE_TENSOR_TYPE
        if self.is_async_collective_tensor(tensor):
            return Const.AC_TENSOR_TYPE
        return Const.TENSOR_TYPE

    def get_dtensor_info(self, tensor):
        dtensor_info = {}
        if not self.is_dtensor(tensor):
            return dtensor_info
        if hasattr(tensor, "device_mesh") and tensor.device_mesh:
            dtensor_info.update({"device_mesh": tensor.device_mesh.mesh.tolist()})

        placements = []
        if hasattr(tensor, "placements") and isinstance(tensor.placements, Iterable):
            for placement in tensor.placements:
                if placement.is_shard() and is_int(placement.dim):
                    placements.append({"Shard": {"dim": placement.dim}})
                    continue
                if placement.is_replicate():
                    placements.append({"Replicate": {}})
                    continue
                if placement.is_partial() and isinstance(placement.reduce_op, str):
                    placements.append({"Partial": {"reduce_op": placement.reduce_op}})
        dtensor_info.update({"placements": placements})
        return dtensor_info

    def save_tensor(self, tensor, file_path):
        common_tensor = self.convert_common_tensor(tensor)
        if self.is_empty_data(common_tensor):
            logger.debug(f"Saving fake tensor or meta tensor is not supported, the current tensor is {file_path}.")
            return
        if common_tensor.untyped_storage().data_ptr() == 0:
            logger.debug(f"Saving null-pointer tensor is not supported, the current tensor is {file_path}.")
            return
        saved_tensor = common_tensor.clone().contiguous().detach()
        save_pt(saved_tensor, file_path)
        self.free_tensor(saved_tensor, file_path)


class PytorchDataProcessor(BaseDataProcessor):
    pytorch_special_type = (
        torch.device,
        torch.dtype,
        torch.Size,
        torch.Tensor,
        torch.memory_format,
        dist.ProcessGroup,
        dist.P2POp,
        dist.ReduceOp
    )
    memory_format = {
        torch.contiguous_format: "contiguous_format",
        torch.channels_last: "channels_last",
        torch.channels_last_3d: "channels_last_3d",
        torch.preserve_format: "preserve_format"
    }

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.torch_object_key = {
            "device": self.analyze_device_in_kwargs,
            "dtype": self.analyze_dtype_in_kwargs
        }
        self._async_dump_cache = {}
        self.tensor_handler = TensorHandler()
        self._crc_executor = ThreadPoolExecutor(max_workers=os.cpu_count() // 2)

    @staticmethod
    def get_md5_for_tensor(x):
        if x.dtype == torch.bfloat16:
            x = x.float()
        tensor_bytes = x.cpu().detach().numpy().tobytes()
        crc32_hash = zlib.crc32(tensor_bytes)
        return f"{crc32_hash:08x}"

    @staticmethod
    def tensor_bytes_view_cpu(t: torch.Tensor):
        """
        返回 t 在当前 dtype 下的原始字节视图（优先零拷贝）。
        需保证：t 已在 CPU 且是 contiguous。
        可能返回 memoryview 或 bytes（兜底拷贝）或者 转为numpy，均可被 zlib.crc32 接受。
        """

        nbytes = t.numel() * t.element_size()
        byte_offset = t.storage_offset() * t.element_size()

        if nbytes == 0:
            return memoryview(b"")

        storage = t.untyped_storage()

        # ctypes 指针构造 memoryview（零拷贝 FFI）
        try:
            addr = storage.data_ptr() + byte_offset
            buf = (ctypes.c_ubyte * nbytes).from_address(addr)
            mv3 = memoryview(buf)

            return mv3
        except Exception as e1:
            logger.warning(f"path_A_failed: {e1}.")

        try:
            data = ctypes.string_at(storage.data_ptr() + byte_offset, nbytes)

            return data  # bytes 也可直接用于 zlib.crc32
        except Exception as e2:
            logger.warning(f"path_B_failed: {e2}.")

        try:
            if t.dtype == torch.bfloat16:
                t = t.float()
            data = t.numpy()

            return data
        except Exception as e3:
            logger.warning(f"path_C_failed: {e3}.")
            return memoryview(b"")

    @staticmethod
    def compute_crc32_from_tensor(t: torch.Tensor) -> str:
        """
        直接对 Tensor 原始字节做 CRC32。
        :
        - "raw": 保持 bfloat16 原始 16bit 字节（推荐，避免升精/增容）
        """

        # 取得字节视图（含多级回退），然后做 CRC
        mv = PytorchDataProcessor.tensor_bytes_view_cpu(t)

        crc = zlib.crc32(mv)

        return f"{crc:08x}"

    @staticmethod
    def analyze_device_in_kwargs(element):
        single_arg = {}
        single_arg.update({'type': "torch.device"})
        if isinstance(element, (int, str)):
            single_arg.update({"value": element})
        elif isinstance(element, torch.device):
            if hasattr(element, "index"):
                device_value = element.type + ":" + str(element.index)
            else:
                device_value = element.type
            single_arg.update({"value": device_value})
        else:
            logger.debug(f"Device type {type(element)} is not supported.")
        return single_arg

    @staticmethod
    def analyze_dtype_in_kwargs(element):
        return {"type": "torch.dtype", "value": str(element)}

    @staticmethod
    def process_group_hash(arg):
        group_ranks = dist.get_process_group_ranks(arg)
        group_ranks_hash = zlib.crc32(str(group_ranks).encode('utf-8'))
        return f"{group_ranks_hash:08x}"

    @staticmethod
    def is_hookable_element(element):
        return (hasattr(element, "register_hook") and callable(element.register_hook)) and \
            (hasattr(element, "requires_grad") and element.requires_grad)

    @staticmethod
    def is_recompute(call_stack=None):
        return is_recomputation(call_stack)

    @staticmethod
    def analyze_api_call_stack(name):
        try:
            call_stack = inspect.stack()
            if name.startswith("Primitive"):
                api_stack = call_stack[4:]
            else:
                api_stack = call_stack[5:]
        except Exception as e:
            logger.warning(f"The call stack of <{name}> failed to retrieve, {e}.")
            api_stack = None
            call_stack = None

        stack_str = []
        if api_stack:
            for (_, path, line, func, code, _) in api_stack:
                if not code:
                    continue
                if any(filter_path in path for filter_path in Const.STACK_FILTER_KEYWORDS) and \
                        Const.CALL_STACK_FLAG not in path:
                    continue
                stack_line = f"File {path}, line {str(line)}, in {func}, \n {code[0].strip()}"
                stack_str.append(stack_line)
        else:
            stack_str.append(Const.WITHOUT_CALL_STACK)
        is_recompute = PytorchDataProcessor.is_recompute(call_stack)
        del call_stack
        return tuple(stack_str), is_recompute

    @staticmethod
    def _analyze_torch_size(arg):
        return {"type": "torch.Size", "value": [int(x) for x in list(arg)]}

    @staticmethod
    def _analyze_memory_format(arg):
        # 获取内存格式
        format_type = PytorchDataProcessor.memory_format.get(arg)
        return {"type": "torch.memory_format", "format": format_type}

    @staticmethod
    def _analyze_process_group(arg):
        group_info = {"type": "torch.ProcessGroup"}
        try:
            group_ranks = dist.get_process_group_ranks(arg)
            group_info.update({"group_ranks": group_ranks})
            group_id = PytorchDataProcessor.process_group_hash(arg)
            group_info.update({"group_id": group_id})
        except Exception as e:
            logger.warning(f"Failed to get process group ranks info with error info: {e}.")
        return group_info

    @staticmethod
    def _analyze_reduce_op(arg):
        op_type = None
        try:
            op_type = str(arg)
        except Exception as e:
            logger.warning(f"Failed to get value of torch.distributed.ReduceOp with error info: {e}.")
        return {"type": "torch.distributed.ReduceOp", "value": op_type}

    @classmethod
    def get_special_types(cls):
        return super().get_special_types() + cls.pytorch_special_type

    def get_stat_info(self, data, async_dump=False, precision=Const.DUMP_PRECISION_LOW):
        tensor_stat = TensorStatInfo()
        if self.tensor_handler.is_empty_data(data):
            return tensor_stat
        data_clone = data.detach()
        if not data_clone.numel() or not data_clone.data_ptr():
            return tensor_stat
        if torch.is_complex(data_clone):
            if async_dump:
                logger.warning("Async dump do not support complex data!")
                return tensor_stat
            data_np = data_clone.cpu().numpy()
            data_abs = np.abs(data_np)
            tensor_stat.max = np.max(data_abs).item()
            tensor_stat.min = np.min(data_abs).item()
            tensor_stat.mean = np.mean(data_abs).item()
        elif data_clone.dtype == torch.bool:
            tensor_stat.max = torch.any(data_clone)
            tensor_stat.min = torch.all(data_clone)
        elif not data_clone.shape:
            tensor_stat.max = tensor_stat.min = tensor_stat.mean = tensor_stat.norm = data_clone.clone()
        else:
            if (precision == Const.DUMP_PRECISION_HIGH or data_clone.dtype == torch.float64
                    or not data_clone.is_floating_point()):
                data_clone = data_clone.float()
            tensor_stat.max = torch.max(data_clone)
            tensor_stat.min = torch.min(data_clone)
            tensor_stat.mean = torch.mean(data_clone)
            tensor_stat.norm = torch.norm(data_clone)
        return tensor_stat

    def dump_async_data(self):
        for file_path, tensor in self._async_dump_cache.items():
            self.tensor_handler.save_tensor(tensor, file_path)
        self._async_dump_cache.clear()

    def analyze_single_element(self, element, suffix_stack):
        if suffix_stack and suffix_stack[-1] in self.torch_object_key:
            return self.torch_object_key[suffix_stack[-1]](element)

        suffix_str = Const.SEP.join(str(s) for s in suffix_stack)
        type_analyzer = [
            (PytorchDataProcessor.builtin_type, self._analyze_builtin),
            (torch.Size, self._analyze_torch_size),
            (torch.Tensor, lambda e: self._analyze_tensor(e, suffix_str)),
            (torch.memory_format, self._analyze_memory_format),
            (dist.ProcessGroup, self._analyze_process_group),
            (dist.P2POp, lambda e: self._analyze_p2pop(e, suffix_str)),
            (dist.ReduceOp, self._analyze_reduce_op),
            (PytorchDataProcessor.np_type[:-1], self._analyze_numpy),
            (np.ndarray, lambda e: self._analyze_ndarray(e, suffix_str)),
        ]
        for type_key, analyze_fn in type_analyzer:
            if isinstance(element, type_key):
                return analyze_fn(element)
        return {}

    def _analyze_p2pop(self, arg, suffix):
        p2pop_info = {"class_type": "torch.distributed.P2POp"}
        try:
            tensor_info = self._analyze_tensor(arg.tensor, suffix)
            p2pop_info.update({"tensor": tensor_info})
            p2pop_info.update({"op": arg.op.__name__})
            p2pop_info.update({"peer": arg.peer})
            p2pop_info.update({"tag": arg.tag})
            group_id = PytorchDataProcessor.process_group_hash(
                arg.group) if arg.group else PytorchDataProcessor.process_group_hash(_get_default_group())
            p2pop_info.update({"group_id": group_id})
        except Exception as e:
            logger.warning(f"Failed to parse the P2POp content with error info: {e}.")
        return p2pop_info

    def _analyze_tensor(self, tensor, suffix):
        common_tensor = self.tensor_handler.convert_common_tensor(tensor)
        tensor_stat = self.get_stat_info(common_tensor, self.config.async_dump, self.config.precision)
        tensor_json = {}
        tensor_json.update({'type': self.tensor_handler.get_tensor_type(tensor)})
        tensor_json.update({'dtype': str(common_tensor.dtype)})
        tensor_json.update({"shape": common_tensor.shape})

        stat_values = [
            tensor_stat.max,
            tensor_stat.min,
            tensor_stat.mean,
            tensor_stat.norm
        ]
        placeholder_index = self.data_writer.append_stat_to_buffer(stat_values)

        tensor_json.update({Const.TENSOR_STAT_INDEX: placeholder_index})
        tensor_json.update({"requires_grad": tensor.requires_grad})
        if self.tensor_handler.is_dtensor(tensor):
            dtensor_info = self.tensor_handler.get_dtensor_info(tensor)
            tensor_json.update(dtensor_info)

        if self.config.summary_mode == Const.MD5 and not self.config.async_dump:
            tensor_md5 = None
            if not self.tensor_handler.is_empty_data(tensor):
                t_cpu = common_tensor

                # 根据设备类型做同步，确保数据已准备好
                if t_cpu.device.type == "cuda":
                    t_cpu = t_cpu.to("cpu", non_blocking=True)
                    torch.cuda.synchronize()
                    # 先异步搬运再进行同步可以显著提升性能
                elif t_cpu.device.type == "npu":
                    t_cpu = t_cpu.to("cpu", non_blocking=True)
                    torch.npu.synchronize()

                t_cpu = t_cpu.detach()
                if not t_cpu.is_contiguous():
                    t_cpu = t_cpu.contiguous()

                future = self._crc_executor.submit(
                    PytorchDataProcessor.compute_crc32_from_tensor,
                    t_cpu
                )

                crc_placeholder = self.data_writer.append_crc32_to_buffer(future)
                tensor_json[Const.MD5_INDEX] = crc_placeholder
            else:
                logger.debug(
                    "Calculating the md5 value of fake tensor or meta tensor is not supported, "
                    f"the current api/module name is {self.current_api_or_module_name}."
                )
                tensor_json.update({Const.MD5: tensor_md5})
        return tensor_json

    def _analyze_and_save_tensor(self, tensor, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        single_arg = PytorchDataProcessor._analyze_tensor(self, tensor, suffix)
        common_tensor = self.tensor_handler.convert_common_tensor(tensor)
        if self.tensor_handler.is_empty_data(common_tensor):
            logger.debug(f"Saving fake tensor or meta tensor is not supported, the current tensor is {file_path}.")
            return single_arg
        if common_tensor.untyped_storage().data_ptr() == 0:
            logger.debug(f"Saving null-pointer tensor is not supported, the current tensor is {file_path}.")
            return single_arg

        single_arg.update({"data_name": dump_data_name})
        if self.config.async_dump:
            self._async_dump_cache[file_path] = common_tensor.clone().detach()
        else:
            self.tensor_handler.save_tensor(common_tensor, file_path)
        return single_arg

    def _analyze_and_save_ndarray(self, ndarray, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        self.tensor_handler.save_tensor(torch.tensor(ndarray), file_path)
        ndarray_json = PytorchDataProcessor._analyze_ndarray(ndarray, suffix)
        ndarray_json.update({"data_name": dump_data_name})
        return ndarray_json


class StatisticsDataProcessor(PytorchDataProcessor):
    def _analyze_tensor(self, tensor, suffix):
        if any(item in self.current_api_or_module_name for item in self.config.tensor_list):
            return self._analyze_and_save_tensor(tensor, suffix)
        else:
            return super()._analyze_tensor(tensor, suffix)

    def _analyze_ndarray(self, ndarray, suffix):
        if any(item in self.current_api_or_module_name for item in self.config.tensor_list):
            return self._analyze_and_save_ndarray(ndarray, suffix)
        else:
            return super()._analyze_ndarray(ndarray, suffix)


class TensorDataProcessor(PytorchDataProcessor):
    def _analyze_tensor(self, tensor, suffix):
        return self._analyze_and_save_tensor(tensor, suffix)

    def _analyze_ndarray(self, ndarray, suffix):
        return self._analyze_and_save_ndarray(ndarray, suffix)


class OverflowCheckDataProcessor(PytorchDataProcessor):
    __slots__ = ["cached_tensors_and_file_paths"]

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.has_overflow = False
        self.support_inf_nan = None
        self.cached_api_info = {}
        self.cached_tensors_and_file_paths = {}
        self.bits_for_overflow = 8
        self.real_overflow_nums = 0
        self.overflow_nums = config.overflow_nums

    @property
    def is_terminated(self):
        if self.overflow_nums == -1:
            return False
        if self.real_overflow_nums >= self.overflow_nums:
            return True
        return False

    def analyze_forward_input(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        self.has_overflow = False
        self._is_support_inf_nan()
        self.cached_api_info = super().analyze_forward_input(name, module, module_input_output)
        return None

    def analyze_forward_output(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        self._is_support_inf_nan()
        api_info_struct = super().analyze_forward_output(name, module, module_input_output)
        if name in self.cached_api_info and name in api_info_struct:
            self.cached_api_info[name].update(api_info_struct[name])
        elif name in api_info_struct:
            self.cached_api_info = api_info_struct
        self.handle_overflow()
        return self.cached_api_info if self.has_overflow else None

    def analyze_forward(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        self.has_overflow = False
        self._is_support_inf_nan()
        api_info_struct = super().analyze_forward(name, module, module_input_output)
        self.handle_overflow()
        return api_info_struct if self.has_overflow else None

    def analyze_backward(self, name, module, module_input_output: ModuleBackwardInputsOutputs):
        self.has_overflow = False
        self._is_support_inf_nan()
        api_info_struct = super().analyze_backward(name, module, module_input_output)
        self.handle_overflow()
        return api_info_struct if self.has_overflow else None

    def analyze_params(self, name, param_name, grad):
        self.has_overflow = False
        self._is_support_inf_nan()
        api_info_struct = super().analyze_params(name, param_name, grad)
        self.handle_overflow()
        return api_info_struct if self.has_overflow else None

    def handle_overflow(self):
        if not self.support_inf_nan:
            self._analyze_maybe_overflow_flag()
        if self.has_overflow:
            for file_path, tensor in self.cached_tensors_and_file_paths.items():
                self.tensor_handler.save_tensor(tensor, file_path)
            self.real_overflow_nums += 1
            if self.overflow_nums != -1 and self.real_overflow_nums >= self.overflow_nums:
                logger.info(f"[{Const.TOOL_NAME}] Reached the preset overflow times, "
                            f"current overflow times: {self.real_overflow_nums}.")
        self.cached_tensors_and_file_paths = {}

    def _is_support_inf_nan(self):
        if self.support_inf_nan is not None:
            return
        try:
            self.support_inf_nan = is_gpu or torch_npu.npu.utils.is_support_inf_nan()
        except Exception:
            logger.warning(f"Unable to determine if the current device supports inf/nan mode, default not supported.")
            self.support_inf_nan = False

    def _analyze_maybe_overflow_flag(self):
        try:
            self.has_overflow = torch_npu.npu.utils.get_npu_overflow_flag()
            if self.has_overflow:
                torch_npu.npu.utils.clear_npu_overflow_flag()
        except Exception as e:
            logger.error(f"Overflow check failed, the current environment may be abnormal.")
            raise RuntimeError(f"overflow check failed") from e

    def _analyze_maybe_overflow_tensor(self, tensor_json):
        tensor_stat_index = tensor_json.get(Const.TENSOR_STAT_INDEX)
        if tensor_stat_index is None:
            logger.warning("tensor_stat_index does not exist in tensor_json.")
            return
        max_tensor = self.data_writer.get_buffer_values_max(tensor_stat_index)
        min_tensor = self.data_writer.get_buffer_values_min(tensor_stat_index)

        if max_tensor is None or min_tensor is None:
            return

        if torch.isinf(max_tensor) or torch.isnan(max_tensor):
            self.has_overflow = True
            return

        if torch.isinf(min_tensor) or torch.isnan(min_tensor):
            self.has_overflow = True

    def _analyze_tensor(self, tensor, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        self.cached_tensors_and_file_paths.update({file_path: tensor})
        single_arg = super()._analyze_tensor(tensor, suffix)
        single_arg.update({"data_name": dump_data_name})
        if not self.has_overflow and self.support_inf_nan:
            self._analyze_maybe_overflow_tensor(single_arg)
        return single_arg


class FreeBenchmarkDataProcessor(PytorchDataProcessor):

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.checker = FreeBenchmarkCheck(config=config)
        self._return_forward_new_output = None
        self._forward_new_output = None

    def update_iter(self, current_iter):
        super().update_iter(current_iter)
        self.checker.update_iter(current_iter)

    def update_unequal_rows(self, unequal_rows: List[UnequalRow]):
        if not unequal_rows:
            return
        for row in unequal_rows:
            data_dict = asdict(row)
            self.data_writer.write_data_to_csv(
                data_dict.values(),
                data_dict.keys(),
                self.data_writer.free_benchmark_file_path
            )
        return

    def analyze_forward_input(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        self.checker.pre_forward(name, module, self, module_input_output.args, module_input_output.kwargs)

    def analyze_forward_output(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        new_output, unequal_rows = self.checker.forward(
            name,
            module,
            module_input_output.args,
            module_input_output.kwargs,
            module_input_output.output,
        )
        self.update_unequal_rows(unequal_rows)
        if self.checker.if_fix():
            self._return_forward_new_output = True
            self._forward_new_output = new_output

    def analyze_backward(self, name, module, module_input_output: ModuleBackwardInputsOutputs):
        self.checker.backward(name, module, module_input_output.grad_input)


class KernelDumpDataProcessor(PytorchDataProcessor):
    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.enable_kernel_dump = True
        self.is_found_output_tensor = False
        self.is_found_grad_input_tensor = False
        self.forward_args = None
        self.forward_kwargs = None
        self.forward_output_tensor = None
        self.grad_input_tensor = None

    @staticmethod
    def start_kernel_dump(config_path):
        torch_npu.npu.synchronize()
        torch_npu.npu.init_dump()
        torch_npu.npu.set_dump(config_path)
        torch_npu.npu.synchronize()

    @staticmethod
    def stop_kernel_dump():
        torch_npu.npu.synchronize()
        torch_npu.npu.finalize_dump()
        torch_npu.npu.synchronize()

    @staticmethod
    def _print_unsupported_log(api_name):
        logger.warning(f"The kernel dump does not support the {api_name} API.")

    def analyze_forward_input(self, name, module, module_input_output):
        if not self.enable_kernel_dump:
            return
        if is_gpu:
            logger.warning("The current environment is not a complete NPU environment, and kernel dump cannot be used.")
            self.enable_kernel_dump = False
            return

        if self.config.is_backward_kernel_dump:
            try:
                self.forward_args = self.clone_and_detach_tensor(module_input_output.args)
                self.forward_kwargs = self.clone_and_detach_tensor(module_input_output.kwargs)
                output = module.forward(*self.forward_args, **self.forward_kwargs)
            except Exception as e:
                if isinstance(e, MsprobeException):
                    logger.warning(str(e))
                self._print_unsupported_log(name)
                self.enable_kernel_dump = False
                return

            self.analyze_element(convert_tuple(output))
            if not self.is_found_output_tensor:
                self._print_unsupported_log(name)
                self.enable_kernel_dump = False
            return
        self.start_kernel_dump(self.config.kernel_config_path)

    def analyze_forward_output(self, name, module, module_input_output):
        if not self.enable_kernel_dump:
            return
        if self.config.is_backward_kernel_dump:
            return
        self.enable_kernel_dump = False
        self.stop_kernel_dump()
        logger.info(f"The kernel data of {name} is dumped successfully.")

    def analyze_backward(self, name, module, module_input_output):
        if not self.enable_kernel_dump:
            return
        self.enable_kernel_dump = False

        self.analyze_element(module_input_output.grad_input)
        if not self.is_found_grad_input_tensor:
            self._print_unsupported_log(name)
            return
        self.start_kernel_dump(self.config.kernel_config_path)

        try:
            self.forward_output_tensor.backward(self.grad_input_tensor, retain_graph=True)
        except Exception:
            self._print_unsupported_log(name)
            self.stop_kernel_dump()
            return

        self.stop_kernel_dump()
        logger.info(f"The kernel data of {name} is dumped successfully.")

    @recursion_depth_decorator(
        "KernelDump: KernelDumpDataProcessor.clone_and_detach_tensor",
        max_depth=Const.DUMP_MAX_DEPTH
    )
    def clone_and_detach_tensor(self, input_params):
        if isinstance(input_params, torch.Tensor):
            if input_params.requires_grad:
                return input_params.clone().detach().requires_grad_()
            return input_params.clone()
        elif isinstance(input_params, tuple):
            return tuple(self.clone_and_detach_tensor(x) for x in input_params)
        elif isinstance(input_params, list):
            return list(self.clone_and_detach_tensor(x) for x in input_params)
        elif isinstance(input_params, dict):
            return {k: self.clone_and_detach_tensor(v) for k, v in input_params.items()}
        else:
            return input_params

    def analyze_single_element(self, element, suffix_stack):
        if isinstance(element, torch.Tensor):
            if not self.is_found_output_tensor:
                if element.requires_grad:
                    self.forward_output_tensor = element
                    self.is_found_output_tensor = True
                return {}
            if not self.is_found_grad_input_tensor:
                self.grad_input_tensor = element.clone()
                self.is_found_grad_input_tensor = True
        return {}

    def reset_status(self):
        self.enable_kernel_dump = True
        self.is_found_output_tensor = False
        self.is_found_grad_input_tensor = False
        self.forward_args = None
        self.forward_kwargs = None
        self.forward_output_tensor = None
        self.grad_input_tensor = None
