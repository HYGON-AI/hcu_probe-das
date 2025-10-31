# Copyright 2024-2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
# ============================================================================

import os
import zlib
from concurrent.futures import ThreadPoolExecutor
import inspect

import mindspore as ms
from mindspore import mint, ops, hal
from mindspore.mint import distributed
from mindspore._c_expression.typing import Number
import numpy as np

from msprobe.core.common.const import Const
from msprobe.core.data_dump.data_processor.base import (BaseDataProcessor, TensorStatInfo,
                                                        ModuleForwardInputsOutputs, ModuleBackwardInputsOutputs)
from msprobe.core.common.file_utils import path_len_exceeds_limit
from msprobe.mindspore.common.utils import convert_bf16_to_fp32, save_tensor_as_npy
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.dump.hook_cell.api_register import get_api_register
from msprobe.mindspore.common.utils import is_mindtorch

has_adump = True
try:
    from msprobe.lib import _msprobe_c
except ImportError:
    has_adump = False

if is_mindtorch():
    from torch import distributed as dist


class MindsporeDataProcessor(BaseDataProcessor):
    if is_mindtorch():
        mindspore_special_type = tuple([ms.Tensor, Number, distributed.P2POp, dist.ProcessGroup])
    else:
        mindspore_special_type = tuple([ms.Tensor, Number, distributed.P2POp])

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.mindspore_object_key = {
            "dtype": self.analyze_dtype_in_kwargs
        }
        self._async_dump_cache = {}
        self.api_register = get_api_register()
        self._crc_executor = ThreadPoolExecutor(max_workers=os.cpu_count() // 2)

    @staticmethod
    def compute_crc32_bytes(tensor_bytes):
        return f"{zlib.crc32(tensor_bytes):08x}"

    @staticmethod
    def get_md5_for_tensor(x):
        x = convert_bf16_to_fp32(x)
        tensor_bytes = x.asnumpy().tobytes()
        crc32_hash = zlib.crc32(tensor_bytes)
        return f"{crc32_hash:08x}"

    @staticmethod
    def analyze_dtype_in_kwargs(element):
        return {"type": "mindspore.dtype", "value": str(element)}

    @staticmethod
    def is_hookable_element(element):
        return hasattr(element, "register_hook") and callable(element.register_hook)

    @staticmethod
    def process_group_hash(arg):
        group_ranks = distributed.get_process_group_ranks(arg)
        group_ranks_hash = zlib.crc32(str(group_ranks).encode('utf-8'))
        return f"{group_ranks_hash:08x}"

    @staticmethod
    def is_recompute(call_stack=None):
        return None

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
        del call_stack
        return tuple(stack_str), None

    @staticmethod
    def _analyze_process_group(arg):
        group_info = {"type": "mindspore.ProcessGroup"}
        try:
            group_ranks = dist.get_process_group_ranks(arg)
            group_info.update({"group_ranks": group_ranks})
            group_ranks_hash = zlib.crc32(str(group_ranks).encode('utf-8'))
            group_id = f"{group_ranks_hash:08x}"
            group_info.update({"group_id": group_id})
        except Exception as e:
            logger.warning(f"Failed to get process group ranks info with error info: {e}.")
        return group_info

    @classmethod
    def get_special_types(cls):
        return super().get_special_types() + cls.mindspore_special_type

    def dump_async_data(self):
        for file_path, tensor in self._async_dump_cache.items():
            save_tensor_as_npy(tensor, file_path)
        self._async_dump_cache.clear()

    def get_stat_info(self, data):
        self.api_register.restore_inner_used_api()
        tensor_stat = TensorStatInfo()
        if data.numel() == 0:
            pass
        elif data.dtype == ms.bool_:
            if self.config.async_dump:
                tensor_stat.max = mint.any(data)
                tensor_stat.min = mint.all(data)
            else:
                data_np = data.asnumpy()
                tensor_stat.max = np.max(data_np).item()
                tensor_stat.min = np.min(data_np).item()
        elif not data.shape:
            tensor_stat.max = tensor_stat.min = tensor_stat.mean = tensor_stat.norm = data.copy()
        elif data.dtype == ms.complex64 or data.dtype == ms.complex128:
            if self.config.async_dump:
                logger.warning("Async dump do not support complex data!")
            else:
                data_abs = np.abs(data.asnumpy())
                tensor_stat.max = np.max(data_abs).item()
                tensor_stat.min = np.min(data_abs).item()
                tensor_stat.mean = np.mean(data_abs).item()
                tensor_stat.norm = np.linalg.norm(data_abs).item()
        else:
            if self.config.precision == Const.DUMP_PRECISION_HIGH or not ops.is_floating_point(
                    data) or data.dtype == ms.float64:
                data = data.to(ms.float32)
            get_norm_value = mint.norm if hasattr(mint, "norm") else ops.norm
            tensor_stat.max = mint.max(data)
            tensor_stat.min = mint.min(data)
            tensor_stat.mean = mint.mean(data)
            tensor_stat.norm = get_norm_value(data)
        self.api_register.register_inner_used_api()
        return tensor_stat

    def analyze_single_element(self, element, suffix_stack):
        if suffix_stack and suffix_stack[-1] in self.mindspore_object_key:
            return self.mindspore_object_key[suffix_stack[-1]](element)

        suffix_str = Const.SEP.join(str(s) for s in suffix_stack)
        type_analyzer = [
            (MindsporeDataProcessor.builtin_type, self._analyze_builtin),
            (ms.Tensor, lambda e: self._analyze_tensor(e, suffix_str)),
            (Number, self.analyze_dtype_in_kwargs),
            (MindsporeDataProcessor.np_type[:-1], self._analyze_numpy),
            (np.ndarray, lambda e: self._analyze_ndarray(e, suffix_str)),
            (distributed.P2POp, lambda e: self._analyze_p2pop(e, suffix_str))
        ]
        if is_mindtorch():
            type_analyzer.append((dist.ProcessGroup, self._analyze_process_group))
        for type_key, analyze_fn in type_analyzer:
            if isinstance(element, type_key):
                return analyze_fn(element)
        return {}

    def _analyze_p2pop(self, arg, suffix):
        p2pop_info = {"class_type": "mindspore.mint.distributed.P2POp"}
        try:
            tensor_info = self._analyze_tensor(arg.tensor, suffix)
            p2pop_info.update({"tensor": tensor_info})
            p2pop_info.update({"op": arg.op})
            p2pop_info.update({"peer": arg.peer})
            p2pop_info.update({"tag": arg.tag})
            group_id = self.process_group_hash(arg.group) if arg.group else None
            p2pop_info.update({"group_id": group_id})
        except Exception as e:
            logger.warning(f"Failed to parse the P2POp content with error info: {e}.")
        return p2pop_info

    def _analyze_tensor(self, tensor, suffix):
        tensor_stat = self.get_stat_info(tensor)
        tensor_json = {
            'type': 'mindspore.Tensor',
            'dtype': str(tensor.dtype),
            'shape': tensor.shape
        }

        # 将统计值存入全局 buffer，并返回占位索引
        stat_values = [
            tensor_stat.max,
            tensor_stat.min,
            tensor_stat.mean,
            tensor_stat.norm
        ]

        placeholder_index = self.data_writer.append_stat_to_buffer(stat_values)

        tensor_json.update({Const.TENSOR_STAT_INDEX: placeholder_index})

        if self.config.summary_mode == Const.MD5 and not self.config.async_dump:
            tensor = convert_bf16_to_fp32(tensor)
            # 拷贝并搬到 CPU
            tensor_bytes = tensor.asnumpy()

            future = self._crc_executor.submit(
                MindsporeDataProcessor.compute_crc32_bytes,
                tensor_bytes
            )

            crc_placeholder = self.data_writer.append_crc32_to_buffer(future)
            tensor_json[Const.MD5_INDEX] = crc_placeholder

        return tensor_json

    def _analyze_and_save_tensor(self, tensor, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        single_arg = MindsporeDataProcessor._analyze_tensor(self, tensor, suffix)
        single_arg.update({"data_name": dump_data_name})
        if self.config.async_dump:
            self._async_dump_cache[file_path] = tensor.copy()
        else:
            save_tensor_as_npy(tensor, file_path)
        return single_arg


class StatisticsDataProcessor(MindsporeDataProcessor):
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


class TensorDataProcessor(MindsporeDataProcessor):
    def _analyze_tensor(self, tensor, suffix):
        return self._analyze_and_save_tensor(tensor, suffix)

    def _analyze_ndarray(self, ndarray, suffix):
        return self._analyze_and_save_ndarray(ndarray, suffix)


class OverflowCheckDataProcessor(MindsporeDataProcessor):
    __slots__ = ["cached_tensors_and_file_paths"]

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.has_overflow = False
        self.cached_api_info = {}
        self.cached_tensors_and_file_paths = {}
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
        self.cached_api_info = super().analyze_forward_input(name, module, module_input_output)
        return None

    def analyze_forward_output(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        api_info_struct = super().analyze_forward_output(name, module, module_input_output)
        if name in self.cached_api_info and name in api_info_struct:
            self.cached_api_info[name].update(api_info_struct[name])
        elif name in api_info_struct:
            self.cached_api_info = api_info_struct
        self.maybe_save_overflow_data()
        return self.cached_api_info if self.has_overflow else None

    def analyze_forward(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        self.has_overflow = False
        api_info_struct = super().analyze_forward(name, module, module_input_output)
        self.maybe_save_overflow_data()
        return api_info_struct if self.has_overflow else None

    def analyze_backward(self, name, module, module_input_output: ModuleBackwardInputsOutputs):
        self.has_overflow = False
        api_info_struct = super().analyze_backward(name, module, module_input_output)
        self.maybe_save_overflow_data()
        return api_info_struct if self.has_overflow else None

    def analyze_params(self, name, param_name, grad):
        self.has_overflow = False
        api_info_struct = super().analyze_params(name, param_name, grad)
        self.maybe_save_overflow_data()
        return api_info_struct if self.has_overflow else None

    def maybe_save_overflow_data(self):
        if self.has_overflow:
            for file_path, tensor in self.cached_tensors_and_file_paths.items():
                save_tensor_as_npy(tensor, file_path)
            self.real_overflow_nums += 1
            if self.overflow_nums != -1 and self.real_overflow_nums >= self.overflow_nums:
                logger.info(f"[{Const.TOOL_NAME}] Reached the preset overflow times, "
                            f"current overflow times: {self.real_overflow_nums}.")
        self.cached_tensors_and_file_paths = {}

    def _analyze_maybe_overflow_tensor(self, tensor_json):
        tensor_stat_index = tensor_json.get(Const.TENSOR_STAT_INDEX)
        if tensor_stat_index is None:
            logger.warning("tensor_stat_index does not exist in tensor_json.")
            return
        max_tensor = self.data_writer.get_buffer_values_max(tensor_stat_index)
        min_tensor = self.data_writer.get_buffer_values_min(tensor_stat_index)
        if max_tensor is None or min_tensor is None:
            return

        def check_inf_nan(value):
            # Use .item() if it's a tensor-like structure
            if hasattr(value, "item"):
                value = value.item()
            return np.isinf(value) or np.isnan(value)

        if check_inf_nan(max_tensor):
            self.has_overflow = True
            return

        if check_inf_nan(min_tensor):
            self.has_overflow = True

    def _analyze_tensor(self, tensor, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        if not path_len_exceeds_limit(file_path):
            self.cached_tensors_and_file_paths.update({file_path: tensor})
        else:
            logger.warning(f'The file path {file_path} length exceeds limit.')
        single_arg = super()._analyze_tensor(tensor, suffix)
        self._analyze_maybe_overflow_tensor(single_arg)
        single_arg.update({"data_name": dump_data_name})
        return single_arg


class KernelDumpDataProcessor(MindsporeDataProcessor):
    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.enable_kernel_dump = True

    @staticmethod
    def start_kernel_dump(config_path):
        hal.synchronize()
        _msprobe_c.init_dump()
        _msprobe_c.set_dump(config_path)
        hal.synchronize()

    @staticmethod
    def stop_kernel_dump():
        hal.synchronize()
        _msprobe_c.finalize_dump()
        hal.synchronize()

    @staticmethod
    def _print_unsupported_log(api_name):
        logger.warning(f"The kernel dump does not support the {api_name} API.")

    def analyze_forward_input(self, name, module, module_input_output):
        if not self.enable_kernel_dump:
            return
        if not has_adump:
            logger.warning("The current msprobe package does not compile adump, and kernel dump cannot be used.")
            self.enable_kernel_dump = False
            return
        self.start_kernel_dump(self.config.kernel_config_path)

    def analyze_forward_output(self, name, module, module_input_output):
        if not self.enable_kernel_dump:
            return
        self.enable_kernel_dump = False
        self.stop_kernel_dump()
        logger.info(f"The kernel data of {name} is dumped successfully.")

    def analyze_backward_input(self, name, module, module_input_output):
        if not self.enable_kernel_dump:
            return
        if not has_adump:
            logger.warning("The current msprobe package does not compile adump, and kernel dump cannot be used.")
            self.enable_kernel_dump = False
            return
        self.start_kernel_dump(self.config.kernel_config_path)

    def analyze_backward(self, name, module, module_input_output):
        if not self.enable_kernel_dump:
            return
        self.enable_kernel_dump = False
        self.stop_kernel_dump()
        logger.info(f"The kernel data of {name} is dumped successfully.")

    def reset_status(self):
        self.enable_kernel_dump = True
