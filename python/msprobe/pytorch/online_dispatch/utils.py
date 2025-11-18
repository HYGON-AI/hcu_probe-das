# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

import inspect

import numpy as np
import psutil
import torch

try:
    import torch_npu
except ImportError:
    pta_cpu_device = None
else:
    pta_cpu_device = torch.device("cpu")

from msprobe.core.common.const import CompareConst
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.pytorch.common.log import logger


cpu_device = torch._C.device("cpu")
COLOR_RED = '\033[31m'
COLOR_GREEN = '\033[32m'
COLOR_YELLOW = '\033[33m'
COLOR_BLUE = '\033[34m'
COLOR_PURPLE = '\033[35m'
COLOR_CYAN = '\033[36m'
COLOR_GRAY = '\033[37m'
COLOR_RESET = '\033[0m'

COMPARE_LOGO = '''
            _ _                                                       
  ___  _ __ | (_)_ __   ___    ___ ___  _ __ ___  _ __   __ _ _ __ ___ 
 / _ \\| '_ \\| | | '_ \\ / _ \\  / __/ _ \\| '_ ` _ \\| '_ \\ / _` | '__/ _ \\ 
| (_) | | | | | | | | |  __/ | (_| (_) | | | | | | |_) | (_| | | |  __/
 \\___/|_| |_|_|_|_| |_|\\___|  \\___\\___/|_| |_| |_| .__/ \\__,_|_|  \\___|
                                                 |_|    
'''

CSV_COLUMN_NAME = [
    CompareConst.NPU_NAME,
    CompareConst.BENCH_NAME,
    CompareConst.NPU_DTYPE,
    CompareConst.BENCH_DTYPE,
    CompareConst.NPU_SHAPE,
    CompareConst.BENCH_SHAPE,
    CompareConst.NPU_MAX,
    CompareConst.NPU_MIN,
    CompareConst.NPU_MEAN,
    CompareConst.BENCH_MAX,
    CompareConst.BENCH_MIN,
    CompareConst.BENCH_MEAN,
    CompareConst.COSINE,
    CompareConst.MAX_ABS_ERR,
    CompareConst.MAX_RELATIVE_ERR,
    CompareConst.ACCURACY,
    CompareConst.STACK,
    CompareConst.ERROR_MESSAGE
]

FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble, np.float32, np.float16]
BOOL_TYPE = [bool, np.uint8]
INT_TYPE = [np.int32, np.int64]


def get_callstack():
    callstack = []
    for (_, path, line, func, code, _) in inspect.stack()[2:]:
        try:
            stack_line = [path, str(line), func, code[0].strip() if code else code]
            callstack.append(stack_line)
        except IndexError:
            logger.error("Failed to get callstack for code:{} index out of range".format(code))
    return callstack


@recursion_depth_decorator("data_to_cpu")
def data_to_cpu(data, deep, data_cpu):
    global cpu_device
    list_cpu = []
    if isinstance(data, torch.Tensor):
        if data.device == cpu_device or data.device == pta_cpu_device:
            tensor_copy = data.clone().detach()
        else:
            tensor_copy = data.cpu().detach()
        if tensor_copy.dtype in [torch.float16, torch.half, torch.bfloat16]:
            tensor_copy = tensor_copy.float()

        if deep == 0:
            data_cpu.append(tensor_copy)
        return tensor_copy
    elif isinstance(data, list):
        for v in data:
            list_cpu.append(data_to_cpu(v, deep + 1, data_cpu))
        if deep == 0:
            data_cpu.append(list_cpu)
        return list_cpu
    elif isinstance(data, tuple):
        for v in data:
            list_cpu.append(data_to_cpu(v, deep + 1, data_cpu))
        tuple_cpu = tuple(list_cpu)
        if deep == 0:
            data_cpu.append(tuple_cpu)
        return tuple_cpu
    elif isinstance(data, dict):
        dict_cpu = {}
        for k, v in data.items():
            dict_cpu[k] = data_to_cpu(v, deep + 1, data_cpu)
        if deep == 0:
            data_cpu.append(dict_cpu)
        return dict_cpu
    elif isinstance(data, torch._C.device):
        return cpu_device
    else:
        if deep == 0:
            data_cpu.append(data)
        return data


def get_sys_info():
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    sys_info = f'Total: {mem.total / 1024 / 1024:.2f}MB ' \
               f'Free: {mem.available / 1024 / 1024:.2f} MB ' \
               f'Used: {mem.used / 1024 / 1024:.2f} MB ' \
               f'CPU: {cpu_percent}% '
    return sys_info


class DispatchException(Exception):
    INVALID_PARAMETER = 0

    def __init__(self, err_code, err_msg=""):
        super(DispatchException, self).__init__()
        self.err_code = err_code
        self.err_msg = err_msg

    def __str__(self):
        return self.err_msg


def check_idx_valid(data, idx):
    if data is not None and data.numel() > 0 and 0 <= idx < data.numel():
        return True
    return False
