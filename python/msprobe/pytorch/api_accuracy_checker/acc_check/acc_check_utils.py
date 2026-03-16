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

import os
from collections import namedtuple
import re
import numpy as np
import torch
try:
    import torch_npu
except ImportError:
    current_device = "cuda"
    from torch.cuda.amp import autocast
    IS_GPU = True
else:
    current_device = "npu"
    from torch_npu.npu.amp import autocast
    IS_GPU = False

from msprobe.core.common.const import FileCheckConst, Const, CompareConst
from msprobe.core.common.file_utils import FileChecker
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException
from msprobe.pytorch.dump.api_dump.api_register import ApiTemplate, get_api_register
from msprobe.pytorch.api_accuracy_checker.common.utils import is_dtype_fp8, is_hifloat8_tensor


hf_32_standard_api = ["conv1d", "conv2d"]
not_detach_set = {'resize_', 'resize_as_', 'set_', 'transpose_', 't_', 'squeeze_', 'unsqueeze_'}
not_raise_dtype_set = {'type_as'}

PRECISION_MAPPING = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float64
}


CpuParams = namedtuple("CpuArgs", ["cpu_args", "cpu_kwargs", "autocast_dtype", "is_autocast"])
ExecParams = namedtuple("ExecParams", ["api_type", "api_name", "device", "args", "kwargs", 
                                       "is_autocast", "autocast_dtype"])


class BackwardMessage:
    MULTIPLE_BACKWARD_MESSAGE = "Multiple backward is not supported."
    UNSUPPORT_BACKWARD_MESSAGE = "function with out=... arguments don't support automatic differentiation, " \
                                  "skip backward."
    NO_BACKWARD_RESULT_MESSAGE = "This API does not have backward input data, skip backward."
    UNSUPPORT_API_MESSAGE = "This API does not support backward ut, skip backward."


class UtDataInfo:
    def __init__(self, bench_grad, device_grad, device_output, bench_output, grad_in, in_fwd_data_list,
                 backward_message, rank=0, is_fp8=False):
        self.bench_grad = bench_grad
        self.device_grad = device_grad
        self.device_output = device_output
        self.bench_output = bench_output
        self.grad_in = grad_in
        self.in_fwd_data_list = in_fwd_data_list
        self.backward_message = backward_message
        self.rank = rank
        self.is_fp8 = is_fp8


def get_validated_result_csv_path(result_csv_path, mode):
    if mode not in ['result', 'detail']:
        raise ValueError("The csv mode must be result or detail")
    result_csv_path_checker = FileChecker(result_csv_path, FileCheckConst.FILE, ability=FileCheckConst.READ_WRITE_ABLE,
                                          file_type=FileCheckConst.CSV_SUFFIX)
    validated_result_csv_path = result_csv_path_checker.common_check()
    if mode == 'result':
        result_csv_name = os.path.basename(validated_result_csv_path)
        pattern = r"^accuracy_checking_result_\d{14}\.csv$"
        if not re.match(pattern, result_csv_name):
            raise ValueError("When continue acc_check, please do not modify the result csv name.")
    return validated_result_csv_path


def get_validated_details_csv_path(validated_result_csv_path):
    result_csv_name = os.path.basename(validated_result_csv_path)
    details_csv_name = result_csv_name.replace('result', 'details')
    details_csv_path = os.path.join(os.path.dirname(validated_result_csv_path), details_csv_name)
    details_csv_path_checker = FileChecker(details_csv_path, FileCheckConst.FILE,
                                           ability=FileCheckConst.READ_WRITE_ABLE, file_type=FileCheckConst.CSV_SUFFIX)
    validated_details_csv_path = details_csv_path_checker.common_check()
    return validated_details_csv_path


def exec_api(exec_params):
    api_type = exec_params.api_type
    api_name = exec_params.api_name
    device = exec_params.device
    args = exec_params.args
    kwargs = exec_params.kwargs
    is_autocast = exec_params.is_autocast
    autocast_dtype = exec_params.autocast_dtype
    out = None

    prefix_map = Const.API_DATA_PREFIX.get(Const.PT_FRAMEWORK, {})
    if not prefix_map or api_type not in prefix_map.values() or \
        api_type not in (
            Const.FUNCTIONAL_API_TYPE_PREFIX,
            Const.TENSOR_API_TYPE_PREFIX,
            Const.TORCH_API_TYPE_PREFIX,
            Const.ATEN_API_TYPE_PREFIX,
            Const.NPU_API_TYPE_PREFIX
    ):
        return out

    api_register = get_api_register()
    api_register.initialize_hook(None)
    api_func_type = list(prefix_map.keys())[list(prefix_map.values()).index(api_type)]
    api_func = api_register.ori_api_attr.get(Const.PT_FRAMEWORK + Const.SEP + api_func_type, {}).get(api_name)
    # ori_api_attr 存的是 list of ([module], callable)，需取出可调用对象
    api_func = api_func[0][1]
    torch_api = ApiTemplate(api_name, api_func, api_type, None, need_hook=False, device=device)
    if is_autocast:
        with autocast(dtype=autocast_dtype):
            out = torch_api.forward(*args, **kwargs)
    else:
        out = torch_api.forward(*args, **kwargs)
    return out


def deal_detach(arg, to_detach=True):
    return arg.detach() if to_detach else arg


def raise_bench_data_dtype(api_name, arg, raise_dtype=None):
    '''
    将标杆数据的dtype转换为raise_dtype
    输入：
        api_name：api名称
        arg：标杆输入
        raise_dtype：需要转换的dtype
    输出： 
        arg: 转换dtype的标杆输入
    '''
    if is_hifloat8_tensor(arg):
        return hif8_to_fp32(arg)
    if is_dtype_fp8(arg.dtype):
        return fp8_to_fp32(arg)
    if api_name in hf_32_standard_api and arg.dtype == torch.float32:
        return arg
    if raise_dtype is None or arg.dtype not in PRECISION_MAPPING or raise_dtype == arg.dtype:
        return arg
    return arg.type(raise_dtype)


def generate_device_params(input_args, input_kwargs, need_backward, api_name):
    is_fp8 = False
    
    def recursive_arg_to_device(arg_in, to_detach, depth=0):
        nonlocal is_fp8
        if depth > Const.MAX_DEPTH:
            logger.error("The depth of arg_in is too large, please check the arg_in.")
            raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_device(arg, to_detach, depth=depth + 1) for arg in arg_in)
        elif isinstance(arg_in, torch.Tensor):
            if is_dtype_fp8(arg_in.dtype) or is_hifloat8_tensor(arg_in):
                is_fp8 = True
            if need_backward and arg_in.requires_grad:
                arg_in = deal_detach(arg_in.clone(), to_detach).to(current_device).requires_grad_()
                temp_arg_in = arg_in * 1
                arg_in = temp_arg_in.type_as(arg_in)
                arg_in.retain_grad()
                return arg_in
            else:
                return deal_detach(arg_in.clone(), to_detach).to(current_device)
        else:
            return arg_in

    is_detach = api_name not in not_detach_set
    device_args = recursive_arg_to_device(input_args, is_detach)
    device_kwargs = \
        {key: recursive_arg_to_device(value, key != "out" and is_detach) for key, value in input_kwargs.items()}
    return device_args, device_kwargs, is_fp8


def generate_cpu_params(input_args, input_kwargs, need_backward, api_name):
    def recursive_arg_to_cpu(arg_in, to_detach, raise_dtype=None, depth=0):
        if depth > Const.MAX_DEPTH:
            logger.error("The depth of arg_in is too large, please check the arg_in.")
            raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_cpu(arg, to_detach, raise_dtype=raise_dtype, depth=depth + 1)
                                for arg in arg_in)
        elif isinstance(arg_in, torch.Tensor):
            if need_backward and arg_in.requires_grad:
                arg_in = deal_detach(raise_bench_data_dtype(
                                     api_name, arg_in.clone(), raise_dtype=raise_dtype), to_detach).requires_grad_()
                temp_arg_in = arg_in * 1
                arg_in = temp_arg_in.type_as(arg_in)
                arg_in.retain_grad()
                return arg_in
            else:
                return deal_detach(raise_bench_data_dtype(api_name, arg_in.clone(), raise_dtype=raise_dtype), to_detach)
        else:
            return arg_in

    def is_tensor_with_raise_precision(arg_in, check_kwargs=False):
        if arg_in.dtype in PRECISION_MAPPING:
            return True
        if check_kwargs and arg_in.dtype in [torch.half, torch.bfloat16]:
            return True
        return False

    def recursive_find_dtypes(arg_in, kwargs=None, check_kwargs=False, depth=0):
        if depth > Const.MAX_DEPTH:
            logger.error("The depth of arg_in is too large, please check the arg_in.")
            raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
        if isinstance(arg_in, (list, tuple)):
            return set().union(*tuple(recursive_find_dtypes(arg, kwargs, check_kwargs=check_kwargs, depth=depth + 1) for
                                      arg in arg_in))
        elif isinstance(arg_in, torch.Tensor) and is_tensor_with_raise_precision(arg_in, check_kwargs):
            return set([arg_in.dtype])
        elif isinstance(arg_in, dict) and check_kwargs:
            return set().union(*tuple(recursive_find_dtypes(v, kwargs, check_kwargs=True, depth=depth + 1) for
                                      v in arg_in.values()))
        return set()

    raise_dtype = None
    autocast_dtype = None
    is_autocast = False
    need_raise_dtypes = recursive_find_dtypes(input_args)
    need_raise_dtypes.update(recursive_find_dtypes(input_kwargs, check_kwargs=True))
    if len(need_raise_dtypes) == 1:
        origin_dtype = need_raise_dtypes.pop()
        raise_dtype = PRECISION_MAPPING.get(origin_dtype, torch.float32)
        autocast_dtype = origin_dtype
        
    elif len(need_raise_dtypes) >= 2:
        raise_dtype = torch.float32
        need_raise_dtypes.discard(torch.float32)
        autocast_dtype = need_raise_dtypes.pop()
        is_autocast = True

    raise_dtype = None if api_name in not_raise_dtype_set else raise_dtype
    is_detach = api_name not in not_detach_set
    cpu_args = recursive_arg_to_cpu(input_args, is_detach, raise_dtype=raise_dtype)
    cpu_kwargs = {key: recursive_arg_to_cpu(value, key != "out" and is_detach, raise_dtype=raise_dtype) for
                  key, value in input_kwargs.items()}
    cpu_params = CpuParams(cpu_args, cpu_kwargs, autocast_dtype, is_autocast)
    return cpu_params


def record_skip_info(api_full_name, compare, compare_alg_results):
    result_info = (api_full_name, CompareConst.SKIP, CompareConst.SKIP, [compare_alg_results], None, 0)
    compare.record_results(result_info)


def is_unsupported_api(api_name, is_overflow_check=False):
    split_name = api_name.split(Const.SEP)[0]
    unsupport_type_list = [Const.DISTRIBUTED, Const.MINDSPEED_API_TYPE_PREFIX]
    flag = (split_name in unsupport_type_list) or (is_overflow_check and split_name == Const.NPU)
    if flag:
        logger.info(f"{split_name} api is not supported for acc_check. SKIP.")
    return flag


def fp8_to_fp32(x):
    """
    将FP8格式的张量转换为FP32格式，保持原始FP8的表示范围
    使用纯PyTorch操作替代NumPy位运算
    
    参数:
    x: 输入的FP8张量，可以是torch.float8_e4m3fn或torch.float8_e5m2类型
    
    返回:
    torch.Tensor: 转换后的FP32张量，保持原始FP8的表示范围
    """
    if x.dtype == torch.float8_e4m3fn:
        # E4M3FN格式：1符号+4指数+3尾数，偏置7
        # 位布局：SEEEEEMM
        
        # 将FP8值视为无符号整数进行位操作
        x_int = x.view(torch.uint8)
        
        # 提取符号位、指数位和尾数位
        sign_bits = (x_int & 0x80) >> 7  # 最高位是符号位
        exp_bits = (x_int & 0x78) >> 3   # 接下来4位是指数位
        mantissa_bits = x_int & 0x07     # 最后3位是尾数位
        
        # 处理规格化数和非规格化数
        is_normal = exp_bits != 0
        
        # 计算FP32的指数部分（偏置127）
        fp32_exp = torch.where(
            is_normal,
            (exp_bits - 7 + 127).to(torch.int32),  # 规格化数：指数 = 原始指数 + 120
            torch.tensor(0, dtype=torch.int32, device=x.device)  # 非规格化数：指数为0
        )
        
        # 计算FP32的尾数部分
        # 规格化数：隐含1，尾数 = 1.0 + 原始尾数 * 2^(-3)
        # 非规格化数：无隐含1，尾数 = 0.0 + 原始尾数 * 2^(-3)
        fp32_mantissa = torch.where(
            is_normal,
            1.0 + mantissa_bits.to(torch.float32) / 8.0,  # 2^(-3) = 1/8
            mantissa_bits.to(torch.float32) / 8.0
        )
        
        # 计算符号值 (-1)^sign
        sign_value = torch.pow(-1.0, sign_bits.to(torch.float32))
        
        # 计算最终FP32值
        # 规格化数：value = (-1)^sign * (1.0 + mantissa/8) * 2^(exp - 7)
        # 非规格化数：value = (-1)^sign * (mantissa/8) * 2^(-6)
        fp32_result = sign_value * fp32_mantissa * torch.pow(2.0, fp32_exp - 127)
        
        return fp32_result
    
    elif x.dtype == torch.float8_e5m2:
        # E5M2格式：1符号+5指数+2尾数，偏置15
        # 位布局：SEEEEEEM
        
        # 将FP8值视为无符号整数进行位操作
        x_int = x.view(torch.uint8)
        
        # 提取符号位、指数位和尾数位
        sign_bits = (x_int & 0x80) >> 7  # 最高位是符号位
        exp_bits = (x_int & 0x7C) >> 2   # 接下来5位是指数位
        mantissa_bits = x_int & 0x03     # 最后2位是尾数位
        
        # 处理规格化数和非规格化数
        is_normal = exp_bits != 0
        
        # 计算FP32的指数部分（偏置127）
        fp32_exp = torch.where(
            is_normal,
            (exp_bits - 15 + 127).to(torch.int32),  # 规格化数：指数 = 原始指数 + 112
            torch.tensor(0, dtype=torch.int32, device=x.device)  # 非规格化数：指数为0
        )
        
        # 计算FP32的尾数部分
        # 规格化数：隐含1，尾数 = 1.0 + 原始尾数 * 2^(-2)
        # 非规格化数：无隐含1，尾数 = 0.0 + 原始尾数 * 2^(-2)
        fp32_mantissa = torch.where(
            is_normal,
            1.0 + mantissa_bits.to(torch.float32) / 4.0,  # 2^(-2) = 1/4
            mantissa_bits.to(torch.float32) / 4.0
        )
        
        # 计算符号值 (-1)^sign
        sign_value = torch.pow(-1.0, sign_bits.to(torch.float32))
        
        # 计算最终FP32值
        fp32_result = sign_value * fp32_mantissa * torch.pow(2.0, fp32_exp - 127)
        
        return fp32_result
    
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}. Expected torch.float8_e4m3fn or torch.float8_e5m2.")


def hif8_to_fp32(x):
    """
    将HiFloat8格式的张量转换为FP32格式，保持原始HiFloat8的表示范围
    使用纯PyTorch操作替代NumPy位运算    
    
    参数:
    x: 输入的HiFloat8张量，可以是torch_npu.HiFloat8Tensor类型
    
    返回:
    torch.Tensor: 转换后的FP32张量，保持原始HiFloat8的表示范围
    """
    requires_grad = x.requires_grad
    x = x.cpu().detach().numpy()
    x = np.array(x)  # 确保输入是numpy数组
    
    # 创建结果数组，保持与输入相同的形状
    res = np.zeros_like(x, dtype=np.float32)
    
    # 获取输入张量的所有维度
    dimensions = x.shape
    # 计算总元素数量
    total_elements = np.prod(dimensions)
    
    # 遍历每个元素
    for idx in range(total_elements):
        # 将一维索引转换为多维索引
        multi_indices = np.unravel_index(idx, dimensions)
        z = x[multi_indices]
        
        # 处理特殊值
        if np.isnan(z) or np.isinf(z):
            res[multi_indices] = z
            continue
            
        # 提取符号位
        s = 1.0 if z >= 0 else -1.0
        tmp = abs(z)
        
        # 处理零值
        if tmp == 0:
            res[multi_indices] = 0.0
            continue
            
        # 确定指数范围和尾数位数
        exponent = np.floor(np.log2(tmp + 1e-100))  # 添加小常量避免log2(0)
        eabs = abs(exponent)
        
        # 根据指数范围确定尾数位数和还原规则
        if eabs <= 3:       # 3-bit Mantissa
            mantissa = (tmp / (2.0 ** exponent)) * 8.0  # 还原尾数部分
            res[multi_indices] = s * (mantissa / 8.0) * (2.0 ** exponent)
        elif eabs <= 7:     # 2-bit Mantissa
            mantissa = (tmp / (2.0 ** exponent)) * 4.0
            res[multi_indices] = s * (mantissa / 4.0) * (2.0 ** exponent)
        elif eabs <= 15:    # 1-bit Mantissa
            mantissa = (tmp / (2.0 ** exponent)) * 2.0
            res[multi_indices] = s * (mantissa / 2.0) * (2.0 ** exponent)
        else:               # 0-bit Mantissa
            res[multi_indices] = s * (2.0 ** exponent)
    
    res = torch.from_numpy(res)
    if requires_grad:
        res = res.requires_grad_()
    return res
