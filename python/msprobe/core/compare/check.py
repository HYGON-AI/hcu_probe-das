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

import os

from msprobe.core.common.log import logger
from msprobe.core.common.utils import check_op_str_pattern_valid, CompareException
from msprobe.core.common.const import Const

cross_dtype_mapping = {
    "Int8": "int",
    "torch.int8": "int",
    "UInt8": "int",
    "torch.uint8": "int",
    "Int16": "int",
    "torch.int16": "int",
    "UInt16": "int",
    "torch.uint16": "int",
    "Int32": "int",
    "torch.int32": "int",
    "UInt32": "int",
    "torch.uint32": "int",
    "Int64": "int",
    "torch.int64": "int",
    "UInt64": "int",
    "torch.uint64": "int",

    "Float16": "float",
    "torch.float16": "float",
    "Float32": "float",
    "torch.float32": "float",
    "Float64": "float",
    "torch.float64": "float",
    "BFloat16": "float",
    "torch.bfloat16": "float",

    "Bool": "bool",
    "torch.bool": "bool",

    "Complex64": "complex",
    "torch.complex64": "complex",
    "Complex128": "complex",
    "torch.complex128": "complex",
}


def check_dump_json_str(op_data, op_name):
    input_list = op_data.get(Const.INPUT_ARGS, None) if op_data.get(Const.INPUT_ARGS, None) else op_data.get(
        Const.INPUT, None)
    input_kwargs = op_data.get(Const.INPUT_KWARGS, None)
    output_list = op_data.get(Const.OUTPUT, None)

    args = [input_list, input_kwargs, output_list]
    for arg in args:
        if not arg:
            continue
        if isinstance(arg, dict):
            check_json_key_value(arg, op_name)
        else:
            for ele in arg:
                if not ele:
                    continue
                check_json_key_value(ele, op_name)


def check_json_key_value(input_output, op_name, depth=0):
    if depth > Const.MAX_DEPTH:
        logger.error(f"string check of data info of {op_name} exceeds the recursion limit.")
        return
    if isinstance(input_output, list):
        for item in input_output:
            check_json_key_value(item, op_name, depth + 1)
    elif isinstance(input_output, dict):
        for key, value in input_output.items():
            if isinstance(value, dict):
                check_json_key_value(value, op_name, depth + 1)
            else:
                valid_key_value(key, value, op_name)


def valid_key_value(key, value, op_name):
    if key == "shape" and not isinstance(value, (list, tuple)):
        logger.error(f"shape of input or output of {op_name} is not list or tuple, please check!")
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)
    elif key == "requires_grad" and not isinstance(value, bool):
        logger.error(f"requires_grad of input or output of {op_name} is not bool, please check!")
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)
    else:
        check_op_str_pattern_valid(value, op_name)


def check_stack_json_str(stack_info, op_name):
    if isinstance(stack_info, list):
        for item in stack_info:
            check_op_str_pattern_valid(item, op_name, stack=True)
    else:
        logger.error(f"Expected stack_info to be a list, but got {type(stack_info).__name__} for '{op_name}'")
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)


def check_configuration_param(config):
    arg_list = [config.stack_mode, config.auto_analyze, config.fuzzy_match,
                config.highlight, config.first_diff_analyze, config.is_print_compare_log]
    arg_names = ['stack_mode', 'auto_analyze', 'fuzzy_match',
                 'highlight', 'first_diff_analyze', 'is_print_compare_log']
    for arg, name in zip(arg_list, arg_names):
        if not isinstance(arg, bool):
            logger.error(f"Invalid input parameter, {name} which should be only bool type.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
