# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

import argparse
import os
import re

from msprobe.infer.utils.file_open_check import FileStat, is_legal_args_path_string
from msprobe.infer.utils.file_utils import check_input_file_path, check_input_dir_path, check_output_dir_path
from msprobe.infer.utils.security_check import is_endswith_extensions

STR_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")
MAX_SIZE_LIMITE_NORMAL_MODEL = 32 * 1024 * 1024 * 1024  # 32GB
MAX_SIZE_LIMITE_FUSION_FILE = 1 * 1024 * 1024 * 1024  # 1GB


def check_model_path_legality(value):
    path_value = value
    if os.path.isdir(path_value):
        check_input_dir_path(path_value)
        if not is_saved_model_valid(path_value):
            raise argparse.ArgumentTypeError(f"model path:{path_value} is not qualified saved_model file. "
                                             f"Please check.")
        return path_value
    else:
        check_input_file_path(path_value, file_max_size=MAX_SIZE_LIMITE_NORMAL_MODEL)
        try:
            file_stat = FileStat(path_value)
        except Exception as err:
            raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.") from err
        if not file_stat.is_basically_legal('read'):
            raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
        if not file_stat.is_legal_file_type(["onnx", "om", "pb"]):
            raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
        if not file_stat.is_legal_file_size(MAX_SIZE_LIMITE_NORMAL_MODEL):
            raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
        return path_value


def is_saved_model_valid(directory):
    if not os.path.isdir(directory):
        return False

    saved_model_pb = os.path.join(directory, "saved_model.pb")
    if not os.path.isfile(saved_model_pb):
        return False

    variables_dir = os.path.join(directory, "variables")
    return os.path.isdir(variables_dir)


def check_om_path_legality(value):
    path_value = value
    if os.path.isdir(path_value):
        check_input_dir_path(path_value)
        if not is_saved_model_valid(path_value):
            raise argparse.ArgumentTypeError(f"model path:{path_value} is not qualified saved_model file. "
                                             f"Please check.")
        return path_value
    else:
        check_input_file_path(path_value, file_max_size=MAX_SIZE_LIMITE_NORMAL_MODEL)
        try:
            file_stat = FileStat(path_value)
        except Exception as err:
            raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.") from err
        if not file_stat.is_basically_legal('read'):
            raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.")
        if not file_stat.is_legal_file_type(["om"]):
            raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.")
        if not file_stat.is_legal_file_size(MAX_SIZE_LIMITE_NORMAL_MODEL):
            raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.")
        return path_value


def check_input_path_legality(value):
    if not value:
        return value
    inputs_list = value.split(',')
    for input_path in inputs_list:
        check_input_file_path(input_path)
        try:
            file_stat = FileStat(input_path)
        except Exception as err:
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.") from err
        if not file_stat.is_basically_legal('read'):
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.")
    return value


def check_debug_compare_input_data_path(path):
    if not isinstance(path, str):
        raise argparse.ArgumentTypeError(f"input data path:{path} is illegal. Please check.")
    if path == '':
        return path
    input_item_paths = path.split(',')
    for input_item_path in input_item_paths:
        input_item_path = check_input_path_legality(input_item_path)
        if not is_endswith_extensions(input_item_path, ['.npy', '.bin']):
            raise argparse.ArgumentTypeError(f"input data path:{path} is illegal. Please check.")
    return path


def check_cann_path_legality(value):
    path_value = value
    check_input_dir_path(path_value)
    if not is_legal_args_path_string(path_value):
        raise argparse.ArgumentTypeError(f"cann path:{path_value} is illegal. Please check.")
    return path_value


def check_output_path_legality(value):
    if not value:
        return value
    path_value = value
    check_output_dir_path(path_value)
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"output path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal("write", strict_permission=False):
        raise argparse.ArgumentTypeError(f"output path:{path_value} is illegal. Please check.")
    return path_value


def check_path_exit(value):
    if not os.path.exists(value):
        raise ValueError

    return value


def check_input_json_path(path):
    if not isinstance(path, str):
        raise argparse.ArgumentTypeError(f"ops json path:{path} is illegal. Please check.")
    check_input_file_path(path)
    if not is_endswith_extensions(path, ".json"):
        raise argparse.ArgumentTypeError(f"ops json path:{path} is illegal. Please check.")
    return path


def check_alone_compare_dir_path(path):
    check_input_dir_path(path)
    return path


def valid_json_file_or_dir(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"input path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal('read', strict_permission=False):
        raise argparse.ArgumentTypeError(f"input path:{path_value} is illegal. Please check.")

    # input type: dir or json
    # input type -> json need additional check
    if not file_stat.is_dir:
        if not file_stat.is_legal_file_type(["json"]):
            raise argparse.ArgumentTypeError(f"input path:{path_value} is illegal. Please check.")

        if not file_stat.is_legal_file_size(MAX_SIZE_LIMITE_NORMAL_MODEL):
            raise argparse.ArgumentTypeError(f"input path:{path_value} is illegal. Please check.")
    return path_value


def check_dict_kind_string(value):
    # just like "input_name1:1,224,224,3;input_name2:3,300"
    if not value:
        return value
    input_shape = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.-]")
    if regex.search(input_shape):
        raise argparse.ArgumentTypeError(f"dym string \"{input_shape}\" is not a legal string")
    return input_shape


def check_device_range_valid(value):
    min_value = 0
    max_value = 255
    try:
        ivalue = int(value)
    except Exception as e:
        raise argparse.ArgumentTypeError("The input device is illegal. \
                                         The value must be of type that can be cast to an int variable!") from e
    if ivalue < min_value or ivalue > max_value:
        raise argparse.ArgumentTypeError(
            "device:{} is invalid. valid value range is [{}, {}]".format(ivalue, min_value, max_value)
        )
    return value


def check_number_list(value):
    # just like "1241414,124141,124424"
    if not value:
        return value
    outsize_list = value.split(',')
    for outsize in outsize_list:
        regex = re.compile(r"[^0-9]")
        if regex.search(outsize):
            raise argparse.ArgumentTypeError(f"output size \"{outsize}\" is not a legal string")
    return value


def check_dym_range_string(value):
    if not value:
        return value
    dym_string = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.\-~]")
    if regex.search(dym_string):
        raise argparse.ArgumentTypeError(f"dym range string \"{dym_string}\" is not a legal string")
    return dym_string


def check_fusion_cfg_path_legality(value):
    if not value:
        return value
    path_value = value
    check_input_file_path(path_value)
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"fusion switch file path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal('read'):
        raise argparse.ArgumentTypeError(f"fusion switch file path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_type(["cfg"]):
        raise argparse.ArgumentTypeError(f"fusion switch file path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_size(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError(f"fusion switch file path:{path_value} is illegal. Please check.")
    return path_value


def check_quant_json_path_legality(value):
    if not value:
        return value
    path_value = value
    check_input_file_path(path_value)
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"quant file path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal('read'):
        raise argparse.ArgumentTypeError(f"quant file path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_type(["json"]):
        raise argparse.ArgumentTypeError(f"quant file path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_size(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError(f"quant file path:{path_value} is illegal. Please check.")
    return path_value


def safe_string(value):
    if not value:
        return value
    if re.search(STR_WHITE_LIST_REGEX, value):
        raise ValueError("String parameter contains invalid characters.")
    return value


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected true, 1, false, 0 with case insensitive.')