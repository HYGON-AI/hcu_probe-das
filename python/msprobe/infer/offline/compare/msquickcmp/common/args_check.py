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

from msprobe.core.common.log import logger
from msprobe.infer.utils.file_open_check import FileStat, is_legal_args_path_string
from msprobe.infer.utils.file_utils import check_input_file_path, check_input_dir_path, check_output_dir_path, \
    check_path_no_group_others_write
from msprobe.infer.utils.security_check import is_endswith_extensions

STR_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")
MAX_SIZE_LIMITE_NORMAL_MODEL = 32 * 1024 * 1024 * 1024  # 32GB
MAX_SIZE_LIMITE_FUSION_FILE = 1 * 1024 * 1024 * 1024  # 1GB


def check_target_model_path_legality(path):
    if not isinstance(path, str) or not path:
        logger.error(f"target model path:{path} is illegal. Please check.")
        raise argparse.ArgumentTypeError
    if os.path.isdir(path):
        check_input_dir_path(path)
        check_path_no_group_others_write(path)
        return path
    else:
        check_input_file_path(path, file_max_size=MAX_SIZE_LIMITE_NORMAL_MODEL)
        if not is_endswith_extensions(path, ['om']):
            logger.error(f"target model path:{path} is illegal. Please check.")
            raise argparse.ArgumentTypeError
        check_path_no_group_others_write(path)
        return path


def check_model_path_legality(path):
    if not isinstance(path, str) or not path:
        logger.error(f"model path:{path} is illegal. Please check.")
        raise argparse.ArgumentTypeError
    if os.path.isdir(path):
        check_input_dir_path(path)
        check_path_no_group_others_write(path)
        return path
    else:
        check_input_file_path(path, file_max_size=MAX_SIZE_LIMITE_NORMAL_MODEL)
        if not is_endswith_extensions(path, ["onnx", "om"]):
            logger.error(f"model path:{path} is illegal. Please check.")
            raise argparse.ArgumentTypeError
        check_path_no_group_others_write(path)
        return path


def check_input_path_legality(value):
    if not value:
        return value
    inputs_list = value.split(',')
    for input_path in inputs_list:
        try:
            file_stat = FileStat(input_path)
        except Exception as err:
            logger.error(f"input path:{input_path} is illegal. Please check.")
            raise argparse.ArgumentTypeError from err
        if not file_stat.is_basically_legal('read'):
            logger.error(f"input path:{input_path} is illegal. Please check.")
            raise argparse.ArgumentTypeError
        check_path_no_group_others_write(input_path)
    return value


def check_input_data_path_legality(value):
    if not value:
        return value
    inputs_list = value.split(',')
    for input_path in inputs_list:
        check_input_file_path(input_path)
        try:
            file_stat = FileStat(input_path)
        except Exception as err:
            logger.error(f"input path:{input_path} is illegal. Please check.")
            raise argparse.ArgumentTypeError from err
        if not file_stat.is_basically_legal('read'):
            logger.error(f"input path:{input_path} is illegal. Please check.")
            raise argparse.ArgumentTypeError
        check_path_no_group_others_write(input_path)
    return value


def check_input_data_path(path):
    if not isinstance(path, str):
        logger.error(f"input data path:{path} is illegal. Please check.")
        raise argparse.ArgumentTypeError
    if path == '':
        return path
    input_item_paths = path.split(',')
    for input_item_path in input_item_paths:
        input_item_path = check_input_data_path_legality(input_item_path)
        if not is_endswith_extensions(input_item_path, ['.npy', '.bin']):
            logger.error(f"input data path:{path} is illegal. Please check.")
            raise argparse.ArgumentTypeError
        check_path_no_group_others_write(input_item_path)
    return path


def check_output_path_legality(path):
    if not path:
        return path
    check_output_dir_path(path)
    return path


def check_dict_kind_string(value):
    # just like "input_name1:1,224,224,3;input_name2:3,300"
    if not value:
        return value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.-]")
    if regex.search(value):
        logger.error(f"string '{value}' is not a legal string")
        raise argparse.ArgumentTypeError
    return value


def check_rank_range_valid(rank):
    min_value = 0
    max_value = 255
    try:
        int_rank = int(rank)
    except Exception as e:
        logger.error("The rank is illegal. The value must be of type that can be cast to an int variable!")
        raise argparse.ArgumentTypeError from e
    if int_rank < min_value or int_rank > max_value:
        logger.error(f"rank:{rank} is invalid. valid value range is [{min_value}, {max_value}]")
        raise argparse.ArgumentTypeError
    return rank


def check_number_list(value):
    # just like "1241414,124141,124424"
    if not value:
        return value
    outsize_list = value.split(',')
    for outsize in outsize_list:
        regex = re.compile(r"[^0-9]")
        if regex.search(outsize):
            logger.error(f"output size \"{outsize}\" is not a legal string")
            raise argparse.ArgumentTypeError
    return value


def check_dym_range_string(value):
    if not value:
        return value
    dym_string = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.\-~]")
    if regex.search(dym_string):
        logger.error(f"dym range string \"{dym_string}\" is not a legal string")
        raise argparse.ArgumentTypeError
    return value


def check_quant_json_path_legality(path):
    if not path:
        return path
    check_input_file_path(path)
    if not is_endswith_extensions(path, ['.json']):
        logger.error(f"quant fusion rule file path:{path} is illegal. Please check.")
        raise argparse.ArgumentTypeError
    return path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        logger.error('Boolean value expected true, 1, false, 0 with case insensitive.')
        raise argparse.ArgumentTypeError
