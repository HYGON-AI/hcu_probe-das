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
import re

from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import check_file_or_directory_path, check_output_dir_path


def check_model_path_legality(path):
    check_file_or_directory_path(path, False, False, [".onnx", ".om"])
    return path


def check_input_data_path(path):
    if not isinstance(path, str):
        logger.error(f"input data path:{path} is illegal. Please check.")
        raise argparse.ArgumentTypeError
    if path == '':
        return path
    input_item_paths = path.split(',')
    for input_item_path in input_item_paths:
        check_file_or_directory_path(input_item_path, False, False, [".npy", ".bin"])
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
