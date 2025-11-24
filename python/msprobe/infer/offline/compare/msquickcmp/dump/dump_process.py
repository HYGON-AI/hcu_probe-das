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

"""
Function:
This class mainly dump model ops inputs and outputs.
"""

import os
import time

from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import check_file_or_directory_path, create_directory
from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.convert import convert_npy_to_bin
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException, get_shape_to_directory_name
from msprobe.infer.offline.compare.msquickcmp.dump.args_adapter import DumpArgsAdapter


def _generate_golden_data_model(args: DumpArgsAdapter, npu_dump_npy_path):
    model_name, extension = utils.get_model_name_and_extension(args.golden_path)
    if extension == ".onnx":
        from msprobe.infer.offline.compare.msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData
        return OnnxDumpData(args, npu_dump_npy_path)
    else:
        logger.error("dump model files whose names end with .onnx are supported, Please check your model type.")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)


def dump_process(args: DumpArgsAdapter):
    """
    Function Description:
        main process function
    Exception Description:
        exit the program when an AccuracyCompare Exception  occurs
    """
    args.golden_path = os.path.realpath(args.golden_path)
    args.cann_path = os.path.realpath(args.cann_path)
    args.input_data = convert_npy_to_bin(args.input_data)
    check_and_dump(args)


def dump_data(args, input_shape, original_out_path):
    if input_shape:
        args.input_shape = input_shape
        args.output_path = os.path.join(original_out_path, get_shape_to_directory_name(args.input_shape))
    cpu_dump_process(args)


def cpu_dump_process(args):
    _, extension = utils.get_model_name_and_extension(args.golden_path)
    golden_dumper = _generate_golden_data_model(args, npu_dump_npy_path=None)
    golden_dumper.generate_inputs_data(npu_dump_data_path=None, use_aipp=False)
    golden_dumper.generate_dump_data()


def check_and_dump(args):
    check_file_or_directory_path(args.golden_path, False)
    utils.check_device_param_valid(args.rank)
    time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.output_path, time_dir))
    args.output_path = original_out_path
    create_directory(args.output_path)
    # deal with the dymShape_range param if exists
    input_shapes = []
    if args.dym_shape_range:
        input_shapes = utils.parse_dym_shape_range(args.dym_shape_range)
    if not input_shapes:
        input_shapes.append("")
    for input_shape in input_shapes:
        dump_data(args, input_shape, original_out_path)
