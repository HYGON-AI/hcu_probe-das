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

from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.args_check import is_saved_model_valid
from msprobe.infer.offline.compare.msquickcmp.common.convert import convert_npy_to_bin
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException, get_shape_to_directory_name
from msprobe.infer.offline.compare.msquickcmp.dump.args_adapter import DumpArgsAdapter


def _generate_golden_data_model(args: DumpArgsAdapter, npu_dump_npy_path):
    if is_saved_model_valid(args.model_path):
        from msprobe.infer.offline.compare.msquickcmp.tf.tf_save_model_dump_data import TfSaveModelDumpData
        return TfSaveModelDumpData(args, args.model_path)
    model_name, extension = utils.get_model_name_and_extension(args.model_path)
    if extension == ".pb":
        from msprobe.infer.offline.compare.msquickcmp.tf.tf_dump_data import TfDumpData
        return TfDumpData(args)
    elif extension == ".onnx":
        from msprobe.infer.offline.compare.msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData
        return OnnxDumpData(args, npu_dump_npy_path)
    else:
        utils.logger.error("cpu dump model files whose names end with .pb or .onnx or saved_model are "
                           "supported, Please check your model type")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)


def _generate_model_adapter(args: DumpArgsAdapter):
    if is_saved_model_valid(args.model_path):
        from msprobe.infer.offline.compare.msquickcmp.npu.npu_tf_adapter_dump_data import NpuTfAdapterDumpData
        return NpuTfAdapterDumpData(args, args.model_path)
    else:
        utils.logger.error("Currently, npu dump supports only saved_model, Please check your model type")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)


def dump_process(args: DumpArgsAdapter, use_cli: bool):
    """
    Function Description:
        main process function
    Exception Description:
        exit the program when an AccuracyCompare Exception  occurs
    """
    args.model_path = os.path.realpath(args.model_path)
    args.cann_path = os.path.realpath(args.cann_path)
    args.input_path = convert_npy_to_bin(args.input_path)
    args.fusion_switch_file = os.path.realpath(args.fusion_switch_file) if args.fusion_switch_file else None
    check_and_dump(args, use_cli)


def dump_data(args, input_shape, original_out_path, use_cli: bool):
    if input_shape:
        args.input_shape = input_shape
        args.out_path = os.path.join(original_out_path, get_shape_to_directory_name(args.input_shape))
    if args.device_pattern == "npu":
        """
        npu dump
        """
        npu_dump_process(args, use_cli)
    elif args.device_pattern == "cpu":
        """
        cpu dump
        """
        cpu_dump_process(args)
    else:
        raise ValueError("-dp only contain cpu and npu, please ensure that -dp id correct.")


def npu_dump_process(args, use_cli):
    # 1. get dumper
    npu_dumper = _generate_model_adapter(args)
    # 2. generate input
    npu_dumper.generate_inputs_data(use_aipp=False)
    # 3. dump data
    npu_dumper.generate_dump_data(use_cli=use_cli)


def cpu_dump_process(args):
    if is_saved_model_valid(args.model_path):
        # 1. get dumper
        golden_dumper = _generate_golden_data_model(args, npu_dump_npy_path="")
        # 2. generate input
        golden_dumper.generate_inputs_data_for_dump()
        # 3. dump data
        if args.tf_json_path is None:
            raise ValueError("when dump saved_model in cpu, please ensure that --tf-json is provided.")
        golden_dumper.generate_dump_data(args.tf_json_path)
    else:
        _, extension = utils.get_model_name_and_extension(args.model_path)
        golden_dumper = _generate_golden_data_model(args, npu_dump_npy_path=None)
        golden_dumper.generate_inputs_data(npu_dump_data_path=None, use_aipp=False)
        golden_dumper.generate_dump_data()


def check_and_dump(args, use_cli: bool):
    utils.check_file_or_directory_path(args.model_path, is_saved_model_valid(args.model_path))
    if args.fusion_switch_file:
        utils.check_file_or_directory_path(args.fusion_switch_file)
    utils.check_device_param_valid(args.device)
    utils.check_file_or_directory_path(os.path.realpath(args.out_path), True)
    time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.out_path, time_dir))
    args.out_path = original_out_path
    # deal with the dymShape_range param if exists
    input_shapes = []
    if args.dym_shape_range:
        input_shapes = utils.parse_dym_shape_range(args.dym_shape_range)
    if not input_shapes:
        input_shapes.append("")
    for input_shape in input_shapes:
        dump_data(args, input_shape, original_out_path, use_cli)
