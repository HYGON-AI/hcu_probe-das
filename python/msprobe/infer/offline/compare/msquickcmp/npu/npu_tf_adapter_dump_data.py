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
This class mainly involves generate tf adapter npu dump data function.
"""

import glob
import os
import shutil

import numpy as np

from msprobe.core.common.log import logger
from msprobe.infer.offline.compare.msquickcmp.atc import atc_utils
from msprobe.infer.offline.compare.msquickcmp.common import utils, tf_common
from msprobe.infer.utils.util import load_file_to_read_common_check
from msprobe.infer.offline.compare.msquickcmp.common.tf_common import load_file_to_read_common_check_with_walk
from msprobe.infer.utils.check.rule import Rule

try:
    import tensorflow as tf
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
except ImportError as e:
    logger.error("TensorFlow is not installed.")
    raise ImportError from e

try:
    import npu_device
except ImportError as e:
    logger.error("npu_device is not installed.")
    raise ImportError from e

try:
    import acl
except ImportError as e:
    logger.error("Please verify that the CANN environment is properly configured.")
    raise ImportError from e


class NpuTfAdapterDumpData(object):
    """
    This class is used to generate NUP dump data of the tf2.6 save_model.
    """

    def __init__(self, arguments, model_path):
        self.serving = arguments.saved_model_signature
        self.tag_set = tf_common.split_tag_set(arguments.saved_model_tag_set)
        self.output_path = os.path.realpath(arguments.out_path)
        self.input = os.path.join(self.output_path, "input")
        self.dump_data_npu = os.path.join(self.output_path, "dump_data", "npu")
        self.model_json_path = os.path.join(self.output_path, "model")
        self.inputs_data = {}
        self.model_path = model_path
        self.input_path = arguments.input_path
        self.input_shape = self.split_input_shape(arguments.input_shape)
        self.inputs_dtype = tf_common.get_model_inputs_dtype(model_path, self.serving, self.tag_set)
        self.cann_path = arguments.cann_path
        self.fusion_switch_file = arguments.fusion_switch_file
        self._create_dir()

    @staticmethod
    def split_input_shape(input_shapes):
        input_list = input_shapes.split(";")
        input_shape_dict = {}
        for shape in input_list:
            input_name, shape_str = shape.split(":")
            shape_dims = list(map(int, shape_str.split(",")))
            input_shape_dict[input_name] = shape_dims

        return input_shape_dict

    @staticmethod
    def get_graph_txt(model_json_path):
        txt_files = glob.glob(os.path.join(model_json_path, '**/*.txt'), recursive=True)
        if len(txt_files) > 1:
            sorted_files = sorted(txt_files, key=lambda x: int(x.split('_')[-2]))
            return sorted_files[-1]

        return txt_files[0]

    def generate_inputs_data(self, use_aipp=False):
        # copy input_data into destination path
        if self.input_path:
            input_path = self.input_path.split(",")
            for input_file in input_path:
                if not os.path.isfile(input_file):
                    logger.error(f"no such file exists: {input_file}")
                    raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
                file_name = os.path.basename(input_file)
                dest_file = os.path.join(self.input, file_name)
                shutil.copy(input_file, dest_file)
                os.chmod(dest_file, 0o640)
                # convert .bin to numpy
                data_type = file_name.rsplit("_", 1)[-1].split(".")[0]
                input_name = file_name.rsplit("_", 1)[0]
                if Rule.input_file().check(input_file, will_raise=True):
                    input_data = np.fromfile(input_file, dtype=data_type).reshape(self.input_shape.get(input_name))
                self.inputs_data[input_name] = input_data
        else:
            for input_name, shape_str in self.input_shape.items():
                data_type = self.inputs_dtype.get(input_name)
                input_data = np.random.random(size=shape_str).astype(data_type)
                self.inputs_data[input_name] = input_data
                input_data.tofile(os.path.join(self.input, input_name + "_" + data_type.__name__ + ".bin"))

    def generate_dump_data(self, use_cli=False):
        # adapt NPU
        npu_device.compat.enable_v1()
        # switch ge graph dump
        os.environ['DUMP_GE_GRAPH'] = '2'
        os.environ['DUMP_GRAPH_LEVEL'] = '3'
        os.environ['DUMP_GRAPH_PATH'] = self.model_json_path
        config_proto = tf.compat.v1.ConfigProto()
        custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(self.dump_data_npu)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0")
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
        if self.fusion_switch_file:
            custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes(self.fusion_switch_file)
        config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        # sess run predict
        with tf.compat.v1.Session(config=config_proto) as sess:
            tag_set = {tf.compat.v1.saved_model.tag_constants.SERVING} if self.tag_set == "" else self.tag_set
            if os.path.isdir(self.model_path):
                load_file_to_read_common_check_with_walk(self.model_path)
            else:
                load_file_to_read_common_check(self.model_path)
            model = tf.compat.v1.saved_model.load(sess, tag_set, self.model_path)
            feed_dict = {
                sess.graph.get_tensor_by_name(input_name + ":0"): input_data
                for input_name, input_data in self.inputs_data.items()
            }
            output_tensors = []
            outputs_tensors_info = model.signature_def[self.serving].outputs
            output_op_names = []
            for output_tensor_info in outputs_tensors_info.values():
                output_op_names.append(output_tensor_info.name.split(':')[0])
            output_tensors.extend(sess.graph.get_operation_by_name(output_op_names[-1]).outputs)
            sess.run(output_tensors, feed_dict=feed_dict)
        logger.info(f"Dump tf adapter data success, data saved in: {self.dump_data_npu}")
        self.dump_data_npu = self._change_dump_data_path()
        graph_txt = self.get_graph_txt(self.model_json_path)
        output_json_path = atc_utils.convert_model_to_json(self.cann_path, graph_txt, self.output_path)

        return self.dump_data_npu, output_json_path

    def _change_dump_data_path(self):
        sub_dirs_with_files = []
        for sub_dir, _, files in os.walk(self.dump_data_npu):
            if files:
                sub_dirs_with_files.append(sub_dir)
        sorted_paths = sorted(sub_dirs_with_files, key=lambda x: int(x.split('/')[-3].split('_')[-1]))

        return sorted_paths[-1]

    def _create_dir(self):
        utils.create_directory(self.input)
        utils.create_directory(self.dump_data_npu)
        utils.create_directory(self.model_json_path)
