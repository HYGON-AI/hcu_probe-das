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
This class is used to generate GPU dump data of the tf2.6 save_model.
"""

import json
import os
import time

import numpy as np

from msprobe.core.common.log import logger
from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.dump_data import DumpData
from msprobe.infer.utils.util import load_file_to_read_common_check
from msprobe.infer.utils.file_open_check import ms_open
from msprobe.infer.utils.constants import TENSOR_MAX_SIZE
from msprobe.infer.utils.check.rule import Rule


def parse_ops_name_from_om_json(tf_json_path):
    op_names = []
    om = utils.parse_json_file(tf_json_path)
    graph_list = om.get('graph')
    for graph in graph_list:
        ops = graph.get('op', [])
        output_desc_list = [op.get('output_desc', []) for op in ops]
        attrs_list = [od.get('attr', []) for od in sum(output_desc_list, [])]
        for attr in sum(attrs_list, []):
            if attr.get('key') == "_datadump_origin_name":
                op_names.append(attr.get('value').get('s'))

    return op_names


class TfSaveModelDumpData(DumpData):
    """
    This class is used to generate GUP dump data of the tf2.6 save_model.
    """

    def __init__(self, arguments, model_path):
        from msprobe.infer.offline.compare.msquickcmp.common import tf_common

        super().__init__()
        self._check_tf_version("2.6.5")
        output_path = os.path.realpath(arguments.out_path)
        self.serving = arguments.saved_model_signature
        self.tag_set = tf_common.split_tag_set(arguments.saved_model_tag_set)
        self.input = os.path.join(output_path, "input")
        self.dump_data_tf = os.path.join(output_path, "dump_data", "tf")
        self.input_path = arguments.input_path
        self.inputs_data = {}
        self.model_path = model_path
        self.input_shape_list = self.split_input_shape(arguments.input_shape)
        self.inputs_dtype = tf_common.get_model_inputs_dtype(model_path, self.serving, self.tag_set)
        self.net_output = {}
        self._create_dir()

    @staticmethod
    def parse_json_file(output_json_path):
        try:
            output_json_path = load_file_to_read_common_check(output_json_path)
            with ms_open(output_json_path, 'r', max_size=TENSOR_MAX_SIZE, encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File '{output_json_path}' not found, Please check whether the json file path is "
                                    f"valid. {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"File '{output_json_path}' is not a valid JSON format. {e}") from e

    @staticmethod
    def split_input_shape(input_shapes):
        input_list = input_shapes.split(";")
        input_shape_list = [
            (name, [int(num) for num in shape_data_str_list])
            for name, shape_data_str in (shape.split(":") for shape in input_list)
            for shape_data_str_list in [shape_data_str.split(",")]
        ]
        return input_shape_list

    @staticmethod
    def _is_op_exists(operation_name_to_check, operations):
        return any(op.name == operation_name_to_check for op in operations)

    @staticmethod
    def _check_tf_version(expected_version):
        import tensorflow as tf

        current_version = tf.__version__
        if current_version != expected_version:
            raise ImportError(
                f"TensorFlow version mismatch: expected version {expected_version}, "
                f"but found version {current_version}. Please install the correct "
                "version of TensorFlow."
            )

    def generate_inputs_data_for_dump(self):
        if self.input_path:
            input_path = self.input_path.split(",")
            for i, input_file in enumerate(input_path):
                if not os.path.isfile(input_file):
                    logger.error(f"no such file exists: {input_file}")
                    raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
                input_name = self.input_shape_list[i][0]
                input_shape = self.input_shape_list[i][1]
                data_type = self.inputs_dtype.get(input_name)
                if Rule.input_file().check(input_file, will_raise=True):
                    input_data = np.fromfile(input_file, dtype=data_type).reshape(input_shape)
                self.inputs_data[input_name] = input_data
        else:
            # generate random input data
            for input_shape_t in self.input_shape_list:
                input_name = input_shape_t[0]
                input_shape = input_shape_t[1]
                data_type = self.inputs_dtype.get(input_name)
                input_data = np.random.random(size=input_shape).astype(data_type)
                self.inputs_data[input_name] = input_data

    def generate_inputs_data(self, is_om_compare):
        """
        Generate tf2.6 save_model inputs data
        return tf2.6 save_model inputs data directory
        """
        if is_om_compare:
            input_files = sorted(os.listdir(self.input), key=lambda x: int(x[-5]))
            input_bin_data = [np.fromfile(os.path.join(self.input, input_bin_file), dtype=np.float32)
                              for input_bin_file in input_files]
            if len(input_files) != len(self.input_shape_list):
                logger.error("numbers of files in input path and input_shape_list unequal, please check.")
                raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INDEX_OUT_OF_BOUNDS_ERROR)
            for index, bin_data in enumerate(input_bin_data):
                bin_data = bin_data.reshape(self.input_shape_list[index][1])
                self.inputs_data[self.input_shape_list[index][0]] = bin_data
        else:
            input_files = os.listdir(self.input)
            for input_file in input_files:
                file_name = os.path.basename(input_file)
                data_type = file_name.rsplit("_", 1)[-1].split(".")[0]
                input_name = file_name.rsplit("_", 1)[0]
                input_shape_dim = None
                for input_shape in self.input_shape_list:
                    if input_shape[0] == input_name:
                        input_shape_dim = input_shape[1]
                if input_shape_dim is not None:
                    input_data = (np.fromfile(os.path.join(self.input, input_file), dtype=np.dtype(data_type))
                                  .reshape(input_shape_dim))
                    self.inputs_data[input_name] = input_data

    def get_net_output_info(self):
        """
        Compatible with ONNX scenarios
        """
        return self.net_output

    def generate_dump_data(self, tf_json_path):
        """
        Generate tf2.6 save_model dump data
        :return tf2.6 save_model dump data directory
        """
        import tensorflow as tf
        from msprobe.infer.offline.compare.msquickcmp.common.tf_common import load_file_to_read_common_check_with_walk

        op_names = parse_ops_name_from_om_json(tf_json_path)
        with tf.compat.v1.keras.backend.get_session() as sess:
            tag_set = {tf.compat.v1.saved_model.tag_constants.SERVING} if self.tag_set == "" else self.tag_set
            if os.path.isdir(self.model_path):
                load_file_to_read_common_check_with_walk(self.model_path)
            else:
                load_file_to_read_common_check(self.model_path)
            _ = tf.compat.v1.saved_model.load(sess, tag_set, self.model_path)
            if not self.inputs_data:
                raise ValueError("inputs_data is empty")
            feed_dict = {
                sess.graph.get_tensor_by_name(input_name + ":0"): input_data
                for input_name, input_data in self.inputs_data.items()
            }
            output_tensors = []
            operations = sess.graph.get_operations()
            for op_name in op_names:
                if self._is_op_exists(op_name, operations):
                    output_tensors.extend(sess.graph.get_operation_by_name(op_name).outputs)

            out = sess.run(output_tensors, feed_dict)
        self._save_dump_data(out, output_tensors)
        logger.info(f"Dump tf data success, data saved in: {self.dump_data_tf}")

        return self.dump_data_tf

    def _save_dump_data(self, out, output_tensors):
        for data, tensor in zip(out, output_tensors):
            tensor_name = tensor.name.replace("/", "_").replace(":", ".") + "." + str(round(time.time() * 1000000))
            npy_file_path = os.path.join(self.dump_data_tf, tensor_name)
            np.save(npy_file_path, data)

    def _create_dir(self):
        utils.create_directory(self.input)
        utils.create_directory(self.dump_data_tf)
