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
This class mainly involves tf common function.
"""
import os
import subprocess
import numpy as np

from msprobe.core.common.log import logger
from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException
from msprobe.infer.utils.util import load_file_to_read_common_check, filter_cmd
from msprobe.infer.utils.check.rule import Rule

try:
    import tensorflow as tf
except ImportError:
    tf = None
    logger.error("TensorFlow is not installed.")

DTYPE_MAP = {
    tf.float16: np.float16,
    tf.float32: np.float32,
    tf.float64: np.float64,
    tf.int64: np.int64,
    tf.int32: np.int32,
    tf.int16: np.int16,
    tf.int8: np.int8,
    tf.uint8: np.uint8,
    tf.bool: np.bool_,
    tf.complex64: np.complex64
}

TF_DEBUG_TIMEOUT = 3600
VERSION_TF2X = "2."
VERSION_TF1X = "1."
MAX_DEPTH = 5
FILE_MAX_CNT = 100


def check_tf_version(version):
    tf_version = tf.__version__
    if tf_version.startswith(version):
        return True
    return False


def execute_command(cmd: list):
    """ Execute shell command
    :param cmd: command
    :return: status code
    """
    if cmd is None:
        logger.error("Command is None.")
        return -1
    cmd = filter_cmd(cmd)
    logger.info(f"[Run CMD]: {' '.join(cmd)}")
    complete_process = subprocess.run(cmd, shell=False)
    return complete_process.returncode


def convert_to_numpy_type(tensor_type):
    """
    Function Description:
        convert to numpy type
    Parameter:
        tensor_type:the tensor type
    Return Value:
        numpy type
    Exception Description:
        When tensor type not in DTYPE_MAP throw exception
    """
    np_type = DTYPE_MAP.get(tensor_type)
    if np_type is not None:
        return np_type
    logger.error(f"unsupported tensor type: {tensor_type},")
    raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_TENSOR_TYPE_ERROR)


def convert_tensor_shape(tensor_shape):
    """
    Function Description:
        convert tensor shape
    Parameter:
        tensor_shape:the tensor shape
    Return Value:
        shape tuple
    Exception Description:
        When tensor dim is none throw exception
    """
    tensor_shape_list = tensor_shape.as_list()
    for i, _ in enumerate(tensor_shape_list):
        if tensor_shape_list[i] is None:
            logger.error(f"The dynamic shape {tensor_shape} are not supported. "
                         f"Please set '-is' or '--input-shape' to fix the dynamic shape.")
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_NOT_SUPPORT_ERROR)
    return tuple(tensor_shape_list)


def verify_and_adapt_dynamic_shape(input_shapes, op_name, tensor):
    """
    verify and adapt dynamic shape
    """
    try:
        model_shape = list(tensor.shape)
    except ValueError:
        tensor.set_shape(input_shapes.get(op_name))
        return tensor
    if op_name in input_shapes:
        fixed_tensor_shape = input_shapes.get(op_name)
        message = "The fixed input tensor dim not equal to model input dim." \
                  "tensor_name:%s, %s vs %s" % (op_name, str(fixed_tensor_shape), str(model_shape))
        if len(fixed_tensor_shape) != len(model_shape):
            logger.error(message)
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
        for index, dim in enumerate(model_shape):
            fixed_tensor_dim = int(fixed_tensor_shape[index])
            if dim is not None and fixed_tensor_dim != dim:
                logger.error(message)
                raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
            model_shape[index] = fixed_tensor_dim
        logger.info(f"Fix dynamic input shape of %s{op_name} to {model_shape}")
    tensor.set_shape(model_shape)
    return tensor


def get_inputs_tensor(global_graph, input_shape_str):
    """
    get input tensor
    """
    input_shapes = utils.parse_input_shape(input_shape_str)
    inputs_tensor = []
    tensor_index = {}
    operations = global_graph.get_operations()
    op_names = [op.name for op in operations if "Placeholder" == op.type]
    logger.info(op_names)
    for _, tensor_name in enumerate(input_shapes):
        utils.check_input_name_in_model(op_names, tensor_name)
    for op in operations:
        # the operator with the 'Placeholder' type is the input operator of the model
        if "Placeholder" == op.type:
            op_name = op.name
            if op_name in tensor_index:
                tensor_index[op_name] += 1
            else:
                tensor_index[op_name] = 0
            tensor = global_graph.get_tensor_by_name(op.name + ":" + str(tensor_index.get(op_name)))
            tensor = verify_and_adapt_dynamic_shape(input_shapes, op.name, tensor)
            inputs_tensor.append(tensor)
    logger.info(f"model inputs tensor:\n{inputs_tensor}\n")
    return inputs_tensor


def get_inputs_data(inputs_tensor, input_paths):
    inputs_map = {}
    input_path = input_paths.split(",")
    if len(input_path) != len(inputs_tensor):
        logger.error("lengths of input_path and input_tensor unequal, please check.")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INDEX_OUT_OF_BOUNDS_ERROR)
    for index, tensor in enumerate(inputs_tensor):
        try:
            if Rule.input_file().check(input_path[index], will_raise=True):
                input_data = np.fromfile(input_path[index], convert_to_numpy_type(tensor.dtype))
        except Exception as err:
            logger.error(f"Failed to load data {input_path[index]}. {err}")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR) from err
        if tensor.shape:
            input_data = input_data.reshape(tensor.shape)
        inputs_map[tensor] = input_data
        logger.info(f"load file name: {os.path.basename(input_path[index])}, "
                    f"shape: {input_data.shape}, "
                    f"dtype: {input_data.dtype}")
    return inputs_map


def get_model_inputs_dtype(model_path, serving, tag_set):
    inputs_dtype = {}
    with tf.compat.v1.Session() as sess:
        tag_set = {tf.compat.v1.saved_model.tag_constants.SERVING} if tag_set == "" else tag_set
        if os.path.isdir(model_path):
            load_file_to_read_common_check_with_walk(model_path)
        else:
            load_file_to_read_common_check(model_path)
        model = tf.compat.v1.saved_model.load(sess, tag_set, model_path)
        inputs = model.signature_def[serving].inputs
        for _, input_tensor in inputs.items():
            np_dtype = tf.dtypes.as_dtype(input_tensor.dtype).as_numpy_dtype
            inputs_dtype[input_tensor.name.split(':')[0]] = np_dtype
    return inputs_dtype


def split_tag_set(saved_model_tag_set):
    tag_sets = saved_model_tag_set.split(',')
    if len(tag_sets) > 1:
        return tag_sets
    return {tag_sets[0]}


def load_file_to_read_common_check_with_walk(model_path):
    file_cnt = 0
    model_path = os.path.realpath(model_path)
    for root, _, files in os.walk(model_path):
        model_path_len = len(model_path.split('/'))
        root_len = len(root.split('/'))
        if root_len - model_path_len >= MAX_DEPTH:
            logger.error("Parse of TF module depth exceeds the max recursion limit 5.")
            raise RecursionError("Maximum recursion depth exceeded in comparison.")
        for filename in files:
            load_file_to_read_common_check(os.path.join(root, filename))
            file_cnt += 1
            if file_cnt > FILE_MAX_CNT:
                logger.error("The number of tf module files shall not exceed 100.")
                raise ValueError("Files are too much.")
        