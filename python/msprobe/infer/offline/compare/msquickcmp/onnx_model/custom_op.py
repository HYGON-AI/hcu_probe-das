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
This class is used to modify ONNX model with Npu custom op.
"""
import os

import numpy as np

from msprobe.infer.offline.surgeon.auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException

from msprobe.infer.utils.check.rule import Rule


DEFORMABLE_CONV2D_TYPE = "DeformableConv2D"
BATCH_MULTI_CLASS_NMS_TYPE = "BatchMultiClassNMS"
ROI_EXTRACTOR_TYPE = "RoiExtractor"


class CustomOp():
    """
    Class for modify onnx model with custom op  
    """
    def __init__(self, custom_op_type, origin_model, npu_dump_path, output_model_path) -> None:
        self.custom_op_type = custom_op_type.split(',')
        self.origin_model = origin_model
        self.npu_dump_path = npu_dump_path
        self.output_model_path = output_model_path

        for custom_op in self.custom_op_type:
            if custom_op not in CUSTIOM_OP_MODIFY_FUNC.keys():
                utils.logger.error("custom op type:%s is invalids, now supports: "
                                   "'DeformableConv2D'、'BatchMultiClassNMS'、'RoiExtractor'", custom_op)
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)

    def remove_custom_op_and_add_inputs(self):
        g = OnnxGraph.parse(self.origin_model)
        g.infer_shape()
        inputs_map = {}
        for custom_op in self.custom_op_type:
            func = CUSTIOM_OP_MODIFY_FUNC.get(custom_op)
            inputs_map.update(func(g, self.npu_dump_path))
        
        file_name = "custom_op_" + os.path.basename(self.origin_model)
        custom_op_path = os.path.join(self.output_model_path, file_name)

        g.infer_shape()
        g.save(custom_op_path)

        utils.logger.info("modify model with custom op succeed, save path: %s", custom_op_path)
        return custom_op_path, inputs_map


def convert_nc1hwc0_to_nchw(shape_from: list, array: any) -> any:
    """
    Convert the data format from NC1HWC0 to NCHW
    :param shape_from: the shape before convert
    :param array: the one-dimensional array
    :return: the data array of NCHW shape
    """
    n_from = shape_from[0]
    c1_from = shape_from[1]
    h_from = shape_from[2]
    w_from = shape_from[3]
    c0_from = shape_from[4]

    array_shape = array.reshape(n_from, c1_from, h_from, w_from, c0_from)
    tmp_input_tensor = np.transpose(array_shape, axes=(0, 1, 4, 2, 3))
    tmp_input_tensor = tmp_input_tensor.reshape((n_from, c1_from * c0_from, h_from, w_from))
    return tmp_input_tensor


def get_deformable_conv2d_inputs_from_npu_dump(npu_dump_path):
    inputs_map = {}
    
    for item in os.listdir(npu_dump_path):
        # file name format: [Optype].[OpName].{time}.[dump_type].[index].npy
        file_name_info = item.split('.')
        if len(file_name_info) < 5:
            continue

        op_type = file_name_info[0]
        op_name = file_name_info[1]
        dump_type = file_name_info[-3]

        if DEFORMABLE_CONV2D_TYPE in op_name and op_type == "Conv2D" and dump_type == "output":
            dump_path = os.path.join(npu_dump_path, item)
            try:
                Rule.input_file().check(dump_path, will_raise=True)
                np_data = np.load(dump_path)
            except Exception as err:
                utils.logger.error(f"Load npu dump failed, exception is {err}, please check!")
                raise

            if len(np_data.shape) == 5:
                np_data = convert_nc1hwc0_to_nchw(np_data.shape, np_data.flatten())

            inputs_map[op_name] = np_data

    return inputs_map


def remove_deformable_conv2d_and_add_inputs(g: OnnxGraph, npu_dump_path):
    extend_inpus_map = get_deformable_conv2d_inputs_from_npu_dump(npu_dump_path=npu_dump_path)
    inputs_map = {}

    for node in g.nodes:
        if node.op_type != DEFORMABLE_CONV2D_TYPE:
            continue

        extend_inpus_map_key = node.name + "_conv2d"
        if extend_inpus_map_key not in extend_inpus_map:
            continue

        input_name = node.outputs[0]
        shape = extend_inpus_map[extend_inpus_map_key].shape

        g.remove(node.name, {})
        g.add_input(input_name, np.float32, shape)
        inputs_map[input_name] = extend_inpus_map[extend_inpus_map_key].astype(np.float32)
        utils.logger.info("remove deforable_conv2d custom op: %s "
                          "and add model input: %s, input shape: %s", node.name, input_name, shape)
    return inputs_map


def get_batch_multi_class_nms_inputs_from_npu_dump(npu_dump_path):
    inputs_map = {}
    
    for item in os.listdir(npu_dump_path):
        # file name format: [Optype].[OpName].{time}.[dump_type].[index].npy
        file_name_info = item.split('.')
        if len(file_name_info) < 5:
            continue

        op_type = file_name_info[0]
        op_name = file_name_info[1]
        dump_type = file_name_info[-3]
        index = file_name_info[-2]

        if BATCH_MULTI_CLASS_NMS_TYPE in op_name and \
            op_type == "BatchMultiClassNonMaxSuppression" and dump_type == "output":
            dump_path = os.path.join(npu_dump_path, item)
            try:
                Rule.input_file().check(dump_path, will_raise=True)
                np_data = np.load(dump_path)
            except Exception as err:
                utils.logger.error(f"Load npu dump failed, exception is {err}, please check!")
                raise

            op_name_info = op_name.split('_')
            if len(op_name_info) < 2:
                utils.logger.error(f"{op_name} cannot split by '_', please check.")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INDEX_OUT_OF_BOUNDS_ERROR)
            op_name = op_name_info[0] + '_' + op_name_info[1]

            inputs_map_key = op_name + ("_%s" % index)
            inputs_map[inputs_map_key] = np_data

    return inputs_map


def remove_batch_multi_class_nms_and_add_inputs(g: OnnxGraph, npu_dump_path):
    extend_inpus_map = get_batch_multi_class_nms_inputs_from_npu_dump(npu_dump_path)
    inputs_map = {}

    for node in g.nodes:
        if node.op_type != BATCH_MULTI_CLASS_NMS_TYPE:
            continue
        
        for index, output in enumerate(node.outputs):
            next_nodes = g.get_next_nodes(output)
            if not next_nodes:
                continue

            custom_op_name = node.name
            extend_inpus_map_key = custom_op_name + ("_%s" % index)

            if extend_inpus_map_key not in extend_inpus_map:
                continue

            data_type = extend_inpus_map[extend_inpus_map_key].dtype
            shape = extend_inpus_map[extend_inpus_map_key].shape

            extend_input_name = output
            g.add_input(extend_input_name, data_type, shape)
            inputs_map[extend_input_name] = extend_inpus_map[extend_inpus_map_key].astype(data_type)

            utils.logger.info("remove deforable_conv2d custom op: %s "
                              "and add model input: %s, input shape: %s, data_type: %s", 
                              node.name, extend_input_name, shape, data_type)
            
        g.remove(node.name, {})

    return inputs_map


def get_roi_extractor_inputs_from_npu_dump(npu_dump_path):
    inputs_map = {}
    
    for item in os.listdir(npu_dump_path):
        # file name format: [Optype].[OpName].{time}.[dump_type].[index].npy
        file_name_info = item.split('.')
        if len(file_name_info) < 5:
            continue

        op_type = file_name_info[0]
        op_name = file_name_info[1]

        dump_type = file_name_info[-3]
        if ROI_EXTRACTOR_TYPE in op_name and op_type == ROI_EXTRACTOR_TYPE and dump_type == "output":
            dump_path = os.path.join(npu_dump_path, item)
            try:
                Rule.input_file().check(dump_path, will_raise=True)
                np_data = np.load(dump_path)
            except Exception as err:
                utils.logger.error(f"Load npu dump failed, exception is {err}, please check!")
                raise

            if len(np_data.shape) == 5:
                np_data = convert_nc1hwc0_to_nchw(np_data.shape, np_data.flatten())

            op_name_info = op_name.split('_')
            op_name = op_name_info[0] + '_' + op_name_info[1]
            inputs_map[op_name] = np_data

    return inputs_map


def remove_roi_extractor_and_add_inputs(g: OnnxGraph, npu_dump_path):
    extend_inpus_map = get_roi_extractor_inputs_from_npu_dump(npu_dump_path)
    inputs_map = {}

    for node in g.nodes:
        if node.op_type != ROI_EXTRACTOR_TYPE:
            continue

        input_name = node.outputs[0]
        if node.name not in extend_inpus_map:
            continue
    
        shape = extend_inpus_map[node.name].shape
        data_type = np.float32

        g.remove(node.name, {})
        g.add_input(input_name, data_type, shape)
        inputs_map[input_name] = extend_inpus_map[node.name].astype(data_type)
        utils.logger.info("remove deforable_conv2d custom op: %s "
                          "and add model input: %s, input shape: %s, data_type: %s", 
                          node.name, input_name, shape, data_type)
    return inputs_map


CUSTIOM_OP_MODIFY_FUNC = \
{
    DEFORMABLE_CONV2D_TYPE: remove_deformable_conv2d_and_add_inputs,
    BATCH_MULTI_CLASS_NMS_TYPE: remove_batch_multi_class_nms_and_add_inputs,
    ROI_EXTRACTOR_TYPE: remove_roi_extractor_and_add_inputs
}
