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
This class is used to generate GUP dump data of the tf model.
"""
import os
import re
import sys
import time

import numpy as np
import pexpect

from msprobe.core.common.log import logger
from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.dump_data import DumpData
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException
from msprobe.infer.utils.file_open_check import ms_open
from msprobe.infer.utils.constants import TENSOR_MAX_SIZE
from msprobe.infer.utils.util import load_file_to_read_common_check, check_str_for_cmd


class TfDumpData(DumpData):
    """
    This class is used to generate GUP dump data of the tf model.
    """

    def __init__(self, arguments):
        from msprobe.infer.offline.compare.msquickcmp.common import tf_common

        super().__init__()
        self.args = arguments
        output_path = os.path.realpath(self.args.out_path)
        self.important_dirs = {
            "input": os.path.join(output_path, "input"),
            "dump_data_tf": os.path.join(output_path, "dump_data/tf"),
            "tmp": os.path.join(output_path, "tmp")
        }
        self.global_graph = None
        self.input_path = self.args.input_path
        self.net_output_name = []
        self.net_output = {}
        self._load_graph()
        self._create_dir()
        self.tf_common = tf_common

    def generate_inputs_data(self, npu_dump_data_path, use_aipp):
        """
        Generate tf model inputs data
        :return tf model inputs data directory
        """
        inputs_tensor = self.tf_common.get_inputs_tensor(self.global_graph, self.args.input_shape)
        self._make_inputs_data(inputs_tensor)

    def generate_dump_data(self, output_json_path=None, npu_dump_path=None, om_parser=None):
        """
        Generate tf model dump data
        :return tf model dump data directory
        """
        outputs_tensor = self._get_outputs_tensor()
        if self.tf_common.check_tf_version(self.tf_common.VERSION_TF2X):
            self._run_model_tf2x(outputs_tensor)
        elif self.tf_common.check_tf_version(self.tf_common.VERSION_TF1X):
            self._run_model_tf1x(outputs_tensor)

        return self.important_dirs.get("dump_data_tf")

    def get_net_output_info(self):
        """
        Compatible with ONNX scenarios
        """
        return self.net_output

    def _create_dir(self):
        # create input directory
        utils.create_directory(self.important_dirs.get("input"))

        # create dump_data/tf directory
        utils.create_directory(self.important_dirs.get("dump_data_tf"))

        # create tmp directory
        utils.create_directory(self.important_dirs.get("tmp"))

    def _load_graph(self):
        import tensorflow as tf

        try:
            self.args.model_path = load_file_to_read_common_check(self.args.model_path)
            with tf.io.gfile.GFile(self.args.model_path, 'rb') as f:
                global_graph_def = tf.compat.v1.GraphDef.FromString(f.read())
        except Exception as err:
            logger.error(f"Failed to load the model {self.args.model_path}. {err}")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_OPEN_FILE_ERROR) from err
        self.global_graph = tf.Graph()
        try:
            with self.global_graph.as_default():
                tf.import_graph_def(global_graph_def, name='')
        except Exception as err:
            logger.error(f"Failed to load the model {self.args.model_path}. {err}")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_OPEN_FILE_ERROR) from err
        logger.info(f"Load the model {self.args.model_path} successfully.")

    def _make_inputs_data(self, inputs_tensor):
        if self.args.input_path == "":
            if os.listdir(self.important_dirs.get("input")):
                input_path = self.important_dirs.get("input")
                self.input_path = ','.join([os.path.join(input_path, ii) for ii in os.listdir(input_path)])
                return

            input_path_list = []
            for index, tensor in enumerate(inputs_tensor):
                if not tensor.shape:
                    logger.error(f"The shape of {tensor.name} is unknown. Please usr -i to assign the input path.")
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR)
                input_data = np.random.random(self.tf_common.convert_tensor_shape(tensor.shape)) \
                    .astype(self.tf_common.convert_to_numpy_type(tensor.dtype))
                input_path = os.path.join(self.important_dirs.get("input"), "input_" + str(index) + ".bin")
                input_path_list.append(input_path)
                try:
                    input_data.tofile(input_path)
                except Exception as err:
                    logger.error(f"Failed to generate data {input_path}. {err}")
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR) from err
                logger.info(f"file name: {input_path}, shape: {input_data.shape}, dtype: {input_data.dtype}")
                self.input_path = ','.join(input_path_list)
        else:
            input_path = self.args.input_path.split(",")
            if len(inputs_tensor) != len(input_path):
                logger.error(f"the number of model inputs tensor is not equal the number of inputs data, "
                             f"inputs tensor is: {len(inputs_tensor)}, inputs data is: {len(input_path)}")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)

    def _run_model_tf2x(self, outputs_tensor):
        tf2x_runner_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../", "tf_debug_runner.py")
        cmd = '%s %s -m %s -i "%s" --output-nodes "%s" -o %s' \
              % (sys.executable, tf2x_runner_path, self.args.model_path, self.input_path,
                 ";".join(outputs_tensor), self.important_dirs.get("dump_data_tf"))
        for _, tensor_name in enumerate(outputs_tensor):
            self.net_output_name.append(tensor_name)
        if self.args.input_shape:
            cmd += " -s " + '"' + self.args.input_shape + '"'
        self.tf_common.execute_command(cmd)

    def _run_model_tf1x(self, outputs_tensor):
        tf_debug_runner_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../", "tf_debug_runner.py")
        cmd = '%s %s -m %s -i "%s" --output-nodes "%s" -o %s' \
              % (sys.executable, tf_debug_runner_path, self.args.model_path, self.input_path,
                 ";".join(outputs_tensor), os.path.join(self.important_dirs.get("tmp"), "tf_dbg"))
        for _, tensor_name in enumerate(outputs_tensor):
            self.net_output_name.append(tensor_name)
        if self.args.input_shape:
            cmd += " -s " + self.args.input_shape
        self._run_tf_dbg_dump(cmd)

    def _make_pt_command(self, tensor_name_path):
        pt_command_list = []
        tensor_count = {}
        with ms_open(tensor_name_path, max_size=TENSOR_MAX_SIZE) as tensor_name_file:
            # skip 3 line
            next(tensor_name_file)
            next(tensor_name_file)
            next(tensor_name_file)
            # start to convert tensor to pt command
            for line in tensor_name_file:
                new_line = line.strip()
                tensor_name = new_line[new_line.rfind(' ') + 1:]
                if tensor_name not in tensor_count:
                    tensor_count[tensor_name] = 0
                else:
                    tensor_count[tensor_name] += 1
                count_tensor_name = tensor_count.get(tensor_name)
                npy_file_name = (f"{tensor_name.replace('/', '_').replace(':', '.')}."
                                 f"{str(round(time.time() * 1000000))}."
                                 f"{count_tensor_name}.npy")
                npy_file_path = os.path.join(self.important_dirs.get("dump_data_tf"), npy_file_name)
                # get the net_output dump file info
                if tensor_name in self.net_output_name:
                    self.net_output[self.net_output_name.index(tensor_name)] = npy_file_path
                check_str_for_cmd(tensor_name, 'tensor_name')
                pt_command_list.append("pt %s -n %d -w %s" % (tensor_name, count_tensor_name, npy_file_path))
        return pt_command_list

    def _run_tf_dbg_dump(self, cmd_line):
        """Run tf debug with pexpect, should set tf debug ui_type='readline'"""
        tf_dbg = pexpect.spawn(cmd_line)
        tf_dbg.logfile = sys.stdout.buffer
        tf_dbg.expect('tfdbg>', timeout=self.tf_common.TF_DEBUG_TIMEOUT)
        logger.info("Start to run. Please wait....")
        tf_dbg.sendline('run')
        index = tf_dbg.expect(['An error occurred during the run', 'tfdbg>'], timeout=self.tf_common.TF_DEBUG_TIMEOUT)
        if index == 0:
            tf_dbg.sendline('exit')
            logger.error(f"Failed to run command: {cmd_line}")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_PYTHON_COMMAND_ERROR)
        tensor_name_path = os.path.join(self.important_dirs.get("tmp"), 'tf_tensor_names.txt')
        tf_dbg.sendline('lt > %s' % tensor_name_path)
        tf_dbg.expect('tfdbg>', timeout=self.tf_common.TF_DEBUG_TIMEOUT)
        if not os.path.exists(tensor_name_path):
            tf_dbg.sendline('exit')
            logger.error("Failed to save tensor name to file.")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_PYTHON_COMMAND_ERROR)
        logger.info(f"Save tensor name to {tensor_name_path} successfully.")
        pt_command_list = self._make_pt_command(tensor_name_path)
        logger.info(f"Start to run {len(pt_command_list)} pt commands. Please wait...")
        for cmd in pt_command_list:
            tf_dbg.sendline(cmd.strip())
            tf_dbg.expect('tfdbg>', timeout=self.tf_common.TF_DEBUG_TIMEOUT)
        tf_dbg.sendline('exit')
        logger.info('Finish dump tf data.')

    def _get_all_node_and_input_node(self):
        input_nodes = []
        node_list = []
        operations = self.global_graph.get_operations()
        for op in operations:
            node_list.append(op.name)
            for tensor in op.inputs:
                input_name = tensor.name.split(':')[0]
                if input_name not in input_nodes:
                    input_nodes.append(input_name)
        return input_nodes, node_list

    def _check_node_output(self, node_name):
        op = self.global_graph.get_operation_by_name(node_name)
        if op.outputs and not node_name.endswith("ReadVariableOp") and "/cond/" not in node_name:
            return True
        return False

    def _check_output_nodes_valid(self, outputs_tensor, node_list):
        for tensor in outputs_tensor:
            tensor_info = tensor.strip().split(':')
            if len(tensor_info) != 2:
                logger.error(
                    f"Invalid output nodes ({tensor}). Only support 'name1:0;name2:1'. Please check the output node.")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            node_name = tensor_info[0].strip()  # 0 for node_name
            if not node_name:
                logger.error(
                    f"Invalid output nodes ({tensor}). Only support 'name1:0;name2:1'. Please check the output node.")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            if node_name not in node_list:
                logger.error(f"The output node ({node_name}) is not in the graph. Please check the output node.")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            index = tensor_info[1].strip()  # 1 for tensor_index
            if not index:
                logger.error(
                    f"Invalid output nodes ({tensor}). Only support 'name1:0;name2:1'. Please check the output node.")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            op = self.global_graph.get_operation_by_name(node_name)
            pattern = re.compile(r'^[0-9]+$')
            match = pattern.match(index)
            if match is None:
                logger.error(f"The index ({index}) of {node_name} is invalid. Please check the output node.")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            if int(index) < 0 or int(index) >= len(op.outputs):
                logger.error(f"The index ({index}) of {node_name} out of range [0, {len(op.outputs)}). "
                             f"Please check the output node.")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)

    def _get_outputs_tensor(self):
        input_nodes, node_list = self._get_all_node_and_input_node()
        outputs_tensor = []
        if self.args.output_nodes:
            outputs_tensor = self.args.output_nodes.strip().split(';')
            self._check_output_nodes_valid(outputs_tensor, node_list)
        else:
            output_nodes = list(set(node_list).difference(set(input_nodes)))
            for name in output_nodes:
                if self._check_node_output(name):
                    outputs_tensor.append(name + ":0")
        logger.info(f"The outputs tensor:\n{outputs_tensor}\n")
        return outputs_tensor
