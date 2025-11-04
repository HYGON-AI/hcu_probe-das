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
This class mainly involves the accuracy_network_compare function.
"""
import csv
import os
import stat
import re
import sys
import subprocess
import time
from collections import namedtuple

import numpy as np

from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException
from msprobe.infer.utils.file_open_check import sanitize_csv_value, MAX_SIZE_LIMITE_NORMAL_FILE
from msprobe.infer.utils.file_open_check import ms_open
from msprobe.infer.utils.check.rule import Rule

from msprobe.infer.utils.util import load_file_to_read_common_check, filter_cmd
from msprobe.infer.utils.constants import TENSOR_MAX_SIZE

MSACCUCMP_DIR_PATH = "toolkit/tools/operator_cmp/compare"
MSACCUCMP_FILE_NAME = ["msaccucmp.py", "msaccucmp.pyc"]
PYC_FILE_TO_PYTHON_VERSION = "3.7.5"
INFO_FLAG = "[INFO]"
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
READ_WRITE_FLAGS = os.O_RDWR | os.O_CREAT | os.O_TRUNC
# index of each member in compare result_*.csv file
NPU_DUMP_TAG = "NPUDump"
GROUND_TRUTH_TAG = "GroundTruth"
MIN_ELEMENT_NUM = 3
ADVISOR_ARGS = "-advisor"
MAX_CMP_SIZE_ARGS = "--max_cmp_size"


class NetCompare(object):
    """
    Class for compare the entire network
    """

    def __init__(self, npu_dump_data_path, cpu_dump_data_path, output_json_path, arguments, golden_json_path=None):
        self.npu_dump_data_path = npu_dump_data_path
        self.cpu_dump_data_path = cpu_dump_data_path
        self.output_json_path = output_json_path
        self.golden_json_path = golden_json_path
        self.quant_fusion_rule_file = arguments.quant_fusion_rule_file
        self.arguments = arguments
        self.msaccucmp_command_dir_path = os.path.join(self.arguments.cann_path, MSACCUCMP_DIR_PATH)
        self.msaccucmp_command_file_path = self._check_msaccucmp_file(self.msaccucmp_command_dir_path)
        self.python_version = sys.executable.split('/')[-1]

        if self.golden_json_path:
            utils.check_file_size_valid(self.golden_json_path, utils.MAX_READ_FILE_SIZE_4G)
        if self.quant_fusion_rule_file:
            utils.check_file_size_valid(self.quant_fusion_rule_file, utils.MAX_READ_FILE_SIZE_4G)

    @staticmethod
    def execute_command_line(cmd):
        cmd = filter_cmd(cmd)
        utils.logger.info('Execute command:%s' % " ".join(cmd))
        process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return process

    @staticmethod
    def _check_msaccucmp_file(msaccucmp_command_dir_path):
        for file_name in MSACCUCMP_FILE_NAME:
            msaccucmp_command_file_path = os.path.join(msaccucmp_command_dir_path, file_name)
            if os.path.exists(msaccucmp_command_file_path):
                return msaccucmp_command_file_path
            else:
                utils.logger.warning(
                    'The path {} is not exist.Please check the file'.format(msaccucmp_command_file_path))
        utils.logger.error(
            'Does not exist in {} directory msaccucmp.py and msaccucmp.pyc file'.format(msaccucmp_command_dir_path))
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

    @staticmethod
    def _check_pyc_to_python_version(msaccucmp_command_file_path, python_version):
        if msaccucmp_command_file_path.endswith(".pyc"):
            if python_version != PYC_FILE_TO_PYTHON_VERSION:
                utils.logger.error(
                    "The python version for executing {} must be 3.7.5".format(msaccucmp_command_file_path))
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_PYTHON_VERSION_ERROR)

    @staticmethod
    def _catch_compare_result(log_line, catch):
        result = []
        header = []
        try:
            if catch:
                # get the compare result
                info = log_line.decode().split(INFO_FLAG)
                if len(info) > 1:
                    info_content = info[1].strip().split(" ")
                    info_content = [item for item in info_content if item != '']
                    pattern_num = re.compile(r'^([0-9]+)\.?([0-9]+)?')
                    pattern_nan = re.compile(r'NaN', re.I)
                    pattern_header = re.compile(r'Cosine|Error|Distance|Divergence|Deviation', re.I)
                    match = pattern_num.match(info_content[0])
                    if match:
                        result = info_content
                    if not match and pattern_nan.match(info_content[0]):
                        result = info_content
                    if not match and pattern_header.search(info_content[0]):
                        header = info_content
            return result, header
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as error:
            utils.logger.error('Failed to parse the alg compare result!')
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_NET_OUTPUT_ERROR) from error
        finally:
            pass

    def accuracy_network_compare(self):
        """
        Function Description:
            invoke the interface for network-wide comparsion
        Exception Description:
            when invalid  msaccucmp command throw exception
        """
        self._check_pyc_to_python_version(self.msaccucmp_command_file_path, self.python_version)
        msaccucmp_cmd = [
            self.python_version, self.msaccucmp_command_file_path, "compare", "-m",
            self.npu_dump_data_path, "-g",
            self.cpu_dump_data_path, "-f", self.output_json_path, "-out", self.arguments.out_path
        ]
        if self._check_msaccucmp_compare_support_advisor():
            msaccucmp_cmd.append(ADVISOR_ARGS)
        if self._check_msaccucmp_compare_support_max_cmp_size():
            msaccucmp_cmd.extend([MAX_CMP_SIZE_ARGS, str(self.arguments.max_cmp_size)])

        if self.golden_json_path is not None:
            msaccucmp_cmd.extend(["-cf", self.golden_json_path])

        if self.quant_fusion_rule_file:
            msaccucmp_cmd.extend(["-q", self.quant_fusion_rule_file])
        
        status_code, _, _ = self.execute_msaccucmp_command(msaccucmp_cmd)
        if status_code == 2 or status_code == 0:
            utils.logger.info("Finish compare the files in directory %s with those in directory %s." % (
                self.npu_dump_data_path, self.cpu_dump_data_path))
        else:
            utils.logger.error("Failed to execute command: %s" % " ".join(msaccucmp_cmd))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)

    def net_output_compare(self, npu_net_output_data_path, golden_net_output_info):
        """
        net_output_compare
        """
        if not golden_net_output_info:
            return
        npu_dump_file = {}
        file_index = 0
        utils.logger.info("=================================compare Node_output=================================")
        utils.logger.info("start to compare the Node_output at now, compare result is:")
        utils.logger.warning("The comparison of Node_output may be incorrect in certain scenarios. If the precision"
                             " is abnormal, please check whether the mapping between the comparison"
                             " data is correct.")
        for dir_path, _, files in os.walk(npu_net_output_data_path):
            for each_file in sorted(files):
                if each_file.endswith(".npy"):
                    npu_dump_file[file_index] = os.path.join(dir_path, each_file)
                    npu_dump_file[file_index] = load_file_to_read_common_check(npu_dump_file.get(file_index))
                    npu_data = np.load(npu_dump_file.get(file_index))
                    golden_net_output_info[file_index] = \
                            load_file_to_read_common_check(golden_net_output_info.get(file_index))
                    golden_data = np.load(golden_net_output_info.get(file_index))
                    np.save(npu_dump_file.get(file_index), npu_data.reshape(golden_data.shape))
                    msaccucmp_cmd = [
                        self.python_version, self.msaccucmp_command_file_path, "compare", "-m",
                        npu_dump_file.get(file_index), "-g", golden_net_output_info.get(file_index)
                    ]
                    status, compare_result, header = self.execute_msaccucmp_command(msaccucmp_cmd, True)
                    if status == 2 or status == 0:
                        self.save_net_output_result_to_csv(npu_dump_file.get(file_index),
                                                           golden_net_output_info.get(file_index),
                                                           compare_result, header)
                        utils.logger.info("Compare Node_output:{} completely.".format(file_index))
                    else:
                        utils.logger.error("Failed to execute command: %s" % " ".join(msaccucmp_cmd))
                        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
                    file_index += 1
        return

    def save_net_output_result_to_csv(self, npu_file, golden_file, result, header):
        """
        save_net_output_result_to_csv
        """
        result_file_path = None
        result_file_backup_path = None
        npu_file_name = os.path.basename(npu_file)
        golden_file_name = os.path.basename(golden_file)
        for dir_path, _, files in os.walk(self.arguments.out_path):
            files = [file for file in files if file.endswith("csv")]
            if files:
                result_file_path = os.path.join(dir_path, files[0])
                result_file_backup = "{}_bak.csv".format(files[0].split(".")[0])
                result_file_backup_path = os.path.join(dir_path, result_file_backup)
                break
        try:
            if not self.arguments.dump:
                result_file_path = ""
                for f in os.listdir(self.arguments.out_path):
                    if f.endswith(".csv"):
                        result_file_path = os.path.join(self.arguments.out_path, f)
                        break
                if not result_file_path:
                    time_suffix = time.strftime("%Y%m%d%H%M%S", time.localtime())
                    file_name = "result_" + time_suffix + ".csv"
                    result_file_path = os.path.join(self.arguments.out_path, file_name)
                else:
                    header = []
                CsvInfo = namedtuple('CsvInfo', 'npu_file_name, golden_file_name, result, header')
                csv_info = CsvInfo(npu_file_name, golden_file_name, result, header)
                with ms_open(result_file_path, "a+") as fp_writer:
                    self._process_result_to_csv(fp_writer, csv_info)
            else:
                # read result file and write it to backup file,update the result of compare Node_output
                Rule.input_file().check(result_file_path, will_raise=True)
                with ms_open(result_file_path, "r", max_size=TENSOR_MAX_SIZE) as fp_read:
                    with ms_open(result_file_backup_path, 'w', newline="") as fp_write:
                        self._process_result_one_line(fp_write, fp_read, npu_file_name, golden_file_name, result)
                os.remove(result_file_path)
                os.rename(result_file_backup_path, result_file_path)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as error:
            utils.logger.error('Failed to write Net_output compare result')
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_NET_OUTPUT_ERROR) from error
        finally:
            pass

    def execute_msaccucmp_command(self, cmd, catch=False):
        """
        Function Description:
            run the following command
        Parameter:
            cmd: command
        Return Value:
            status code
        """
        result = []
        header = []
        process = self.execute_command_line(cmd)
        while process.poll() is None:
            line = process.stdout.readline().strip()
            if line:
                compare_result, header_result = self._catch_compare_result(line, catch)
                result = compare_result if compare_result else result
                header = header_result if header_result else header
        return process.returncode, result, header

    def _process_result_one_line(self, fp_write, fp_read, npu_file_name, golden_file_name, result):
        writer = csv.writer(fp_write)
        # write header to file
        table_header_info = next(fp_read)
        header_list = table_header_info.strip().split(',')
        writer.writerow(header_list)
        npu_dump_index = header_list.index(NPU_DUMP_TAG)
        ground_truth_index = header_list.index(GROUND_TRUTH_TAG)

        result_reader = csv.reader(fp_read)
        # update result data
        new_content = []
        for line in result_reader:
            if len(line) < MIN_ELEMENT_NUM:
                utils.logger.warning('The content of line is {}'.format(line))
                continue
            if line[npu_dump_index] != "Node_Output":
                for ele in line:
                    sanitize_csv_value(ele)
                writer.writerow(line)
            else:
                new_content = [
                    line[0], "NaN", "Node_Output", "NaN", "NaN",
                    npu_file_name, "NaN", golden_file_name, "[]"
                ]
                if self._check_msaccucmp_compare_support_advisor():
                    new_content.append("NaN")
                new_content.extend(result)
                new_content.extend([""])
                if line[ground_truth_index] != "*":
                    writer.writerow(line)
        writer.writerow(new_content)

    def _check_msaccucmp_compare_support_args(self, compare_args):
        check_cmd = [self.python_version, self.msaccucmp_command_file_path, "compare", "-h"]
        process = self.execute_command_line(check_cmd)
        while process.poll() is None:
            line = process.stdout.readline().strip()
            if line:
                line_decode = line.decode(encoding="utf-8")
                if compare_args in line_decode:
                    return True
        else:
            utils.logger.warning(f'Current version does not support {compare_args} function')
            return False

    def _check_msaccucmp_compare_support_advisor(self):
        return self.arguments.advisor and \
               self._check_msaccucmp_compare_support_args(ADVISOR_ARGS)
    
    def _check_msaccucmp_compare_support_max_cmp_size(self):
        return self.arguments.max_cmp_size and \
               self._check_msaccucmp_compare_support_args(MAX_CMP_SIZE_ARGS)


    def _process_result_to_csv(self, fp_write, csv_info):
        writer = csv.writer(fp_write)
        if csv_info.header:
            header_base_info = [
                'Index', 'OpType', 'NPUDump', 'DataType', 'Address',
                'GroundTruth', 'DataType', 'TensorIndex', 'Shape'
                ]
            header_base_info.extend(csv_info.header)
            writer.writerow(header_base_info)
        fp_write.seek(0, 0)
        index = len(fp_write.readlines()) - 1
        new_content = [
            str(index), "NaN", "Node_Output", "NaN", "NaN",
            csv_info.npu_file_name, "NaN", csv_info.golden_file_name, "[]"
        ]
        new_content.extend(csv_info.result)
        writer.writerow(new_content)