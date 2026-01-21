# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

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

import numpy as np

from msprobe.core.common.log import logger
from msprobe.infer.offline.compare.msquickcmp.common.utils import check_file_size_valid, AccuracyCompareException, \
    ACCURACY_COMPARISON_NET_OUTPUT_ERROR, ACCURACY_COMPARISON_INVALID_DATA_ERROR, MAX_READ_FILE_SIZE_4G
from msprobe.infer.utils.file_open_check import sanitize_csv_value
from msprobe.infer.utils.file_open_check import ms_open
from msprobe.infer.utils.check.rule import Rule
from msprobe.infer.utils.util import load_file_to_read_common_check, filter_cmd
from msprobe.infer.utils.constants import TENSOR_MAX_SIZE
from msprobe.msaccucmp import msaccucmp

INFO_FLAG = "[INFO]"
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
READ_WRITE_FLAGS = os.O_RDWR | os.O_CREAT | os.O_TRUNC
# index of each member in compare result_*.csv file
NPU_DUMP_TAG = "NPUDump"
GROUND_TRUTH_TAG = "GroundTruth"
MIN_ELEMENT_NUM = 3


class NetCompare(object):
    """
    Class for compare the entire network
    """

    def __init__(self, npu_dump_data_path, cpu_dump_data_path, output_json_path, arguments, golden_json_path=None):
        self.npu_dump_data_path = npu_dump_data_path
        self.cpu_dump_data_path = cpu_dump_data_path
        self.output_json_path = output_json_path
        self.golden_json_path = golden_json_path
        self.arguments = arguments
        self.python_version = sys.executable.split('/')[-1]

        if self.golden_json_path:
            check_file_size_valid(self.golden_json_path, MAX_READ_FILE_SIZE_4G)

    @staticmethod
    def execute_command_line(cmd):
        logger.info(f"Execute command:{' '.join(cmd)}")
        process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return process

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
            logger.error('Failed to parse the alg compare result!')
            raise AccuracyCompareException(ACCURACY_COMPARISON_NET_OUTPUT_ERROR) from error
        finally:
            pass

    @staticmethod
    def _process_result_one_line(fp_write, fp_read, npu_file_name, golden_file_name, result):
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
                logger.warning(f"The content of line is {line}")
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
                new_content.extend(result)
                new_content.extend([""])
                if line[ground_truth_index] != "*":
                    writer.writerow(line)
        writer.writerow(new_content)

    @staticmethod
    def _process_result_to_csv(fp_write, csv_info):
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

    def accuracy_network_compare(self):
        """
        Function Description:
            invoke the interface for network-wide comparison
        Exception Description:
            when invalid msaccucmp command throw exception
        """
        msaccucmp_cmd = [
            self.python_version, msaccucmp.__file__, "compare", "-m",
            self.npu_dump_data_path, "-g",
            self.cpu_dump_data_path, "-f", self.output_json_path, "-out", self.arguments.output_path
        ]

        if self.golden_json_path is not None:
            msaccucmp_cmd.extend(["-cf", self.golden_json_path])

        msaccucmp_cmd = filter_cmd(msaccucmp_cmd)
        status_code, _, _ = self.execute_msaccucmp_command(msaccucmp_cmd)
        if status_code == 2 or status_code == 0:
            logger.info(f"Finish compare the files in directory {self.npu_dump_data_path} "
                        f"with those in directory {self.cpu_dump_data_path}.")
        else:
            logger.error(f"Failed to execute command: {' '.join(msaccucmp_cmd)}")
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_DATA_ERROR)

    def net_output_compare(self, npu_net_output_data_path, golden_net_output_info):
        """
        net_output_compare
        """
        if not golden_net_output_info:
            return
        npu_dump_file = {}
        file_index = 0
        logger.info("=================================compare Node_output=================================")
        logger.info("start to compare the Node_output at now, compare result is:")
        logger.warning("The comparison of Node_output may be incorrect in certain scenarios. "
                       "If the precision is abnormal, "
                       "please check whether the mapping between the comparison data is correct.")
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
                        logger.info(f"Compare Node_output:{file_index} completely.")
                    else:
                        logger.error(f"Failed to execute command: {' '.join(msaccucmp_cmd)}")
                        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_DATA_ERROR)
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
        for dir_path, _, files in os.walk(self.arguments.output_path):
            files = [file for file in files if file.endswith("csv")]
            if files:
                result_file_path = os.path.join(dir_path, files[0])
                result_file_backup = "{}_bak.csv".format(files[0].split(".")[0])
                result_file_backup_path = os.path.join(dir_path, result_file_backup)
                break
        try:
            # read result file and write it to backup file,update the result of compare Node_output
            Rule.input_file().check(result_file_path, will_raise=True)
            with ms_open(result_file_path, "r", max_size=TENSOR_MAX_SIZE) as fp_read:
                with ms_open(result_file_backup_path, 'w', newline="") as fp_write:
                    self._process_result_one_line(fp_write, fp_read, npu_file_name, golden_file_name, result)
            os.remove(result_file_path)
            os.rename(result_file_backup_path, result_file_path)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as error:
            logger.error("Failed to write Net_output compare result")
            raise AccuracyCompareException(ACCURACY_COMPARISON_NET_OUTPUT_ERROR) from error
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

    def _check_msaccucmp_compare_support_args(self, compare_args):
        check_cmd = [self.python_version, self.msaccucmp_command_file_path, "compare", "-h"]
        check_cmd = filter_cmd(check_cmd)
        process = self.execute_command_line(check_cmd)
        while process.poll() is None:
            line = process.stdout.readline().strip()
            if line:
                line_decode = line.decode(encoding="utf-8")
                if compare_args in line_decode:
                    return True
        else:
            logger.warning(f'Current version does not support {compare_args} function')
            return False
