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

import os
import sys
import numpy as np

from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException
from msprobe.infer.utils.security_check import ms_makedirs
from msprobe.infer.utils.util import load_file_to_read_common_check


MSACCUCMP_FILE_PATH = "toolkit/tools/operator_cmp/compare/msaccucmp.py"


def convert_bin_dump_data_to_npy(npu_dump_data_path, npu_net_output_data_path, cann_path):
    """
    Function Description:
        provide the interface for dump data conversion
    Exception Description:
        when invalid msaccucmp command throw exception
    """
    common_path = os.path.commonprefix([npu_dump_data_path, npu_net_output_data_path])
    npu_dump_data_path_diff = os.path.relpath(npu_dump_data_path, common_path)
    try:
        time_stamp_file_path = npu_dump_data_path_diff.split(os.path.sep)[1]
    except IndexError as e:
        utils.logger.error("npu_dump_data_path_diff cannot be split by path separators.")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INDEX_OUT_OF_BOUNDS_ERROR) from e
    convert_dir_path = npu_dump_data_path.replace(time_stamp_file_path, time_stamp_file_path + '_bin2npy')
    convert_dir_path = os.path.normpath(convert_dir_path)
    if not os.path.exists(convert_dir_path):
        ms_makedirs(convert_dir_path)
    msaccucmp_command_file_path = os.path.join(cann_path, MSACCUCMP_FILE_PATH)
    python_version = sys.executable.split('/')[-1]
    bin2npy_cmd = [
        python_version,
        msaccucmp_command_file_path,
        "convert",
        "-d",
        npu_dump_data_path,
        "-out",
        convert_dir_path,
    ]
    utils.execute_command(bin2npy_cmd, False)
    utils.logger.info("msaccucmp command line: %s " % " ".join(bin2npy_cmd))
    return convert_dir_path


def convert_bin_file_to_npy(bin_file_path, npy_dir_path, cann_path):
    """
    Function Description:
        convert a bin file to npy file.
    Parameter:
        bin_file_path: the path of the bin file needed to be converted to npy
        npy_dir_path: the dest dir to save the converted npy file
        cann_path: user or system cann_path for using msaccucmp.py
    """
    python_version = sys.executable.split('/')[-1]
    msaccucmp_command_file_path = os.path.join(cann_path, MSACCUCMP_FILE_PATH)
    bin2npy_cmd = [python_version, msaccucmp_command_file_path, "convert", "-d", bin_file_path, "-out", npy_dir_path]
    utils.logger.info("convert dump data: %s to npy file" % (bin_file_path))
    utils.execute_command(bin2npy_cmd)


def convert_npy_to_bin(npy_input_path):
    """
    Function Description:
        convert a  file to bin file.
    Parameter:
        npy_file_path: the path of the npy file needed to be converted to bin
    """
    input_initial_path = npy_input_path.split(",")
    outputs = []
    for input_item in input_initial_path:
        input_item_path = os.path.realpath(input_item)
        if input_item_path.endswith('.npy'):
            bin_item = input_item[:-4] + '.bin'
            bin_path = input_item_path[:-4] + '.bin'
            input_item_path = load_file_to_read_common_check(input_item_path)
            npy_data = np.load(input_item_path)
            if os.path.islink(bin_path):
                os.unlink(bin_path)
            if os.path.exists(bin_path):
                os.remove(bin_path)
            npy_data.tofile(bin_path)
            outputs.append(bin_item)
        else:
            outputs.append(input_item)
    return ",".join(outputs)
