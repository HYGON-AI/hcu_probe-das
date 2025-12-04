# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import re

PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")

CONFIG_FILE_MAX_SIZE = 1 * 1024 * 1024  # work for .ini config file
TEXT_FILE_MAX_SIZE = 100 * 1024 * 1024  # work for txt, py
CSV_FILE_MAX_SIZE = 1024 * 1024 * 1024
JSON_FILE_MAX_SIZE = 1024 * 1024 * 1024
ONNX_MODEL_MAX_SIZE = 2 * 1024 * 1024 * 1024
TENSOR_MAX_SIZE = 10 * 1024 * 1024 * 1024
MODEL_WEIGHT_MAX_SIZE = 300 * 1024 * 1024 * 1024
INPUT_FILE_MAX_SIZE = 5 * 1024 * 1024 * 1024
LOG_FILE_MAX_SIZE = 100 * 1024 * 1024

EXT_SIZE_MAPPING = {
    ".ini": CONFIG_FILE_MAX_SIZE,
    '.csv': CSV_FILE_MAX_SIZE,
    '.json': JSON_FILE_MAX_SIZE,
    '.txt': TEXT_FILE_MAX_SIZE,
    '.py': TEXT_FILE_MAX_SIZE,
    '.pth': MODEL_WEIGHT_MAX_SIZE,
    '.bin': MODEL_WEIGHT_MAX_SIZE,
    '.onnx': ONNX_MODEL_MAX_SIZE,
}

MAX_RECUR_DEPTH = 998


class FileCheckConst:
    """
    Class for file check const
    """
    READ_ABLE = "r"
    WRITE_ABLE = "w"
    EXECUTE_ABLE = "x"
    READ_WRITE_ABLE = "rw"
    READ_EXECUTE_ABLE = "rx"
    WRITE_EXECUTE_ABLE = "wx"
    READ_WRITE_EXECUTE_ABLE = "rwx"
    PERM_OPTIONS = [READ_ABLE, WRITE_ABLE, EXECUTE_ABLE, READ_WRITE_ABLE, READ_EXECUTE_ABLE, READ_WRITE_EXECUTE_ABLE]

    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FILE_VALID_PATTERN = r"^[a-zA-Z0-9_.:/-]+$"
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    JSON_SUFFIX = ".json"
    PT_SUFFIX = ".pt"
    BIN_SUFFIX = ".bin"
    CSV_SUFFIX = ".csv"
    XLSX_SUFFIX = ".xlsx"
    YAML_SUFFIX = ".yaml"
    IR_SUFFIX = ".ir"
    ZIP_SUFFIX = ".zip"
    SHELL_SUFFIX = ".sh"
    LOG_SUFFIX = ".log"
    ONNX_SUFFIX = '.onnx'
    OM_SUFFIX = '.om'
    MAX_PKL_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_NUMPY_SIZE = 10737418240  # 10 * 1024 * 1024 * 1024
    MAX_JSON_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_PT_SIZE = 10737418240  # 10 * 1024 * 1024 * 1024
    MAX_BIN_SIZE = 10737418240  # 10 * 1024 * 1024 * 1024
    MAX_CSV_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_XLSX_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_YAML_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_IR_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_ZIP_SIZE = 10737418240  # 10 * 1024 * 1024 * 1024
    MAX_FILE_IN_ZIP_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_FILE_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    COMMON_FILE_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_LOG_SIZE = 10737418240  # 1 * 1024 * 1024 * 1024
    MAX_COMMON_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB
    MAX_ONNX_SIZE = 10737418240  # 1 * 1024 * 1024 * 1024
    MAX_OM_SIZE = 10737418240  # 10 * 1024 * 1024 * 1024
    DIR = "dir"
    FILE = "file"
    DATA_DIR_AUTHORITY = 0o750
    DATA_FILE_AUTHORITY = 0o640
    FILE_SIZE_DICT = {
        PKL_SUFFIX: MAX_PKL_SIZE,
        NUMPY_SUFFIX: MAX_NUMPY_SIZE,
        JSON_SUFFIX: MAX_JSON_SIZE,
        PT_SUFFIX: MAX_PT_SIZE,
        BIN_SUFFIX: MAX_BIN_SIZE,
        CSV_SUFFIX: MAX_CSV_SIZE,
        XLSX_SUFFIX: MAX_XLSX_SIZE,
        YAML_SUFFIX: MAX_YAML_SIZE,
        IR_SUFFIX: MAX_IR_SIZE,
        ZIP_SUFFIX: MAX_ZIP_SIZE,
        LOG_SUFFIX: MAX_LOG_SIZE,
        ONNX_SUFFIX: MAX_ONNX_SIZE,
        OM_SUFFIX: MAX_OM_SIZE
    }
    CSV_BLACK_LIST = r'^[＋－＝％＠\+\-=%@]|;[＋－＝％＠\+\-=%@]'
