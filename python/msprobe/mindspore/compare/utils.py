# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
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

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_npy, FileChecker, FileCheckConst
from msprobe.core.common.utils import detect_framework_by_dump_json, CompareException, check_op_str_pattern_valid
from msprobe.core.common.log import logger


def read_npy_data(dir_path, file_name):
    if not file_name:
        return None

    data_path = os.path.join(dir_path, file_name)
    path_checker = FileChecker(data_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE,
                               FileCheckConst.NUMPY_SUFFIX, False)
    data_path = path_checker.common_check()
    data_value = load_npy(data_path)
    return data_value


def check_cross_framework(bench_json_path):
    framework = detect_framework_by_dump_json(bench_json_path)
    return framework == Const.PT_FRAMEWORK


def check_name_map_dict(name_map_dict):
    if not isinstance(name_map_dict, dict):
        logger.error("'map_dict' should be a dict, please check!")
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)
    check_op_str_pattern_valid(str(name_map_dict))
