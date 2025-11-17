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

import torch

from msprobe.core.common.utils import logger, CompareException
from msprobe.core.common.file_utils import FileChecker, FileCheckConst
from msprobe.pytorch.common.utils import load_pt


def read_pt_data(dir_path, file_name):
    if not file_name:
        return None

    data_path = os.path.join(dir_path, file_name)
    path_checker = FileChecker(data_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE, FileCheckConst.PT_SUFFIX)
    data_path = path_checker.common_check()
    try:
        # detach because numpy can not process gradient information
        data_value = load_pt(data_path, to_cpu=True).detach()
    except RuntimeError as e:
        # 这里捕获 load_pt 中抛出的异常
        data_path_file_name = os.path.basename(data_path)
        logger.error(f"Failed to load the .pt file at {data_path_file_name}.")
        raise CompareException(CompareException.INVALID_FILE_ERROR) from e
    except AttributeError as e:
        # 这里捕获 detach 方法抛出的异常
        logger.error(f"Failed to detach the loaded tensor.")
        raise CompareException(CompareException.DETACH_ERROR) from e
    if data_value.dtype == torch.bfloat16:
        data_value = data_value.to(torch.float32)
    data_value = data_value.numpy()
    return data_value
