# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

import re
from msprobe.core.grad_probe.constant import GradConst
from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import write_csv, check_path_before_create, change_mode
from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.utils import is_int
import matplotlib.pyplot as plt


def data_in_list_target(data, lst):
    return not lst or len(lst) == 0 or data in lst


def check_numeral_list_ascend(lst):
    if any(not isinstance(item, (int, float)) for item in lst):
        raise Exception("The input list should only contain numbers")
    if lst != sorted(lst):
        raise Exception("The input list should be ascending")


def check_param(param_name):
    if not re.match(GradConst.PARAM_VALID_PATTERN, param_name):
        raise RuntimeError("The parameter name contains special characters.")


def check_str(string, variable_name):
    if not isinstance(string, str):
        raise ValueError(f'The variable: "{variable_name}" is not a string.')


def check_bounds_element(bound):
    return GradConst.BOUNDS_MINIMUM <= bound <= GradConst.BOUNDS_MAXIMUM


def check_param_element(param):
    if not re.match(GradConst.PARAM_VALID_PATTERN, param):
        return False
    else:
        return True


def check_bounds(bounds):
    if not isinstance(bounds, list):
        raise Exception(f"bounds must be a list")
    prev = GradConst.BOUNDS_MINIMUM - 1
    for element in bounds:
        if not is_int(element) and not isinstance(element, float):
            raise Exception("bounds element is not int or float")
        if not check_bounds_element(element):
            raise Exception("bounds element is out of int64 range")
        if prev >= element:
            raise Exception("bounds list is not ascending")
        prev = element


class ListCache(list):
    threshold = 1000

    def __init__(self, *args):
        super().__init__(*args)
        self._output_file = None

    def __del__(self):
        self.flush()

    def flush(self):
        if len(self) == 0:
            return
        if not self._output_file:
            logger.warning("dumpfile path is not set.")
        write_csv(self, self._output_file)
        logger.info(f"write {len(self)} items to {self._output_file}.")
        self.clear()

    def append(self, data):
        list.append(self, data)
        if len(self) >= ListCache.threshold:
            self.flush()

    def set_output_file(self, output_file):
        self._output_file = output_file


def plt_savefig(fig_save_path):
    check_path_before_create(fig_save_path)
    try:
        plt.savefig(fig_save_path)
    except Exception as e:
        raise RuntimeError(f"save plt figure {fig_save_path} failed") from e
    change_mode(fig_save_path, FileCheckConst.DATA_FILE_AUTHORITY)
