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

import math
from enum import Enum
from dataclasses import dataclass

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException


def str2float(percentage_str):
    """
    百分比字符串转换转换为浮点型
    Args:
        percentage_str: '0.00%', '23.4%'
    Returns: float 0.00, 0.234
    """
    try:
        percentage_str = percentage_str.strip('%')
        return float(percentage_str) / 100
    except (ValueError, AttributeError):
        return 0


def is_inf_or_nan(value):
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    elif isinstance(value, str):
        value = value.strip().lower()
        if value in ['inf', '-inf', 'nan']:
            return True
    return False


def divide_result_df(result_df):
    """
    result_df就是输出csv前的数据，按api粒度划分每行数据

    return: dict
        {
            api_name1: [[data1, data2, ...], [data1, data2, ...]],
            api_name2: [[data1, data2, ...], [data1, data2, ...]],
            ...
        }
    """
    result_lists = result_df.values.tolist()
    try:
        api_name_index = result_df.columns.get_loc(Const.API_ORIGIN_NAME)
    except KeyError as e:
        logger.error_log_with_exp(f"An error occurred during get index from dataframe: {e}",
                                  MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
    result_dict = {}
    for result_list in result_lists:
        if not result_list:
            continue
        api_name = result_list[api_name_index]
        if api_name in result_dict:
            result_dict.get(api_name).append(result_list)
        else:
            result_dict[api_name] = [result_list]
    return result_dict


class ResultLevel(Enum):
    PASS = CompareConst.PASS
    ERROR = CompareConst.ERROR
    WARNING = CompareConst.WARNING

    def __lt__(self, other):
        """支持比较，用于汇总结果 ERROR > WARNING > PASS"""
        order = [self.PASS, self.WARNING, self.ERROR]
        return order.index(self) < order.index(other)


class CompareMode(Enum):
    TENSOR = Const.ALL
    STATISTICS = Const.SUMMARY
    MD5 = Const.MD5


@dataclass
class AlgorithmResult:
    """算法执行结果"""
    status: ResultLevel = ResultLevel.PASS
    message: list = None


class IgnoreInfo(Enum):
    ALL_IGNORE = 'all_ignore'
    INPUT_IGNORE = 'input_ignore'
    NO_IGNORE = 'no_ignore'


def get_data_list_by_ignore_info(api_data, ignore_info: IgnoreInfo):
    if ignore_info == IgnoreInfo.ALL_IGNORE:
        data_lists = []
    elif ignore_info == IgnoreInfo.INPUT_IGNORE:
        data_lists = api_data.output_data
    else:
        data_lists = api_data.input_data + api_data.output_data
    return data_lists
