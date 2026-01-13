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

from typing import List

from msprobe.core.compare.indicator_analysis.utils import CompareMode, ResultLevel, str2float
from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException


class ApiData:
    header_index_mapping_cache = {}

    def __init__(self, mode, data_lists: List[List]):
        self.input_data = []
        self.output_data = []
        self.mode = mode
        self.data_lists = data_lists
        self.header = self._get_header()
        # 表头与索引映射，可以基于表头拿到索引，从input_data和output_data中获取对应数据
        self.header_index_mapping = self.get_header_index_mapping()
        self._init_data()

    def get_header_index_mapping(self):
        if self.mode in ApiData.header_index_mapping_cache:
            return ApiData.header_index_mapping_cache[self.mode]

        mapping = {item: index for index, item in enumerate(self.header)}
        ApiData.header_index_mapping_cache[self.mode] = mapping
        return mapping

    def get_data_by_header(self, header: str, data_list: List):
        """
        基于表头从data list获取数据
        """
        index = self.header_index_mapping.get(header)
        try:
            data = data_list[index]
        except Exception as e:
            logger.error(f'Unable to get data from the data list based on the header: {e}')
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR) from e
        return data

    def set_data_by_header(self, header: str, data_list: List, new_data):
        index = self.header_index_mapping.get(header)
        try:
            data_list[index] = new_data
        except Exception as e:
            logger.error(f'Unable to set data from the data list based on the header: {e}')
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR) from e

    def set_result(self, data_list: List, status: ResultLevel = ResultLevel.PASS):
        index = self.header_index_mapping.get(CompareConst.RESULT)
        try:
            current_status = data_list[index]
            if not isinstance(current_status, ResultLevel) or status > current_status:
                data_list[index] = status
        except Exception as e:
            logger.error(f'Unable to set status from the data list based on the header: {e}')
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR) from e

    def set_err_msg(self, data_list: List, msg: str = '', init_msg=False):
        index = self.header_index_mapping.get(CompareConst.ERROR_MESSAGE)
        try:
            if init_msg:
                data_list[index] = []
            else:
                current_msg = data_list[index]
                if not isinstance(current_msg, list):
                    current_msg = [current_msg] if current_msg else []
                    data_list[index] = current_msg
                if msg:
                    current_msg.append(msg)
        except Exception as e:
            logger.error(f'Unable to set err msg from the data list based on the header: {e}')
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR) from e

    def get_min_or_max_value(self, header, is_input=True, is_min=True):
        """
        获取多个输入或输出参数中的最小或最大指标
        """
        default_value = 1.0 if is_min else 0.0
        data_lists = self.input_data if is_input else self.output_data
        for data_list in data_lists:
            value = self.get_data_by_header(header, data_list)

            if value is None or value in [CompareConst.NAN, CompareConst.N_A]:
                continue

            if isinstance(value, str) and value.endswith('%'):
                value = str2float(value)

            if isinstance(value, str):
                continue

            default_value = min(default_value, value) if is_min else max(default_value, value)
        return default_value

    def _get_header(self):
        if self.mode == CompareMode.STATISTICS.value:
            return CompareConst.SUMMARY_COMPARE_RESULT_HEADER
        elif self.mode == CompareMode.TENSOR.value:
            return CompareConst.COMPARE_RESULT_HEADER
        elif self.mode == CompareMode.MD5.value:
            return CompareConst.MD5_COMPARE_RESULT_HEADER
        else:
            logger.error(f'The parameter "mode" error, '
                         f'expected {CompareMode.STATISTICS.value}/{CompareMode.TENSOR.value}/{CompareMode.MD5.value}, '
                         f'actually {self.mode}.')
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

    def _init_data(self):
        for data_list in self.data_lists:
            if not data_list:
                continue

            if len(data_list) < len(self.header):
                data_list = data_list + [''] * (len(self.header) - len(data_list))

            # 初始化result为pass，error message为list
            self.set_result(data_list)
            self.set_err_msg(data_list, init_msg=True)

            if f'{Const.SEP}{Const.OUTPUT}{Const.SEP}' in data_list[0]:
                self.output_data.append(data_list)
            else:
                self.input_data.append(data_list)
