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

import json
from typing import List

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.log import logger
from msprobe.core.compare.indicator_analysis.utils import CompareMode, ResultLevel, divide_result_df, IgnoreInfo
from msprobe.core.compare.indicator_analysis.algorithm import BaseAlgorithm, TENSOR_CHECKERS, STATISTICS_CHECKERS, \
    MD5_CHECKERS
from msprobe.core.compare.indicator_analysis.api_data import ApiData


class ApiIndicatorCalculator:
    def __init__(self, mode):
        self.mode = mode
        self.all_ignore_list = ['empty', 'empty_like', 'numpy', 'to', '__setitem__', 'empty_with_format',
                                'new_empty_strided', 'new_empty', 'empty_strided']
        self.input_ignore_list = ['_reduce_scatter_base', '_all_gather_base', 'all_to_all_single', 'batch_isend_irecv']
        self.algorithms: List[BaseAlgorithm] = []
        self._add_algorithm()

    @staticmethod
    def get_api_indicator_and_msg(api_data: ApiData):
        """
        基于all_data_lists（api或模块的所有参数的数据）得到一个api或模块的指标和异常信息

        indicator取所有参数最差的（error）
        err msg取所有参数汇总

        Return:
            精度比对指标（pass/warning/error）
        """
        all_data_lists = api_data.input_data + api_data.output_data
        final_indicator = ResultLevel.PASS
        for data_list in all_data_lists:
            indicator = api_data.get_data_by_header(CompareConst.RESULT, data_list)
            if isinstance(indicator, ResultLevel):
                api_data.set_data_by_header(CompareConst.RESULT, data_list, indicator.value)
                if indicator > final_indicator:
                    final_indicator = indicator
            err_msg = api_data.get_data_by_header(CompareConst.ERROR_MESSAGE, data_list)
            if isinstance(err_msg, list):
                api_data.set_data_by_header(CompareConst.ERROR_MESSAGE, data_list, json.dumps(err_msg))

        return final_indicator.value

    def get_api_ignore_info(self, api_data: ApiData):
        """
        api是否需要忽略判断规则的情况
        """
        if not api_data.input_data:
            return IgnoreInfo.NO_IGNORE
        npu_param_name = api_data.get_data_by_header(CompareConst.NPU_NAME, api_data.input_data[0])
        name_split = npu_param_name.split(Const.SEP)
        if len(name_split) < 2:
            return IgnoreInfo.NO_IGNORE
        if name_split[1] in self.all_ignore_list:
            return IgnoreInfo.ALL_IGNORE
        elif name_split[1] in self.input_ignore_list:
            return IgnoreInfo.INPUT_IGNORE

        return IgnoreInfo.NO_IGNORE

    def calculate(self, raw_data_list: List[List]):
        """
        计算入口
        """
        api_data = ApiData(self.mode, raw_data_list)

        ignore_info = self.get_api_ignore_info(api_data)

        self.execute_all(api_data, ignore_info)

        return self.get_api_indicator_and_msg(api_data)

    def add_algorithm(self, algorithm: BaseAlgorithm):
        if not isinstance(algorithm, BaseAlgorithm):
            msg = 'It must be an instance of a subclass of BaseAlgorithm.'
            logger.error(msg)
            raise TypeError(msg)
        self.algorithms.append(algorithm)

    def execute_all(self, api_data: ApiData, ignore_info: IgnoreInfo):
        for algorithm in self.algorithms:
            try:
                algorithm.run(api_data, ignore_info)
            except Exception as e:
                msg = f'Run algorithm failed.'
                logger.error(msg)
                raise RuntimeError(msg) from e

    def _add_algorithm(self):
        if self.mode == CompareMode.STATISTICS.value:
            for checker in STATISTICS_CHECKERS:
                self.add_algorithm(checker())
        elif self.mode == CompareMode.TENSOR.value:
            for checker in TENSOR_CHECKERS:
                self.add_algorithm(checker())
        elif self.mode == CompareMode.MD5.value:
            for checker in MD5_CHECKERS:
                self.add_algorithm(checker())


def calculate_excel_result_df(result_df, mode):
    """
    仅适用于excel比对场景，得到表格每行数据的精度比对指标（pass/warning/error）

    Args:
        result_df: DataFrame数据结构，即转换成excel前的表单结构
        mode: 比对模式，分为 tensor 模式、统计量模式和 md5 模式
    """
    result_dict = divide_result_df(result_df)
    calculator = ApiIndicatorCalculator(mode)
    calculated_result_lists = []
    for data_lists in result_dict.values():
        calculator.calculate(data_lists)
        calculated_result_lists.extend(data_lists)

    # 将列表中的列表元素转换为str，避免list转换为ndarray报错
    # 可以使用一行列表推导式实现，但会被codecheck拦截（推导式和生成器表达式仅用于简单的逻辑表达）
    result = []
    for sublist in calculated_result_lists:
        processed_sublist = []
        for item in sublist:
            processed_sublist.append(json.dumps(item)) if isinstance(item, list) else processed_sublist.append(item)
        result.append(processed_sublist)
    result_df[:] = result


def calculate_result(result, mode):
    """
    得到一个api或模块的指标和异常信息

    Args:
        result: List[List]数据结构，每个list元素代表api或模块参数的具体信息
        mode: 比对模式，分为 tensor 模式、统计量模式和 md5 模式

    Return:
        精度比对指标（pass/warning/error）
    """
    calculator = ApiIndicatorCalculator(mode)
    return calculator.calculate(result)
