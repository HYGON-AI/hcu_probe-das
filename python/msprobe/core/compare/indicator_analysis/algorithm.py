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


from abc import ABC, abstractmethod
from msprobe.core.compare.indicator_analysis.api_data import ApiData
from msprobe.core.compare.indicator_analysis.utils import is_inf_or_nan, str2float, ResultLevel, IgnoreInfo, \
    get_data_list_by_ignore_info
from msprobe.core.common.const import CompareConst


class BaseAlgorithm(ABC):
    """比对算法基类"""

    @abstractmethod
    def run(self, api_data: ApiData, ignore_info: IgnoreInfo):
        """
        算法执行接口
        :param api_data: 结构化的 API 数据
        :param ignore_info: 当前 API 数据需要忽略的指标信息
        """
        pass


class InfNanErrChecker(BaseAlgorithm):
    """
    适用于真实数据模式、统计数据模式
    一个 API 或模块的 NPU 的最大值或最小值中存在 nan/inf/-inf 标记为 error
    但如果 bench 侧也有相同现象，则忽略
    """

    def __init__(self):
        self.result_level = ResultLevel.ERROR
        self.err_msg = f'{self.result_level.value}: There is nan/inf/-inf in the maximum or minimum value of NPU.'

    def run(self, api_data: ApiData, ignore_info: IgnoreInfo):
        data_lists = get_data_list_by_ignore_info(api_data, ignore_info)
        if not data_lists:
            return

        for data_list in data_lists:
            bench_max = api_data.get_data_by_header(CompareConst.BENCH_MAX, data_list)
            bench_min = api_data.get_data_by_header(CompareConst.BENCH_MIN, data_list)

            if is_inf_or_nan(bench_max) or is_inf_or_nan(bench_min):
                continue

            npu_max = api_data.get_data_by_header(CompareConst.NPU_MAX, data_list)
            npu_min = api_data.get_data_by_header(CompareConst.NPU_MIN, data_list)

            if is_inf_or_nan(npu_max) or is_inf_or_nan(npu_min):
                api_data.set_result(data_list, self.result_level)
                api_data.set_err_msg(data_list, self.err_msg)


class RelativeErrChecker(BaseAlgorithm):
    """
    适用于统计数据模式
    指标需要结合输入和输出共同计算得到
    一个 API 或模块的 input 的相对误差 < 0.1 且 output 的相对误差 > 0.5，默认选取norm relative err观测， 标记为 error
    """

    def __init__(self):
        self.in_threshold = 0.1
        self.out_threshold = 0.5
        self.result_level = ResultLevel.ERROR
        self.err_msg = (f'{self.result_level.value}: The {CompareConst.NORM_RELATIVE_ERR} of output '
                        f'is greater than {self.out_threshold}.')

    def run(self, api_data: ApiData, ignore_info: IgnoreInfo):
        if ignore_info in [IgnoreInfo.ALL_IGNORE, IgnoreInfo.INPUT_IGNORE]:
            return

        norm_relative_err_max = abs(api_data.get_min_or_max_value(CompareConst.NORM_RELATIVE_ERR, is_min=False))

        if norm_relative_err_max < self.in_threshold:
            for data_list in api_data.output_data:
                norm_relative_err = str2float(api_data.get_data_by_header(CompareConst.NORM_RELATIVE_ERR, data_list))
                if abs(norm_relative_err) > self.out_threshold:
                    api_data.set_result(data_list, self.result_level)
                    api_data.set_err_msg(data_list, self.err_msg)


class OneThousandthErrChecker(BaseAlgorithm):
    """
    适用于真实数据模式
    指标需要结合输入和输出共同计算得到
    一个 API 或模块的 One Thousandth Err Ratio 的 input/parameters > 0.9 同时 output < 0.6， 标记为 error
    """

    def __init__(self):
        self.input_threshold = 0.9
        self.output_threshold = 0.1
        self.result_level = ResultLevel.ERROR
        self.err_msg = (f'{self.result_level.value}: The input/parameters of '
                        f'One Thousandth Err Ratio > 0.9 while the output < 0.6.')

    def run(self, api_data: ApiData, ignore_info: IgnoreInfo):
        if ignore_info in [IgnoreInfo.ALL_IGNORE, IgnoreInfo.INPUT_IGNORE]:
            return
        if not api_data.output_data:
            return

        min_input_ratio = api_data.get_min_or_max_value(CompareConst.ONE_THOUSANDTH_ERR_RATIO)
        min_output_ratio = api_data.get_min_or_max_value(CompareConst.ONE_THOUSANDTH_ERR_RATIO, is_input=False)

        if min_input_ratio > self.input_threshold and min_output_ratio < self.output_threshold:
            api_data.set_result(api_data.output_data[0], self.result_level)
            api_data.set_err_msg(api_data.output_data[0], self.err_msg)


class RequiresGradErrChecker(BaseAlgorithm):
    """
    适用于真实数据模式、统计数据模式
    一个 API 或模块的 Requires_grad Consistent 为 False，标记为 error
    """

    def __init__(self):
        self.result_level = ResultLevel.ERROR
        self.err_msg = f'{self.result_level.value}: The Required_Grad of NPU and Bench are inconsistent'

    def run(self, api_data: ApiData, ignore_info: IgnoreInfo):
        data_lists = get_data_list_by_ignore_info(api_data, ignore_info)
        if not data_lists:
            return

        for data_list in data_lists:
            # not match
            bench_name = api_data.get_data_by_header(CompareConst.BENCH_NAME, data_list)
            if not bench_name or bench_name == CompareConst.N_A:
                continue

            npu_req_grad = api_data.get_data_by_header(CompareConst.NPU_REQ_GRAD, data_list)
            bench_req_grad = api_data.get_data_by_header(CompareConst.BENCH_REQ_GRAD, data_list)

            if not npu_req_grad or not bench_req_grad:
                continue

            if npu_req_grad != bench_req_grad:
                api_data.set_result(data_list, self.result_level)
                api_data.set_err_msg(data_list, self.err_msg)


class ParametersErrChecker(BaseAlgorithm):
    """
    适用于真实数据模式、统计数据模式
    一个 API 或模块的非 tensor 标量参数，NPU 和 Bench 不一致，标记为 error
    """

    def __init__(self):
        self.result_level = ResultLevel.ERROR
        self.err_msg = f'{self.result_level.value}: The scalar parameters of NPU and Bench are inconsistent.'

    def run(self, api_data: ApiData, ignore_info: IgnoreInfo):
        if ignore_info in [IgnoreInfo.ALL_IGNORE, IgnoreInfo.INPUT_IGNORE]:
            return

        for data_list in api_data.input_data:
            # not match
            bench_name = api_data.get_data_by_header(CompareConst.BENCH_NAME, data_list)
            if not bench_name or bench_name == CompareConst.N_A:
                continue

            npu_dtype = api_data.get_data_by_header(CompareConst.NPU_DTYPE, data_list)
            bench_dtype = api_data.get_data_by_header(CompareConst.BENCH_DTYPE, data_list)
            if not npu_dtype or not bench_dtype:
                continue
            # 非tensor标量的dtype一定包含'class'，例如int类型的dtype为<class int>
            if 'class' not in npu_dtype or 'class' not in bench_dtype:
                continue

            npu_shape = api_data.get_data_by_header(CompareConst.NPU_SHAPE, data_list)
            # 以shape是否为[]判断其是否为标量
            if str(npu_shape) != '[]':
                continue
            npu_max = api_data.get_data_by_header(CompareConst.NPU_MAX, data_list)
            bench_max = api_data.get_data_by_header(CompareConst.BENCH_MAX, data_list)
            if npu_max != bench_max:
                api_data.set_result(data_list, self.result_level)
                api_data.set_err_msg(data_list, self.err_msg)


class CRC32ErrChecker(BaseAlgorithm):
    """
    适用于MD5模式
    NPU 与标杆的 CRC-32 值不一致，标记为 error
    NPU 与标杆的参数未匹配上，标记为 warning
    """

    def __init__(self):
        self.err_level = ResultLevel.ERROR
        self.err_msg = f'{self.err_level.value}: The CRC-32 value of NPU differs from that of the bench.'
        self.warn_level = ResultLevel.WARNING
        self.warn_msg = f'{self.warn_level.value}: The parameter of NPU does not match the bench.'

    def run(self, api_data: ApiData, ignore_info: IgnoreInfo):
        data_lists = get_data_list_by_ignore_info(api_data, ignore_info)
        if not data_lists:
            return

        null_set = (CompareConst.N_A, CompareConst.NAN)
        for data_list in data_lists:
            npu_md5 = api_data.get_data_by_header(CompareConst.NPU_MD5, data_list)
            bench_md5 = api_data.get_data_by_header(CompareConst.BENCH_MD5, data_list)
            if npu_md5 != bench_md5:
                # 参数未匹配上
                if npu_md5 in null_set or bench_md5 in null_set:
                    api_data.set_result(data_list, self.warn_level)
                    api_data.set_err_msg(data_list, self.warn_msg)
                else:
                    api_data.set_result(data_list, self.err_level)
                    api_data.set_err_msg(data_list, self.err_msg)


class DTypeErrChecker(BaseAlgorithm):
    """
    适用于真实数据模式、统计数据模式
    一个 API 或模块的 dtype 不一致，标记为 error
    """

    def __init__(self):
        self.result_level = ResultLevel.ERROR
        self.err_msg = f'{self.result_level.value}: The dtype of NPU and Bench are inconsistent.'

    def run(self, api_data: ApiData, ignore_info: IgnoreInfo):
        data_lists = get_data_list_by_ignore_info(api_data, ignore_info)
        if not data_lists:
            return

        for data_list in data_lists:
            # not match
            bench_name = api_data.get_data_by_header(CompareConst.BENCH_NAME, data_list)
            if not bench_name or bench_name == CompareConst.N_A:
                continue

            npu_dtype = api_data.get_data_by_header(CompareConst.NPU_DTYPE, data_list)
            bench_dtype = api_data.get_data_by_header(CompareConst.BENCH_DTYPE, data_list)

            if not npu_dtype or not bench_dtype:
                continue

            if npu_dtype != bench_dtype:
                api_data.set_result(data_list, self.result_level)
                api_data.set_err_msg(data_list, self.err_msg)


class ShapeErrChecker(BaseAlgorithm):
    """
    适用于真实数据模式、统计数据模式
    一个 API 或模块的 shape 不一致，标记为 error
    """

    def __init__(self):
        self.result_level = ResultLevel.ERROR
        self.err_msg = f'{self.result_level.value}: The shape of NPU and Bench are inconsistent.'

    def run(self, api_data: ApiData, ignore_info: IgnoreInfo):
        data_lists = get_data_list_by_ignore_info(api_data, ignore_info)
        if not data_lists:
            return
        for data_list in data_lists:
            # not match
            bench_name = api_data.get_data_by_header(CompareConst.BENCH_NAME, data_list)
            if not bench_name or bench_name == CompareConst.N_A:
                continue

            npu_shape = api_data.get_data_by_header(CompareConst.NPU_SHAPE, data_list)
            bench_shape = api_data.get_data_by_header(CompareConst.BENCH_SHAPE, data_list)

            if npu_shape is None or bench_shape is None:
                continue

            if npu_shape != bench_shape:
                api_data.set_result(data_list, self.result_level)
                api_data.set_err_msg(data_list, self.err_msg)


class RelativeWarnChecker(BaseAlgorithm):
    """
    适用于统计数据模式
    指标需要结合输入和输出共同计算得到
    一个 API 或模块的 output 相对误差是 input 相对误差的10倍，标记为 warning，默认选取norm观测
    """

    def __init__(self):
        self.threshold = 10
        self.result_level = ResultLevel.WARNING
        self.err_msg = (f'{self.result_level.value}: The norm relative error of output '
                        f'is {self.threshold} times that of input.')

    def run(self, api_data: ApiData, ignore_info: IgnoreInfo):
        if ignore_info in [IgnoreInfo.ALL_IGNORE, IgnoreInfo.INPUT_IGNORE]:
            return

        if not api_data.output_data:
            return

        norm_relative_error_max_in = abs(api_data.get_min_or_max_value(CompareConst.NORM_RELATIVE_ERR, is_min=False))
        norm_relative_error_max_out = abs(
            api_data.get_min_or_max_value(CompareConst.NORM_RELATIVE_ERR, is_input=False, is_min=False))

        should_set = False

        if norm_relative_error_max_in == 0:
            if norm_relative_error_max_out > 0.1:
                should_set = True
        elif norm_relative_error_max_out / norm_relative_error_max_in > self.threshold:
            should_set = True

        if should_set:
            api_data.set_result(api_data.output_data[0], self.result_level)
            api_data.set_err_msg(api_data.output_data[0], self.err_msg)


class CosineWarnChecker(BaseAlgorithm):
    """
    适用于真实数据模式
    指标需要结合输入和输出共同计算得到
    一个 API 或模块的 Cosine 的 input/parameters > 0.9 且 input/parameters - output > 0.1
    """

    def __init__(self):
        self.input_threshold = 0.9
        self.output_threshold = 0.1
        self.result_level = ResultLevel.WARNING
        self.err_msg = (f'{self.result_level.value}: The input/parameters of Cosine > {self.input_threshold}, '
                        f'and input/parameters - output > {self.output_threshold}')

    def run(self, api_data: ApiData, ignore_info: IgnoreInfo):
        if ignore_info in [IgnoreInfo.ALL_IGNORE, IgnoreInfo.INPUT_IGNORE]:
            return

        if not api_data.output_data:
            return

        min_input_cosine = api_data.get_min_or_max_value(CompareConst.COSINE)
        min_output_cosine = api_data.get_min_or_max_value(CompareConst.COSINE, is_input=False)

        if min_input_cosine > self.input_threshold and min_input_cosine - min_output_cosine > self.output_threshold:
            api_data.set_result(api_data.output_data[0], self.result_level)
            api_data.set_err_msg(api_data.output_data[0], self.err_msg)


TENSOR_CHECKERS = [InfNanErrChecker, OneThousandthErrChecker, RequiresGradErrChecker, ParametersErrChecker,
                   DTypeErrChecker, ShapeErrChecker, CosineWarnChecker]
STATISTICS_CHECKERS = [InfNanErrChecker, RelativeErrChecker, RequiresGradErrChecker, ParametersErrChecker,
                       DTypeErrChecker, ShapeErrChecker, RelativeWarnChecker]
MD5_CHECKERS = [CRC32ErrChecker]
STATISTICS_CHECKERS_PARALLEL_MERGE = [InfNanErrChecker, RelativeErrChecker, RequiresGradErrChecker,
                                      ParametersErrChecker, DTypeErrChecker, RelativeWarnChecker]
