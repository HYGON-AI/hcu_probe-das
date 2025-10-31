# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import abc

import numpy as np

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException, format_value


def handle_inf_nan(n_value, b_value):
    def convert_to_float(value):
        try:
            if isinstance(value, np.ndarray):
                return value.astype(float)
            else:
                return float(value)
        except ValueError as e:
            logger.error('\n'.join(e.args))
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e

    n_value_convert, b_value_convert = convert_to_float(n_value), convert_to_float(b_value)
    """处理inf和nan的数据"""
    n_inf = np.isinf(n_value_convert)
    b_inf = np.isinf(b_value_convert)
    n_nan = np.isnan(n_value_convert)
    b_nan = np.isnan(b_value_convert)
    n_invalid = np.any(n_inf) or np.any(n_nan)
    b_invalid = np.any(b_inf) or np.any(b_nan)
    if n_invalid or b_invalid:
        if np.array_equal(n_inf, b_inf) and np.array_equal(n_nan, b_nan):
            n_value[n_inf] = 0
            b_value[b_inf] = 0
            n_value[n_nan] = 0
            b_value[b_nan] = 0
        else:
            return CompareConst.NAN, CompareConst.NAN
    return n_value, b_value


def get_error_flag_and_msg(n_value, b_value, error_flag=False, error_file=None):
    """判断数据是否有异常并返回异常的n_value, b_value，同时返回error_flag和error_msg"""
    err_msg = ""
    if error_flag:
        err_msg = f"Dump file: {error_file} not found or read failed."
        return CompareConst.READ_NONE, CompareConst.READ_NONE, error_flag, err_msg

    if n_value.size == 0:  # 判断读取到的数据是否为空
        err_msg = "This is empty data, can not compare."
        error_flag = True
        return CompareConst.NONE, CompareConst.NONE, error_flag, err_msg
    if not n_value.shape:  # 判断数据是否为0维张量
        err_msg = (f"This is type of 0-d tensor, can not calculate '{CompareConst.COSINE}', '{CompareConst.EUC_DIST}', "
                   f"'{CompareConst.ONE_THOUSANDTH_ERR_RATIO}' and '{CompareConst.FIVE_THOUSANDTHS_ERR_RATIO}'. ")
        error_flag = False  # 0-d tensor 最大绝对误差、最大相对误差仍然支持计算，因此error_flag设置为False，不做统一处理
        return n_value, b_value, error_flag, err_msg
    if n_value.shape != b_value.shape:  # 判断NPU和bench的数据结构是否一致
        err_msg = "Shape of NPU and bench tensor do not match. Skipped."
        error_flag = True
        return CompareConst.SHAPE_UNMATCH, CompareConst.SHAPE_UNMATCH, error_flag, err_msg

    try:
        n_value, b_value = handle_inf_nan(n_value, b_value)  # 判断是否有nan/inf数据
    except CompareException:
        logger.error('Numpy data is unreadable, please check!')
        err_msg = "Data is unreadable."
        error_flag = True
        return CompareConst.UNREADABLE, CompareConst.UNREADABLE, error_flag, err_msg
    if n_value is CompareConst.NAN or b_value is CompareConst.NAN:
        err_msg = "The position of inf or nan in NPU and bench Tensor do not match."
        error_flag = True
        return CompareConst.NAN, CompareConst.NAN, error_flag, err_msg

    if n_value.dtype != b_value.dtype:  # 判断数据的dtype是否一致
        err_msg = "Dtype of NPU and bench tensor do not match."
        error_flag = False
        return n_value, b_value, error_flag, err_msg

    return n_value, b_value, error_flag, err_msg


def reshape_value(n_value, b_value):
    """返回reshape后的数据"""
    if not n_value.shape:  # 判断数据是否为0维tensor， 如果0维tensor，不会转成1维tensor，直接返回
        if n_value.dtype == bool:
            n_value = n_value.astype(float)
            b_value = b_value.astype(float)
        return n_value, b_value

    n_value = n_value.reshape(-1).astype(float)  # 32转64为了防止某些数转dataframe时出现误差
    b_value = b_value.reshape(-1).astype(float)
    return n_value, b_value


def npy_data_check(n_value, b_value):
    error_message = ""
    if not isinstance(n_value, np.ndarray) or not isinstance(b_value, np.ndarray):
        error_message += "Dump file is not ndarray.\n"

    # 检查 n_value 和 b_value 是否为空
    if not error_message and (n_value.size == 0 or b_value.size == 0):
        error_message += "This is empty data, can not compare.\n"

    if not error_message:
        if not n_value.shape or not b_value.shape:
            error_message += "This is type of scalar data, can not compare.\n"
        if n_value.shape != b_value.shape:
            error_message += "Shape of NPU and bench Tensor do not match.\n"
        if n_value.dtype != b_value.dtype:
            error_message += "Dtype of NPU and bench Tensor do not match. Skipped.\n"

    if not error_message:
        try:
            n_value, b_value = handle_inf_nan(n_value, b_value)  # 判断是否有nan/inf数据
        except CompareException:
            logger.error('Numpy data is unreadable, please check!')
            return True, 'Numpy data is unreadable, please check!'
        # handle_inf_nan 会返回'Nan'或ndarray类型，使用类型判断是否存在无法处理的nan/inf数据
        if not isinstance(n_value, np.ndarray) or not isinstance(b_value, np.ndarray):
            error_message += "The position of inf or nan in NPU and bench Tensor do not match.\n"
    if error_message == "":
        error_flag = False
    else:
        error_flag = True
    return error_flag, error_message


def statistics_data_check(result_dict):
    error_message = ""

    if result_dict.get(CompareConst.NPU_NAME) is None or result_dict.get(CompareConst.BENCH_NAME) is None:
        error_message += "Dump file not found.\n"

    if not result_dict.get(CompareConst.NPU_SHAPE) or not result_dict.get(CompareConst.BENCH_SHAPE):
        error_message += "This is type of scalar data, can not compare.\n"
    elif result_dict.get(CompareConst.NPU_SHAPE) != result_dict.get(CompareConst.BENCH_SHAPE):
        error_message += "Tensor shapes do not match.\n"

    if result_dict.get(CompareConst.NPU_DTYPE) != result_dict.get(CompareConst.BENCH_DTYPE):
        error_message += "Dtype of NPU and bench Tensor do not match. Skipped.\n"

    if error_message == "":
        error_flag = False
    else:
        error_flag = True
    return error_flag, error_message


class TensorComparisonBasic(abc.ABC):
    """NPU和bench中npy数据的比较模板"""

    @abc.abstractmethod
    def apply(self, n_value, b_value, relative_err, err_msg):
        raise NotImplementedError


def get_relative_err(n_value, b_value):
    """计算相对误差"""
    with np.errstate(divide='ignore', invalid='ignore'):
        if n_value.dtype not in CompareConst.FLOAT_TYPE:
            n_value = n_value.astype(float)
        if b_value.dtype not in CompareConst.FLOAT_TYPE:
            b_value = b_value.astype(float)

        n_value_copy = n_value.copy()
        b_value_copy = b_value.copy()
        zero_mask = (b_value_copy == 0)
        b_value_copy[zero_mask] += Const.FLOAT_EPSILON
        n_value_copy[zero_mask] += Const.FLOAT_EPSILON
        relative_err = np.divide((n_value_copy - b_value_copy), b_value_copy)
    return np.abs(relative_err)


class GetCosineSimilarity(TensorComparisonBasic):
    """计算cosine相似度"""

    @staticmethod
    def correct_data(result):
        if result == CompareConst.NAN:
            return result
        if float(result) > CompareConst.COSINE_THRESHOLD:
            return round(float(result), 6)
        return result

    def apply(self, n_value, b_value, relative_err, err_msg):
        if "This is type of 0-d tensor" in err_msg:
            return CompareConst.UNSUPPORTED, err_msg

        with np.errstate(divide="ignore", invalid="ignore"):
            if len(n_value) == 1:
                return CompareConst.UNSUPPORTED, "This is a 1-d tensor of length 1."
            num = n_value.dot(b_value)
            a_norm = np.linalg.norm(n_value)
            b_norm = np.linalg.norm(b_value)

            if a_norm <= Const.FLOAT_EPSILON and b_norm <= Const.FLOAT_EPSILON:
                return 1.0, ""
            if a_norm <= Const.FLOAT_EPSILON:
                return CompareConst.NAN, "Cannot compare by Cosine Similarity, All the data is Zero in npu dump data."
            if b_norm <= Const.FLOAT_EPSILON:
                return CompareConst.NAN, "Cannot compare by Cosine Similarity, All the data is Zero in Bench dump data."

            cos = num / (a_norm * b_norm)
            if np.isnan(cos):
                return CompareConst.NAN, "Cannot compare by Cosine Similarity, the dump data has NaN."
            result = format_value(cos)
            result = self.correct_data(result)
        return result, ""


class GetEuclideanDistance(TensorComparisonBasic):
    """计算欧式距离"""

    def apply(self, n_value, b_value, relative_err, err_msg):
        if "This is type of 0-d tensor" in err_msg:
            return CompareConst.UNSUPPORTED, err_msg

        distance = np.linalg.norm(n_value - b_value, ord=2)

        return distance, ""


class GetMaxAbsErr(TensorComparisonBasic):
    """计算最大绝对误差"""

    def apply(self, n_value, b_value, relative_err, err_msg):
        temp_res = n_value - b_value
        max_value = np.max(np.abs(temp_res))
        if np.isnan(max_value):
            msg = "Cannot compare by MaxAbsError, the data contains nan/inf/-inf in dump data."
            return CompareConst.NAN, msg
        return format_value(max_value), ""


class GetMaxRelativeErr(TensorComparisonBasic):
    """计算最大相对误差"""

    def apply(self, n_value, b_value, relative_err, err_msg):
        max_relative_err = np.max(np.abs(relative_err))
        if np.isnan(max_relative_err):
            msg = "Cannot compare by MaxRelativeError, the data contains nan/inf/-inf in dump data."
            return CompareConst.NAN, msg
        return format_value(max_relative_err), ""


class GetErrRatio(TensorComparisonBasic):
    """计算相对误差小于指定阈值(千分之一、千分之五)的比例"""

    def __init__(self, threshold):
        self.threshold = threshold

    def apply(self, n_value, b_value, relative_err, err_msg):
        if "This is type of 0-d tensor" in err_msg:
            return CompareConst.UNSUPPORTED, err_msg

        if not np.size(relative_err):
            return CompareConst.NAN, ""

        ratio = np.sum(relative_err < self.threshold) / np.size(relative_err)
        return format_value(ratio), ""


class CompareOps:
    compare_ops = {
        "cosine_similarity": GetCosineSimilarity(),
        "euclidean_distance": GetEuclideanDistance(),
        "max_abs_error": GetMaxAbsErr(),
        "max_relative_error": GetMaxRelativeErr(),
        "one_thousand_err_ratio": GetErrRatio(CompareConst.THOUSAND_RATIO_THRESHOLD),
        "five_thousand_err_ratio": GetErrRatio(CompareConst.FIVE_THOUSAND_RATIO_THRESHOLD)
    }


def error_value_process(n_value):
    if n_value in [CompareConst.READ_NONE, CompareConst.UNREADABLE, CompareConst.NONE,
                   CompareConst.NO_REAL_DATA, CompareConst.API_UNMATCH]:
        return CompareConst.UNSUPPORTED, ""
    if n_value == CompareConst.SHAPE_UNMATCH:
        return CompareConst.SHAPE_UNMATCH, ""
    if n_value == CompareConst.NAN:
        return CompareConst.N_A, ""
    return CompareConst.N_A, ""


def compare_ops_apply(n_value, b_value, error_flag, err_msg):
    result_list = []
    if error_flag:
        result, msg = error_value_process(n_value)
        result_list = [result] * len(CompareOps.compare_ops)
        err_msg += msg
        return result_list, err_msg

    relative_err = get_relative_err(n_value, b_value)
    n_value, b_value = reshape_value(n_value, b_value)

    for op in CompareOps.compare_ops.values():
        result, msg = op.apply(n_value, b_value, relative_err, err_msg)
        result_list.append(result)
        err_msg += msg
    return result_list, err_msg
