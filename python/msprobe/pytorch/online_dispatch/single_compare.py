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

import logging
from collections import namedtuple
from functools import wraps

import torch
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.online_dispatch.utils import check_idx_valid
from prettytable import PrettyTable


def func_log_wrapper():
    def _out_wrapper(func):
        @wraps(func)
        def _in_wrapper(*kargs, **kwargs):
            logger.info(f"start to run: {func.__name__}")
            x = func(*kargs, **kwargs)
            logger.info(f"end to run: {func.__name__}")
            return x

        return _in_wrapper

    return _out_wrapper


class SingleBenchmarkCompareStandard:
    def __init__(self, high_precision=True):
        self.high_precision = high_precision
        self.small_value = 1.0
        self.error_thd = {torch.float16: [2 ** -11, 2 ** -7],
                          torch.bfloat16: [2 ** -8, 2 ** -6],
                          torch.float32: [2 ** -14, 2 ** -11],
                          torch.float64: [2 ** -14, 2 ** -11]}
        self.eb_thd = {torch.float16: 2 ** -10,
                       torch.bfloat16: 2 ** -7,
                       torch.float32: 2 ** -14,
                       torch.float64: 2 ** -14}

    def get_error_thd(self, dtype):
        if dtype in self.error_thd.keys():
            if dtype == torch.float64:
                logging.warning("the output data of fp64 uses the same standard as fp32.")
            return self.error_thd.get(dtype)[0] if self.high_precision else self.error_thd.get(dtype)[1]
        logging.error(
            "Single benchmark compare only supports floating point "
            "in fp16, bf16, fp32. "
        )
        return None

    def get_eb_thd(self, dtype):
        if dtype in self.eb_thd.keys():
            return self.eb_thd.get(dtype)
        return None


class SingleBenchmarkAccuracyResult:
    def __init__(
        self,
        result=True,
        error_balance=None,
        max_abs_diff=None,
        max_abs_idx=None,
        max_rel_diff=None,
        max_rel_idx=None
    ):
        self.result = result
        self.error_balance = error_balance
        self.max_abs_diff = max_abs_diff
        self.max_abs_idx = max_abs_idx
        self.max_rel_diff = max_rel_diff
        self.max_rel_idx = max_rel_idx

    def get_result(self, eb_thd, error_thd):
        if (
            self.error_balance > eb_thd
            or self.max_abs_diff > error_thd
            or self.max_rel_diff > error_thd
        ):
            self.result = False
        else:
            self.result = True


class SingleBenchmarkAccuracyCompare:
    @classmethod
    @func_log_wrapper()
    def check_output_size(cls, npu_out, bench_out):
        acc_result = None
        if npu_out.numel() == 0 and bench_out.numel() == 0:
            info = (
                "The npu_output is [], and it is same as benchmark_output, "
                "the result of data_compare is Pass"
            )
            logging.debug(info)
            acc_result = SingleBenchmarkAccuracyResult(result=True)

        if npu_out.size() != bench_out.size():
            error_info = (
                f"the size of npu output[{npu_out.size()}] and"
                f"benchmark[{bench_out.size()}] is not equal"
            )

            logging.error(error_info)
            acc_result = SingleBenchmarkAccuracyResult(result=False)
        return acc_result

    @classmethod
    @func_log_wrapper()
    def check_output_invalid_value(cls, output):
        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()
        return has_nan or has_inf

    @classmethod
    @func_log_wrapper()
    def precision_compare_for_case(cls, npu_out, bench_out, benchmark_standard: SingleBenchmarkCompareStandard):
        error_thd = None
        eb_thd = None
        acc_result = cls.check_output_size(npu_out, bench_out)
        CompareResultInfo = namedtuple("CompareResultInfo",
                                       ['accuracy_result', 'error_threshold', 'eb_threshold', 'failed_information'])

        if acc_result:
            failed_info = "比对数据的shape不一致"
            return CompareResultInfo(acc_result, error_thd, eb_thd, failed_info)

        if cls.check_output_invalid_value(bench_out):
            logging.info("The benchmark result contains nan/inf value. ")
            failed_info = "标杆结果存在nan值或inf值, 依照单标杆标准该用例通过"
            acc_result = SingleBenchmarkAccuracyResult(result=True)
            return CompareResultInfo(acc_result, error_thd, eb_thd, failed_info)

        if cls.check_output_invalid_value(npu_out):
            logging.info("The NPU result contains nan/inf value. ")
            failed_info = "NPU结果存在nan值或inf值, 依照单标杆标准该用例不通过"
            acc_result = SingleBenchmarkAccuracyResult(result=False)
            return CompareResultInfo(acc_result, error_thd, eb_thd, failed_info)

        data_type = npu_out.dtype
        if data_type not in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
            acc_result = cls.compute_binary_diff(npu_out, bench_out)
        else:
            error_thd = benchmark_standard.get_error_thd(data_type)
            eb_thd = benchmark_standard.get_eb_thd(data_type)
            if error_thd is None:
                logging.error(
                    "single benchmark not support the comparison of %s", str(data_type)
                )
                acc_result = SingleBenchmarkAccuracyResult(result=False)
            else:
                if npu_out.dtype in [torch.float16, torch.bfloat16] and bench_out.dtype in [torch.float32]:
                    npu_out = npu_out.to(torch.float32)
                error_balance = cls.compute_error_balance(npu_out, bench_out, benchmark_standard)
                max_abs_diff, max_abs_idx = cls.compute_abs_diff(npu_out, bench_out, error_thd, benchmark_standard)
                max_rel_diff, max_rel_idx = cls.compute_rel_diff(npu_out, bench_out, error_thd, benchmark_standard)
                acc_result = SingleBenchmarkAccuracyResult(
                    error_balance=error_balance,
                    max_abs_diff=max_abs_diff,
                    max_abs_idx=max_abs_idx,
                    max_rel_diff=max_rel_diff,
                    max_rel_idx=max_rel_idx
                )
                acc_result.get_result(eb_thd, error_thd)
        return CompareResultInfo(acc_result, error_thd, eb_thd, None)

    @classmethod
    @func_log_wrapper()
    def compute_binary_diff(cls, npu_out, bench_out):
        result = torch.equal(npu_out, bench_out)
        if result:
            logger.info("二进制精度比对通过, 无需单标杆比对法验证")
        return SingleBenchmarkAccuracyResult(result=result, max_abs_diff=0, max_rel_diff=0, error_balance=0)

    @classmethod
    @func_log_wrapper()
    def compute_error_balance(cls, npu_out, bench_out, benchmark_standard: SingleBenchmarkCompareStandard):
        ones = torch.ones_like(npu_out)
        zeros = torch.zeros_like(npu_out)
        abs_mask_idx = torch.where(torch.abs(bench_out) < benchmark_standard.small_value, ones, zeros)
        abs_mask_idx = abs_mask_idx.type(torch.bool)
        diff_value = torch.subtract(npu_out, bench_out)
        diff_value_rel = diff_value / (torch.abs(bench_out) + torch.finfo(torch.float).eps)
        rel_and_abs = torch.where(abs_mask_idx, diff_value, diff_value_rel)
        eb_float = float(torch.mean(rel_and_abs))
        return eb_float

    @classmethod
    @func_log_wrapper()
    def compute_abs_diff(cls, npu_out, bench_out, error_thd, benchmark_standard: SingleBenchmarkCompareStandard):
        max_abs_diff = 0
        max_abs_idx = None

        ones = torch.ones_like(npu_out)
        zeros = torch.zeros_like(npu_out)
        diff_value = torch.subtract(npu_out, bench_out)
        diff_abs = torch.abs(diff_value)
        abs_mask_idx = torch.where(torch.abs(bench_out) >= benchmark_standard.small_value, ones, zeros)
        abs_err_idx = torch.where(diff_abs > error_thd, ones, zeros)
        abs_err_idx = abs_err_idx * abs_mask_idx
        abs_err = diff_abs[torch.where(abs_err_idx == 1)]

        if len(abs_err) > 0:
            err_for_max = torch.where(abs_err_idx == 1, diff_abs, zeros)
            logging.debug("err_for_max for abs %s", err_for_max)
            max_abs_idx = torch.argmax(err_for_max)
            if check_idx_valid(diff_abs, max_abs_idx):
                max_abs_diff = diff_abs[max_abs_idx]
        elif torch.sum(abs_mask_idx) > 0:
            err_for_max = torch.where(abs_mask_idx == 1, diff_abs, zeros)
            logging.debug("error_for_max for abs %s", err_for_max)
            max_abs_idx = torch.argmax(err_for_max)
            if err_for_max.max() != 0 and check_idx_valid(diff_abs, max_abs_idx):
                max_abs_diff = diff_abs[max_abs_idx]
        return (float(max_abs_diff), int(max_abs_idx) if torch.is_tensor(max_abs_idx) else max_abs_idx)

    @classmethod
    @func_log_wrapper()
    def compute_rel_diff(cls, npu_out, bench_out, error_thd, benchmark_standard: SingleBenchmarkCompareStandard):
        max_rel_diff = 0
        max_rel_idx = None

        ones = torch.ones_like(npu_out)
        zeros = torch.zeros_like(npu_out)
        diff_value = torch.subtract(npu_out, bench_out)
        diff_abs = torch.abs(diff_value)

        rel_mask_idx = torch.where(torch.abs(bench_out) >= benchmark_standard.small_value, ones, zeros)
        rel_err = diff_abs / (torch.abs(bench_out) + torch.finfo(torch.float).eps)
        diff_rel = rel_err
        rel_err_idx = torch.where(rel_err > error_thd, ones, zeros)
        rel_err_idx = rel_err_idx * rel_mask_idx
        rel_err = rel_err[torch.where(rel_err_idx == 1)]
        if len(rel_err) > 0:
            err_for_max = torch.where(rel_err_idx == 1, diff_rel, zeros)
            logging.debug("error_for_max for rel %s", err_for_max)
            max_rel_idx = torch.argmax(err_for_max)
            if check_idx_valid(diff_rel, max_rel_idx):
                max_rel_diff = diff_rel[max_rel_idx]
        elif torch.sum(rel_mask_idx > 0):
            err_for_max = torch.where(rel_mask_idx == 1, diff_rel, zeros)
            logging.debug("err_for_max for rel %s", err_for_max)
            max_rel_idx = torch.argmax(err_for_max)
            if torch.sum(err_for_max) != 0 and check_idx_valid(diff_rel, max_rel_idx):
                max_rel_diff = diff_rel[max_rel_idx]
        return (float(max_rel_diff), int(max_rel_idx) if torch.is_tensor(max_rel_idx) else max_rel_idx)


class SingleBenchSummary:
    def __init__(self, precision_result: SingleBenchmarkAccuracyResult, npu_dtype=None,
                 bench_dtype=None, shape=None, error_thd=None, eb_thd=None, failed_info=None):
        self.npu_dtype = npu_dtype
        self.bench_dtype = bench_dtype
        self.shape = shape
        self.result = precision_result.result
        self.error_balance = precision_result.error_balance
        self.max_abs_diff = precision_result.max_abs_diff
        self.max_abs_idx = precision_result.max_abs_idx
        self.max_rel_diff = precision_result.max_rel_diff
        self.max_rel_idx = precision_result.max_rel_idx
        self.eb_thd = eb_thd
        self.error_thd = error_thd
        self.failed_info = failed_info

    def get_check_result(self):
        if self.result:
            return "PASS"
        else:
            return "FAILED"

    def get_result_msg(self):
        result_str = ""
        if self.failed_info:
            result_str = self.failed_info
            return result_str

        if self.result:
            result_str += "误差均衡性EB: %s <= 阈值%s\n" % (self.error_balance, self.eb_thd)
            result_str += "最大绝对误差: %s <= 阈值%s\n" % (self.max_abs_diff, self.error_thd)
            result_str += "最大相对误差: %s <= 阈值%s\n" % (self.max_rel_diff, self.error_thd)
        else:
            if self.error_balance > self.eb_thd:
                result_str += "误差均衡性EB超过阈值%s: EB = %s\n" % (
                    self.eb_thd,
                    self.error_balance,
                )
            if self.max_abs_diff > self.error_thd:
                result_str += "小值域最大绝对误差超过阈值%s: idx = %s, 绝对误差 = %s\n" % (
                    self.error_thd,
                    self.max_abs_idx,
                    self.max_abs_diff
                )
            if self.max_rel_diff > self.error_thd:
                result_str += "大值域最大相对误差超过阈值%s: idx = %s, 相对误差 = %s\n" % (
                    self.error_thd,
                    self.max_rel_idx,
                    self.max_rel_diff,
                )
        return result_str

    def print_detail_table(self):
        table = PrettyTable()
        table.title = "Single Benchmark Metrics Info"
        table.field_names = ["Index", "Result", "Threshold"]
        table.add_row(["error_balance", self.error_balance, self.eb_thd])
        table.add_row(["max_abs_diff", self.max_abs_diff, self.error_thd])
        table.add_row(["max_abs_idx", self.max_abs_idx, "-"])
        table.add_row(["max_rel_diff", self.max_rel_diff, self.error_thd])
        table.add_row(["max_rel_idx", self.max_rel_idx, "-"])

        logger.info(table)

    def to_column_value(self):
        return [self.bench_dtype, self.npu_dtype, self.shape, self.error_balance,
                self.max_abs_diff, self.max_abs_idx, self.max_rel_diff, self.max_rel_idx,
                self.eb_thd, self.error_thd, self.result, self.failed_info]


def single_benchmark_compare(npu_out: torch.Tensor, bench_out: torch.Tensor, high_precision: bool = True):
    benchmark_standard = SingleBenchmarkCompareStandard(high_precision)
    npu_out = npu_out.flatten()
    bench_out = bench_out.flatten()

    compare_results = SingleBenchmarkAccuracyCompare.precision_compare_for_case(npu_out, bench_out, benchmark_standard)
    (
        precision_result,
        error_thd,
        eb_thd,
        failed_info
    ) = (compare_results.accuracy_result, compare_results.error_threshold,
         compare_results.eb_threshold, compare_results.failed_information)

    summary = SingleBenchSummary(precision_result, str(npu_out.dtype), str(bench_out.dtype), tuple(npu_out.shape),
                                 error_thd, eb_thd, failed_info)
    result = summary.result
    details = summary.to_column_value()
    return result, details


def calc_status_details_list_tuple(npu_out, bench_out, summary):
    status, details = [], []
    if len(bench_out) != len(npu_out):
        summary.result = False
        summary.failed_info = "bench and npu output structure is different."
        return False, summary.to_column_value()
    for b_out_i, n_out_i in zip(bench_out, npu_out):
        status_i, details_i = single_benchmark_compare_wrap(n_out_i, b_out_i)
        status.append(status_i)
        details.append(details_i)
    return status, details


def calc_status_details_dict(npu_out, bench_out, summary):
    b_keys, n_keys = set(bench_out.keys()), set(npu_out.keys())
    if b_keys != n_keys:
        summary.result = False
        summary.failed_info = "bench and npu_output dict keys are different."
        return False, summary.to_column_value()
    else:
        status, details = single_benchmark_compare_wrap(list(bench_out.values()), list(npu_out.values()))
        return status, details


def calc_status_details_tensor(npu_out, bench_out, summary):
    return single_benchmark_compare(npu_out, bench_out)


def calc_status_details_builtin(npu_out, bench_out, summary):
    summary.bench_dtype = str(type(bench_out))
    summary.npu_dtype = str(type(npu_out))
    status = bench_out == npu_out
    summary.result = status
    return status, summary.to_column_value()


def calc_status_details_none(npu_out, bench_out, summary):
    summary.result = True
    summary.failed_info = "Output is None."
    return True, summary.to_column_value()


def single_benchmark_compare_wrap(npu_output: torch.Tensor, bench_output: torch.Tensor):
    type_method_dict = {
        (list, tuple): calc_status_details_list_tuple,
        dict: calc_status_details_dict,
        torch.Tensor: calc_status_details_tensor,
        (bool, int, float, str): calc_status_details_builtin,
        None: calc_status_details_none,
    }

    result = SingleBenchmarkAccuracyResult(result=True)
    bench_summary = SingleBenchSummary(result)
    for type1, func in type_method_dict.items():
        if isinstance(bench_output, type1):
            return func(npu_output, bench_output, bench_summary)

    bench_summary.result = True
    bench_summary.failed_info = "Unexpected output type: {}".format(type(bench_output))
    return True, bench_summary.to_column_value()
