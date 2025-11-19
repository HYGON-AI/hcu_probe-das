# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

import unittest
from unittest.mock import patch, MagicMock

import torch

from msprobe.pytorch.online_dispatch.single_compare import SingleBenchmarkCompareStandard, \
    SingleBenchmarkAccuracyResult, SingleBenchmarkAccuracyCompare, SingleBenchSummary, calc_status_details_list_tuple, \
    calc_status_details_dict, calc_status_details_builtin, calc_status_details_none


class TestSingleBenchmarkCompareStandard(unittest.TestCase):
    def setUp(self):
        self.single_benchmark_compare_standard = SingleBenchmarkCompareStandard()

    @patch('logging.warning')
    def test_get_error_thd_when_input_f64(self, mock_warning):
        self.single_benchmark_compare_standard.get_error_thd(torch.float64)
        mock_warning.assert_called_once_with("the output data of fp64 uses the same standard as fp32.")

    @patch('logging.error')
    def test_get_error_thd_when_input_bool(self, mock_error):
        self.single_benchmark_compare_standard.get_error_thd(torch.bool)
        mock_error.assert_called_once_with("Single benchmark compare only supports floating point "
            "in fp16, bf16, fp32. ")

    def test_get_eb_thd_when_input_f16(self):
        self.assertEqual(self.single_benchmark_compare_standard.get_eb_thd(torch.float16), 2 ** -10)

    def test_get_eb_thd_when_input_bool(self):
        self.assertIsNone(self.single_benchmark_compare_standard.get_eb_thd(torch.bool))


class TestSingleBenchmarkAccuracyResult(unittest.TestCase):
    def setUp(self):
        self.single_benchmark_accuracy_result = SingleBenchmarkAccuracyResult(True, 1, 1,
                                                                              1, 1, 1)
    
    def test_get_result_result_false(self):
        self.single_benchmark_accuracy_result.get_result(0.5, 0.5)
        self.assertFalse(self.single_benchmark_accuracy_result.result)

    def test_get_result_result_true(self):
        self.single_benchmark_accuracy_result.get_result(2, 2)
        self.assertTrue(self.single_benchmark_accuracy_result.result)


class TestSingleBenchmarkAccuracyCompare(unittest.TestCase):
    @patch('logging.debug')
    @patch('logging.error')
    def test_check_output_size_pass(self, mock_error, mock_debug):
        # 模拟 npu_out 和 bench_out，它们的 numel() 返回 0
        npu_out = MagicMock()
        bench_out = MagicMock()
        npu_out.numel.return_value = 0
        bench_out.numel.return_value = 0
        npu_out.size.return_value = 0
        bench_out.size.return_value = 0

        result = SingleBenchmarkAccuracyCompare.check_output_size(npu_out, bench_out)

        self.assertTrue(result.result)
        mock_debug.assert_called_once_with(
            "The npu_output is [], and it is same as benchmark_output, "
            "the result of data_compare is Pass"
        )
        mock_error.assert_not_called()

    @patch('logging.debug')
    @patch('logging.error')
    def test_check_output_size_fail(self, mock_error, mock_debug):
        # 模拟 npu_out 和 bench_out，它们的 size() 不一样
        npu_out = MagicMock()
        bench_out = MagicMock()
        npu_out.numel.return_value = 10
        bench_out.numel.return_value = 10
        npu_out.size.return_value = [2, 3]
        bench_out.size.return_value = [3, 2]

        result = SingleBenchmarkAccuracyCompare.check_output_size(npu_out, bench_out)

        self.assertFalse(result.result)
        mock_error.assert_called_once_with(
            "the size of npu output[[2, 3]] and"
            "benchmark[[3, 2]] is not equal"
        )
        mock_debug.assert_not_called()

    def test_check_output_invalid_value_with_nan_and_inf(self):
        output = torch.tensor([float('nan'), float('inf')])
        result = SingleBenchmarkAccuracyCompare.check_output_invalid_value(output)

        self.assertTrue(result)

    def test_check_output_invalid_value_without_nan_or_inf(self):
        output = torch.tensor([1.0, 2.0, 3.0])
        result = SingleBenchmarkAccuracyCompare.check_output_invalid_value(output)

        self.assertFalse(result)

    @patch('msprobe.pytorch.common.log.logger.info')
    def test_compute_binary_diff(self, mock_info):
        npu_out = torch.Tensor([1, 2, 3, 4])
        bench_out = torch.Tensor([1, 2, 3, 4])
        result = SingleBenchmarkAccuracyCompare.compute_binary_diff(npu_out, bench_out)

        self.assertTrue(result.result)
        self.assertEqual(mock_info.call_count, 3)

    def test_compute_abs_diff_rather_than_0(self):
        npu_out = torch.Tensor([2, 2, 2, 2])
        bench_out = torch.Tensor([1, 1, 1, 1])
        error_thd = 0.5
        benchmark_standard = SingleBenchmarkCompareStandard()
        result = SingleBenchmarkAccuracyCompare.compute_abs_diff(npu_out, bench_out, error_thd, benchmark_standard)
        self.assertEqual(result, (1.0, 0))

    def test_compute_rel_diff_rather_than_0(self):
        npu_out = torch.Tensor([2, 2, 2, 2])
        bench_out = torch.Tensor([1, 1, 1, 1])
        error_thd = 0.5
        benchmark_standard = SingleBenchmarkCompareStandard()
        result = SingleBenchmarkAccuracyCompare.compute_rel_diff(npu_out, bench_out, error_thd, benchmark_standard)
        self.assertEqual(result, (float(torch.tensor(1)/(torch.tensor(1) + torch.finfo(torch.float).eps)), 0))


class TestSingleBenchSummary(unittest.TestCase):
    def test_get_check_result_pass(self):
        # 构造 precision_result 的模拟对象，result 为 True
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = True
        precision_result.error_balance = None
        precision_result.max_abs_diff = None
        precision_result.max_abs_idx = None
        precision_result.max_rel_diff = None
        precision_result.max_rel_idx = None

        summary = SingleBenchSummary(precision_result)
        self.assertEqual(summary.get_check_result(), "PASS")

    def test_get_check_result_failed(self):
        # 构造 precision_result 的模拟对象，result 为 False
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = False
        precision_result.error_balance = None
        precision_result.max_abs_diff = None
        precision_result.max_abs_idx = None
        precision_result.max_rel_diff = None
        precision_result.max_rel_idx = None

        summary = SingleBenchSummary(precision_result)
        self.assertEqual(summary.get_check_result(), "FAILED")

    def test_get_result_msg_failed_info(self):
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = False
        precision_result.error_balance = None
        precision_result.max_abs_diff = None
        precision_result.max_abs_idx = None
        precision_result.max_rel_diff = None
        precision_result.max_rel_idx = None

        summary = SingleBenchSummary(precision_result, failed_info="test fail")
        self.assertEqual(summary.get_result_msg(), "test fail")

    def test_get_result_msg_result_true(self):
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = True
        precision_result.error_balance = 0
        precision_result.max_abs_diff = 0
        precision_result.max_abs_idx = None
        precision_result.max_rel_diff = 0
        precision_result.max_rel_idx = None

        summary = SingleBenchSummary(precision_result, error_thd=1, eb_thd=1)
        result = summary.get_result_msg()
        target_result_str = "误差均衡性EB: 0 <= 阈值1\n最大绝对误差: 0 <= 阈值1\n最大相对误差: 0 <= 阈值1\n"
        self.assertEqual(result, target_result_str)

    def test_get_result_msg_result_false(self):
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = False
        precision_result.error_balance = 1
        precision_result.max_abs_diff = 1
        precision_result.max_abs_idx = 0
        precision_result.max_rel_diff = 1
        precision_result.max_rel_idx = 0

        summary = SingleBenchSummary(precision_result, error_thd=0, eb_thd=0)
        result = summary.get_result_msg()
        target_result_str = ("误差均衡性EB超过阈值0: EB = 1\n小值域最大绝对误差超过阈值0: idx = 0, "
                             "绝对误差 = 1\n大值域最大相对误差超过阈值0: idx = 0, 相对误差 = 1\n")
        self.assertEqual(result, target_result_str)

    @patch('msprobe.pytorch.common.log.logger.info')
    def test_print_detail_table(self, mock_info):
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = True
        precision_result.error_balance = 0
        precision_result.max_abs_diff = 1
        precision_result.max_abs_idx = 2
        precision_result.max_rel_diff = 3
        precision_result.max_rel_idx = 4

        summary = SingleBenchSummary(precision_result, eb_thd=6, error_thd=7)

        summary.print_detail_table()

        mock_info.assert_called_once()

    def test_to_column_value(self):
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = True
        precision_result.error_balance = 0
        precision_result.max_abs_diff = 1
        precision_result.max_abs_idx = 2
        precision_result.max_rel_diff = 3
        precision_result.max_rel_idx = 4

        summary = SingleBenchSummary(precision_result, npu_dtype='torch.float32', bench_dtype='torch.float32',
                                     shape=(4,), eb_thd=6, error_thd=6, failed_info=None)
        result = summary.to_column_value()
        target_result = ['torch.float32', 'torch.float32', (4,), 0, 1, 2, 3, 4, 6, 6, True, None]
        self.assertEqual(result, target_result)


class Testcalc(unittest.TestCase):
    def test_calc_status_details_list_tuple_lens_unequal(self):
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = True
        precision_result.error_balance = 0
        precision_result.max_abs_diff = 1
        precision_result.max_abs_idx = 2
        precision_result.max_rel_diff = 3
        precision_result.max_rel_idx = 4

        summary = SingleBenchSummary(precision_result)

        npu_out = torch.Tensor([1, 2, 3, 4])
        bench_out = torch.Tensor([1, 2, 3, 4, 5])

        status, details = calc_status_details_list_tuple(npu_out, bench_out, summary)
        self.assertFalse(status)
        self.assertEqual(details, [None, None, None, 0, 1, 2, 3, 4, None, None, False, "bench and npu output structure is different."])

    def test_calc_status_details_list_tuple_lens_equal(self):
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = True
        precision_result.error_balance = 0
        precision_result.max_abs_diff = 1
        precision_result.max_abs_idx = 2
        precision_result.max_rel_diff = 3
        precision_result.max_rel_idx = 4

        summary = SingleBenchSummary(precision_result)

        npu_out = torch.Tensor([1])
        bench_out = torch.Tensor([1])
        status, details = calc_status_details_list_tuple(npu_out, bench_out, summary)
        self.assertEqual(status, [True])
        self.assertEqual(details, [['torch.float32', 'torch.float32', (1,),
                                    0.0, 0.0, 0, 0.0, 0, 2 ** -14, 2 ** -14, True, None]])

    def test_calc_status_details_dict_key_differ(self):
        npu_out = {'a': 1, 'b': 2}
        bench_out = {'a': 1, 'c': 3}
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = True
        precision_result.error_balance = 0
        precision_result.max_abs_diff = 1
        precision_result.max_abs_idx = 2
        precision_result.max_rel_diff = 3
        precision_result.max_rel_idx = 4
        summary = SingleBenchSummary(precision_result)

        status, details = calc_status_details_dict(npu_out, bench_out, summary)

        self.assertFalse(status)
        self.assertFalse(summary.result, [None, None, None, 0, 1, 2, 3, 4,
                                          None, None, False, "bench and npu_output dict keys are different."])

    @patch('msprobe.pytorch.online_dispatch.single_compare.single_benchmark_compare_wrap')
    def test_calc_status_details_dict_key_same(self, mock_compare_wrap):
        npu_out = {'a': 1, 'b': 2}
        bench_out = {'a': 1, 'b': 2}
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = True
        precision_result.error_balance = 0
        precision_result.max_abs_diff = 1
        precision_result.max_abs_idx = 2
        precision_result.max_rel_diff = 3
        precision_result.max_rel_idx = 4
        summary = SingleBenchSummary(precision_result)

        mock_compare_wrap.return_value = (True, "details_info")

        status, details = calc_status_details_dict(npu_out, bench_out, summary)

        self.assertTrue(status)
        self.assertEqual(details, "details_info")
        mock_compare_wrap.assert_called_once_with(list(bench_out.values()), list(npu_out.values()))

    def test_calc_status_details_builtin(self):
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = True
        precision_result.error_balance = 0
        precision_result.max_abs_diff = 1
        precision_result.max_abs_idx = 2
        precision_result.max_rel_diff = 3
        precision_result.max_rel_idx = 4

        summary = SingleBenchSummary(precision_result)

        npu_out = 1
        bench_out = 1
        status, details = calc_status_details_builtin(npu_out, bench_out, summary)
        self.assertTrue(status)
        self.assertEqual(details, ["<class 'int'>", "<class 'int'>", None, 0, 1, 2, 3, 4, None, None, True, None])

    def test_calc_status_details_none(self):
        precision_result = type('SingleBenchmarkAccuracyResult', (), {})()
        precision_result.result = True
        precision_result.error_balance = 0
        precision_result.max_abs_diff = 1
        precision_result.max_abs_idx = 2
        precision_result.max_rel_diff = 3
        precision_result.max_rel_idx = 4

        summary = SingleBenchSummary(precision_result)

        npu_out = 1
        bench_out = 1
        status, details = calc_status_details_none(npu_out, bench_out, summary)
        self.assertTrue(status)
        self.assertEqual(details, [None, None, None, 0, 1, 2, 3, 4, None, None, True, 'Output is None.'])
