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

import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import pandas as pd

from msprobe.pytorch.online_dispatch.compare import Saver, Comparator


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.dict_json_path = "./dict.json"
        self.list_json_path = "./list.json"
        Path(self.dict_json_path).touch()
        Path(self.list_json_path).touch()

    def tearDown(self):
        if os.path.exists(self.dict_json_path):
            os.remove(self.dict_json_path)
        if os.path.exists(self.list_json_path):
            os.remove(self.list_json_path)


class TestSaver(unittest.TestCase):
    def setUp(self):
        self.save_path = "./saver_save.csv"
        self.detail_save_path = "./saver_detail.csv"
        self.saver = Saver(self.save_path, self.detail_save_path, False)
        Path(self.save_path).touch()
        Path(self.detail_save_path).touch()

    def tearDown(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        if os.path.exists(self.detail_save_path):
            os.remove(self.detail_save_path)

    def test_write_csv_title(self):
        self.saver.write_csv_title()
        mock_data_save = {self.saver.COLUMN_API_NAME: {},
                          self.saver.COLUMN_FORWARD_SUCCESS: {},
                          self.saver.COLUMN_BACKWARD_SUCCESS: {},
                          "Message": {}}
        mock_data_detail = {'Npu Name': {}, 'Bench Dtype': {}, 'NPU Dtype': {}, 'Shape': {}, 'error_balance': {},
                            'max_abs_diff': {}, 'max_abs_idx': {}, 'max_rel_diff': {}, 'max_rel_idx': {}, 'eb_thd': {},
                            'error_thd': {}, 'Status': {}, 'Message': {}}
        self.assertEqual(pd.read_csv(self.save_path).to_dict(), mock_data_save)
        self.assertEqual(pd.read_csv(self.detail_save_path).to_dict(), mock_data_detail)

    @patch('msprobe.pytorch.online_dispatch.compare.Saver.get_statistics_from_result_csv')
    @patch('rich.console.Console')
    @patch('rich.console.Console.print')
    def test_print_pretest_result(self, mock_console_print, mock_console, mock_get_stats):
        my_test_class = self.saver

        my_test_class.test_result_cnt = {
            "total_num": 100,
            "success_num": 80,
            "forward_and_backward_fail_num": 10,
            "forward_or_backward_fail_num": 5,
            "forward_fail_num": 3,
            "backward_fail_num": 2
        }
        my_test_class.print_pretest_result()

        mock_console_print.assert_called()
        self.assertEqual(mock_console_print.call_count, 2)

    @patch('msprobe.pytorch.online_dispatch.compare.read_csv')
    @patch('os.path.basename')
    def test_get_statistics_from_result_csv_success(self, mock_basename, mock_read_csv):
        mock_data = pd.DataFrame({
            0: ['test1', 'test2', 'test3'],
            1: ['TRUE', 'FALSE', 'SKIP'],
            2: ['TRUE', 'FALSE', 'N/A']
        })
        mock_read_csv.return_value = mock_data
        mock_basename.return_value = 'mock_file.csv'

        self.saver.get_statistics_from_result_csv()
        self.assertEqual(self.saver.test_result_cnt['total_num'], 2)
        self.assertEqual(self.saver.test_result_cnt['success_num'], 1)
        self.assertEqual(self.saver.test_result_cnt['forward_and_backward_fail_num'], 1)

    @patch('msprobe.pytorch.online_dispatch.compare.read_csv')
    @patch('os.path.basename')
    def test_get_statistics_from_result_csv_incorrect_column_number(self, mock_basename, mock_read_csv):
        mock_data = pd.DataFrame({
            0: ['test1', 'test2'],
            1: ['TRUE', 'FALSE']
        })
        mock_read_csv.return_value = mock_data
        mock_basename.return_value = 'mock_file.csv'

        with self.assertRaises(ValueError) as context:
            self.saver.get_statistics_from_result_csv()
        self.assertIn("The number of columns in mock_file.csv is incorrect", str(context.exception))

    @patch('msprobe.pytorch.online_dispatch.compare.read_csv')
    @patch('os.path.basename')
    def test_get_statistics_from_result_csv_invalid_values(self, mock_basename, mock_read_csv):
        mock_data = pd.DataFrame({
            0: ['test1', 'test2'],
            1: ['INVALID', 'FALSE'],
            2: ['TRUE', 'FALSE']
        })
        mock_read_csv.return_value = mock_data
        mock_basename.return_value = 'mock_file.csv'

        with self.assertRaises(ValueError) as context:
            self.saver.get_statistics_from_result_csv()
        self.assertIn("The value in the 2nd or 3rd column of mock_file.csv is wrong", str(context.exception))

    def test_write_summary_csv(self):
        mock_test_result = Mock()
        mock_test_result.api_name = "api_name"
        mock_test_result.is_fwd_success = "SKIP"
        mock_test_result.is_bwd_success = "NOSKIP"
        mock_test_result.fwd_compare_alg_results = "result"
        self.saver.write_summary_csv(mock_test_result)
        mock_data_save = {"api_name": {}, "SKIP": {}, "NOSKIP": {}, "result": {}}
        self.assertEqual(pd.read_csv(self.save_path).to_dict(), mock_data_save)

    def test_write_detail_csv(self):
        mock_test_result = Mock()
        mock_test_result.api_name = "api_name"
        mock_test_result.fwd_compare_alg_results = ["f"]
        mock_test_result.bwd_compare_alg_results = ["b"]
        self.saver.write_detail_csv(mock_test_result)
        mock_data_detail = {'api_name.forward.output.0': {0: 'api_name.backward.output.0'}, 'f': {0: 'b'}}

        self.assertEqual(pd.read_csv(self.detail_save_path).to_dict(), mock_data_detail)


class TestComparator(unittest.TestCase):
    def setUp(self):
        self.save_path = "./saver_save.csv"
        self.detail_save_path = "./saver_detail.csv"
        self.comparator = Comparator(self.save_path, self.detail_save_path, False)
        Path(self.save_path).touch()
        Path(self.detail_save_path).touch()
        self.error_thd = {torch.float16: [2 ** -11, 2 ** -7],
                          torch.bfloat16: [2 ** -8, 2 ** -6],
                          torch.float32: [2 ** -14, 2 ** -11],
                          torch.float64: [2 ** -14, 2 ** -11]}
        self.eb_thd = {torch.float16: 2 ** -10,
                       torch.bfloat16: 2 ** -7,
                       torch.float32: 2 ** -14,
                       torch.float64: 2 ** -14}

    def tearDown(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        if os.path.exists(self.detail_save_path):
            os.remove(self.detail_save_path)

    def test_compare_core_wrapper(self):
        bench_out = torch.Tensor([1, 2, 3, 4])
        npu_out = torch.Tensor([1, 2, 3, 4])
        test_final_success, detailed_result_total = self.comparator._compare_core_wrapper(bench_out, npu_out)
        self.assertTrue(test_final_success)
        self.assertEqual(detailed_result_total, [['torch.float32', 'torch.float32', (4,), 0.0, 0.0, 0, 0.0, 0,
                                                  self.error_thd[torch.float32][0], self.eb_thd[torch.float32],
                                                  True, None]])

    @patch('msprobe.pytorch.online_dispatch.compare.ELEMENT_NUM_THRESHOLD', 4)
    def test_compare_dropout_success(self):
        # 元素数量大于阈值，零值分布接近的情况
        bench_out = torch.tensor([0, 0, 1, 1, 0])  # 3个零
        npu_out = torch.tensor([0, 1, 1, 0, 0])  # 3个零

        result, code = self.comparator._compare_dropout(bench_out, npu_out)
        self.assertTrue(result)
        self.assertEqual(code, 1)

    @patch('msprobe.pytorch.online_dispatch.compare.ELEMENT_NUM_THRESHOLD', 4)
    def test_compare_dropout_failure(self):
        # 元素数量大于阈值，零值分布差异大的情况
        bench_out = torch.tensor([0, 0, 1, 1, 0])  # 3个零
        npu_out = torch.tensor([1, 1, 1, 0, 1])  # 1个零

        result, code = self.comparator._compare_dropout(bench_out, npu_out)
        self.assertFalse(result)
        self.assertEqual(code, 0)

    @patch('msprobe.pytorch.online_dispatch.compare.ELEMENT_NUM_THRESHOLD', 4)
    def test_compare_dropout_tensor_num_less_than_threshold(self):
        # 元素数量小于阈值的情况
        bench_out = torch.tensor([1, 1, 1])
        npu_out = torch.tensor([1, 1, 1])

        result, code = self.comparator._compare_dropout(bench_out, npu_out)
        self.assertTrue(result)
        self.assertEqual(code, 1)

    def test_compare_dropout_empty_tensors(self):
        # 空张量的情况
        bench_out = torch.tensor([])
        npu_out = torch.tensor([])

        result, code = self.comparator._compare_dropout(bench_out, npu_out)
        self.assertTrue(result)
        self.assertEqual(code, 1)

    def test_compare_output_fwd(self):
        api_name = 'Tensor.add.0.forward.input.0'
        bench_out = torch.Tensor([1, 2, 3, 4])
        npu_out = torch.Tensor([1, 2, 3, 4])
        is_fwd_success, is_bwd_success = self.comparator.compare_output(api_name, bench_out, npu_out)
        self.assertTrue(is_fwd_success)
        self.assertTrue(is_bwd_success)

    @patch('msprobe.pytorch.online_dispatch.compare.ELEMENT_NUM_THRESHOLD', 4)
    def test_compare_output_fwd_dropout(self):
        api_name = 'Tensor.add.0.forward.input.dropout.0'
        bench_out = torch.tensor([0, 0, 1, 1, 0])  # 3个零
        npu_out = torch.tensor([1, 1, 1, 0, 1])  # 1个零
        is_fwd_success, is_bwd_success = self.comparator.compare_output(api_name, bench_out, npu_out)
        self.assertFalse(is_fwd_success)
        self.assertTrue(is_bwd_success)

    def test_compare_output_bwd(self):
        api_name = 'Tensor.add.0.forward.input.0'
        bench_out = torch.Tensor([1, 2, 3, 4])
        npu_out = torch.Tensor([1, 2, 3, 4])
        bench_grad = torch.Tensor([1])
        npu_grad = torch.Tensor([1])
        is_fwd_success, is_bwd_success = self.comparator.compare_output(api_name, bench_out, npu_out,
                                                                        bench_grad, npu_grad)
        self.assertTrue(is_fwd_success)
        self.assertTrue(is_bwd_success)

    @patch('msprobe.pytorch.online_dispatch.compare.ELEMENT_NUM_THRESHOLD', 4)
    def test_compare_output_bwd_dropout(self):
        api_name = 'Tensor.add.0.forward.input.dropout.0'
        bench_out = torch.tensor([0, 0, 1, 1, 0])  # 3个零
        npu_out = torch.tensor([1, 1, 1, 0, 1])  # 1个零
        bench_grad = [torch.Tensor([1, 2, 3, 4])]
        npu_grad = [torch.Tensor([1, 2, 3, 4])]
        is_fwd_success, is_bwd_success = self.comparator.compare_output(api_name, bench_out, npu_out,
                                                                        bench_grad, npu_grad)
        self.assertFalse(is_fwd_success)
        self.assertTrue(is_bwd_success)
