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

import unittest

import torch

from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check_utils import *
from msprobe.core.common.file_utils import create_directory, write_csv


class TestRunUtUtils(unittest.TestCase):
    def setUp(self):
        save_path = "temp_save_path"
        create_directory(save_path)
        self.save_path = os.path.realpath(save_path)
        self.result_file_name = "accuracy_checking_result_12345678901234.csv"
        self.detail_file_name = "accuracy_checking_details_12345678901234.csv"
        self.invalid_file_name = "invalid_file_name.csv"
        self.result_csv_path = os.path.join(self.save_path, self.result_file_name)
        self.details_csv_path = os.path.join(self.save_path, self.detail_file_name)
        self.invalid_csv_path = os.path.join(self.save_path, self.invalid_file_name)
        content = [["api_name", "metric_name", "metric_value", "metric_unit"]]
        write_csv(content, self.result_csv_path)
        write_csv(content, self.details_csv_path)
        write_csv(content, self.invalid_csv_path)
        
    def tearDown(self):
        for filename in os.listdir(self.save_path):
            os.remove(os.path.join(self.save_path, filename))
        os.rmdir(self.save_path)

    def test_get_validated_result_csv_patht_valid_mode(self):
        validated_path = get_validated_result_csv_path(self.result_csv_path, 'result')
        self.assertEqual(validated_path, self.result_csv_path)

    def test_get_validated_result_csv_path_invalid_mode(self):
        with self.assertRaises(ValueError):
            get_validated_result_csv_path(self.result_csv_path, 'invalid_mode')

    def test_get_validated_result_csv_path_file_path_validation(self):
        validated_path = get_validated_result_csv_path(self.result_csv_path, 'result')
        self.assertEqual(validated_path, self.result_csv_path)

    def test_get_validated_result_csv_patht_result_csv_name_pattern(self):
        validated_path = get_validated_result_csv_path(self.result_csv_path, 'result')
        self.assertEqual(validated_path, self.result_csv_path)

    def test_get_validated_result_csv_path_invalid_result_csv_name_pattern(self):
        with self.assertRaises(ValueError):
            get_validated_result_csv_path(self.invalid_csv_path, 'result')

    def test_get_validated_details_csv_path_file_name_replacement(self):
        validated_details_csv_path = get_validated_details_csv_path(self.result_csv_path)
        self.assertEqual(validated_details_csv_path, self.details_csv_path)

    def test_exec_api_functional_api(self):
        api_name = "sigmoid"
        args = (torch.tensor([1]))
        kwargs={}
        api_type = "Functional"
        exec_params = ExecParams(api_type, api_name, "cpu", args, kwargs, False, None)
        result = exec_api(exec_params)
        self.assertTrue(torch.allclose(result, torch.tensor(0.7311), atol=1e-4))

    def test_exec_api_tensor_api(self):
        api_name = "add"
        args = (torch.tensor(1), torch.tensor(2))
        kwargs={}
        api_type = "Tensor"
        exec_params = ExecParams(api_type, api_name, "cpu", args, kwargs, False, None)
        result = exec_api(exec_params)
        self.assertEqual(result, torch.tensor(3))

    def test_exec_api_torch_api(self):
        api_name = "add"
        args = (torch.tensor(1), torch.tensor(2))
        kwargs={}
        api_type = "Torch"
        exec_params = ExecParams(api_type, api_name, "cpu", args, kwargs, False, None)
        result = exec_api(exec_params)
        self.assertEqual(result, torch.tensor(3))

    def test_raise_bench_data_dtype_dtype_unchanged(self):
        arg = torch.tensor(1.0, dtype=torch.float32)
        raise_dtype = torch.float32
        result = raise_bench_data_dtype("api_name", arg, raise_dtype)
        self.assertEqual(result, arg)

    def test_raise_bench_data_dtype_dtype_changed(self):
        arg = torch.tensor(1.0, dtype=torch.float32)
        raise_dtype = torch.float64
        result = raise_bench_data_dtype("api_name", arg, raise_dtype)
        self.assertEqual(result.dtype, raise_dtype)

    def test_raise_bench_data_dtype_hf_32_standard_api(self):
        arg = torch.tensor(1.0, dtype=torch.float32)
        result = raise_bench_data_dtype("conv2d", arg)
        self.assertEqual(result, arg)
