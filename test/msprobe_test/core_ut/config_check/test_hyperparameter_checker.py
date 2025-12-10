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

import os
import json
import unittest
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open

from msprobe.core.config_check.checkers.hyperparameter_checker import (
    refine_json_keys, to_str_if_number, HyperparameterChecker, parameter_name_mapping
)
from msprobe.core.common.const import Const


class TestHelperFunctions(unittest.TestCase):
    """测试辅助函数"""

    def test_refine_json_keys(self):
        """测试refine_json_keys函数"""
        input_dict = {
            "a.b.c-d": "value1",
            "x.y.z": "value2",
            "p-q.r": "value3"
        }
        expected = {
            "c_d": "a.b.c-d",
            "z": "x.y.z",
            "r": "p-q.r"
        }
        result = refine_json_keys(input_dict)
        self.assertEqual(result, expected)

    def test_to_str_if_number(self):
        """测试to_str_if_number函数"""
        self.assertEqual(to_str_if_number(123), "123")
        self.assertEqual(to_str_if_number(3.14), "3.14")
        self.assertEqual(to_str_if_number("string"), "string")
        self.assertEqual(to_str_if_number(None), None)


class TestHyperparameterChecker(unittest.TestCase):
    """测试HyperparameterChecker类"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.bench_dir = os.path.join(self.temp_dir, "bench")
        self.cmp_dir = os.path.join(self.temp_dir, "cmp")
        self.hyper_dir = os.path.join(HyperparameterChecker.target_name_in_zip)
        os.makedirs(os.path.join(self.bench_dir, self.hyper_dir), exist_ok=True)
        os.makedirs(os.path.join(self.cmp_dir, self.hyper_dir), exist_ok=True)

        # 创建测试用的超参数文件
        self.bench_hyper_params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_epochs": 10,
            "optimizer": "Adam"
        }
        self.cmp_hyper_params = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "num_epochs": 20,
            "optimizer": "SGD"
        }

        # 写入benchmark超参数文件
        with open(os.path.join(self.bench_dir, self.hyper_dir, HyperparameterChecker.hyperparameters_file_list[0]), "w") as f:
            json.dump(self.bench_hyper_params, f)

        # 写入compare超参数文件
        with open(os.path.join(self.cmp_dir, self.hyper_dir, HyperparameterChecker.hyperparameters_file_list[0]), "w") as f:
            json.dump(self.cmp_hyper_params, f)

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('msprobe.core.config_check.checkers.hyperparameter_checker.ParserFactory')
    @patch('msprobe.core.config_check.checkers.hyperparameter_checker.config_checking_print')
    @patch('os.path.isfile')
    def test_pack_with_invalid_shell_path(self, mock_isfile, mock_print, mock_parser_factory):
        """测试pack方法，带有无效的shell_path参数"""
        # Mock
        mock_isfile.return_value = False

        # 创建pack_input对象
        pack_input = MagicMock()
        pack_input.shell_path = ["non_existent.sh"]
        pack_input.output_zip_path = "output.zip"

        # 执行测试
        HyperparameterChecker.pack(pack_input)

        # 验证
        mock_print.assert_called_with("Warning: Failed to extract hyperparameters from script ['non_existent.sh']")

    @patch('msprobe.core.config_check.checkers.hyperparameter_checker.process_pass_check')
    def test_compare(self, mock_process_pass):
        """测试compare方法"""
        # Mock process_pass_check
        mock_process_pass.return_value = False

        # 执行测试
        result = HyperparameterChecker.compare(self.bench_dir, self.cmp_dir, self.temp_dir)
        # 验证结果
        self.assertEqual(result[0], HyperparameterChecker.target_name_in_zip)
        self.assertEqual(result[1], False)
        self.assertIsInstance(result[2], pd.DataFrame)
        self.assertEqual(len(result[2]), 3)  # 4个超参数，其中batch_size相同，其他3个不同

    def test_compare_param(self):
        """测试compare_param方法"""
        result = HyperparameterChecker.compare_param(self.bench_hyper_params, self.cmp_hyper_params, "test.json")

        # 验证结果
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # 4个超参数，其中batch_size相同，其他3个不同

        # 检查batch_size是否相同（不应该出现在结果中）
        batch_size_diff = [r for r in result if "batch_size" in r]
        self.assertEqual(len(batch_size_diff), 0)

        # 检查其他参数是否不同
        learning_rate_diff = [r for r in result if "learning_rate" in r[1]]
        self.assertEqual(len(learning_rate_diff), 1)
        self.assertEqual(learning_rate_diff[0][3], "0.001")  # bench_value
        self.assertEqual(learning_rate_diff[0][4], "0.01")   # cmp_value
        self.assertEqual(learning_rate_diff[0][6], Const.CONFIG_CHECK_ERROR)  # level


    @patch('msprobe.core.config_check.checkers.hyperparameter_checker.parameter_name_mapping', {
        "learning_rate": ["lr", "rate"],
        "batch_size": ["bs", "batch"]
    })
    def test_fuzzy_match_parameter(self):
        """测试_fuzzy_match_parameter方法"""
        available_params = {
            "lr": "value1",
            "batch": "value2",
            "optimizer": "value3"
        }

        # 测试精确匹配
        name, matched_with = HyperparameterChecker._fuzzy_match_parameter("lr", available_params)
        self.assertEqual(name, "lr")
        self.assertEqual(matched_with, Const.MATCH_MODE_NAME)

        # 测试映射匹配
        name, matched_with = HyperparameterChecker._fuzzy_match_parameter("learning_rate", available_params)
        self.assertEqual(name, "lr")
        self.assertEqual(matched_with, Const.MATCH_MODE_MAPPING)

        # 测试别名匹配
        name, matched_with = HyperparameterChecker._fuzzy_match_parameter("bs", available_params)
        self.assertEqual(name, "batch")
        self.assertEqual(matched_with, Const.MATCH_MODE_MAPPING)

        # 测试模糊匹配（相似度）
        name, matched_with = HyperparameterChecker._fuzzy_match_parameter("optimize", available_params)
        self.assertEqual(name, "optimizer")
        self.assertTrue(matched_with.startswith(Const.MATCH_MODE_SIMILARITY))

        # 测试不匹配情况
        name, matched_with = HyperparameterChecker._fuzzy_match_parameter("nonexistent", available_params)
        self.assertIsNone(name)
        self.assertIsNone(matched_with)