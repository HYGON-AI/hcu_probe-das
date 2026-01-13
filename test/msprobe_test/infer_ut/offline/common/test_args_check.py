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

import unittest
from unittest.mock import patch
import argparse
from contextlib import ExitStack

from msprobe.infer.offline.compare.msquickcmp.common.args_check import (
    check_model_path_legality, check_output_path_legality, check_dict_kind_string,
    check_rank_range_valid, check_number_list, check_dym_range_string, str2bool
)


class BastCheckTestCase(unittest.TestCase):
    @staticmethod
    def apply_patches(mock_env):
        """Helper method to apply all patches"""
        stack = ExitStack()
        for patch_item in mock_env.setup_mocks():
            stack.enter_context(patch_item)
        return stack


class TestCheckModelPathLegality(BastCheckTestCase):
    @patch("msprobe.infer.offline.compare.msquickcmp.common.args_check.check_file_or_directory_path", return_value=None)
    def test_normal_file_success(self, mock_check_file_or_directory_path):
        """测试普通文件的成功场景"""
        result = check_model_path_legality("test_model.onnx")
        self.assertEqual(result, "test_model.onnx")


class TestCheckOutputPathLegality(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(check_output_path_legality(""))
        self.assertIsNone(check_output_path_legality(None))

    @patch("msprobe.infer.offline.compare.msquickcmp.common.args_check.check_output_dir_path", return_value=None)
    def test_valid_path(self, mock_check_output_dir_path):
        """测试有效输出路径"""
        result = check_output_path_legality("valid/path")
        self.assertEqual(result, "valid/path")


class TestCheckDictKindString(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(check_dict_kind_string(""))
        self.assertIsNone(check_dict_kind_string(None))

    def test_valid_input(self):
        """测试有效输入"""
        valid_inputs = [
            "input_name1:1,224,224,3",
            "input_name1:1,224,224,3;input_name2:3,300",
            "test.name:1,2,3",
            "test_name:1,2,3"
        ]
        for input_str in valid_inputs:
            self.assertEqual(check_dict_kind_string(input_str), input_str)

    def test_invalid_input(self):
        """测试无效输入"""
        invalid_inputs = [
            "input@name:1,2,3",
            "input name:1,2,3",
            "input+name:1,2,3",
            "input$name:1,2,3"
        ]
        for input_str in invalid_inputs:
            with self.assertRaises(argparse.ArgumentTypeError):
                check_dict_kind_string(input_str)


class TestCheckRankRangeValid(BastCheckTestCase):
    def test_valid_range(self):
        """测试有效范围的值"""
        valid_values = ["0", "100", "255"]
        for value in valid_values:
            self.assertEqual(check_rank_range_valid(value), value)

    def test_invalid_range(self):
        """测试无效范围的值"""
        invalid_values = ["-1", "256", "1000"]
        for value in invalid_values:
            with self.assertRaises(argparse.ArgumentTypeError):
                check_rank_range_valid(value)

    def test_invalid_input(self):
        """测试非数字输入"""
        invalid_inputs = ["abc", "12.34", ""]
        for value in invalid_inputs:
            with self.assertRaises(argparse.ArgumentTypeError):
                check_rank_range_valid(value)


class TestCheckNumberList(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(check_number_list(""))
        self.assertIsNone(check_number_list(None))

    def test_valid_number_list(self):
        """测试有效的数字列表"""
        valid_inputs = [
            "123",
            "123,456",
            "123,456,789"
        ]
        for input_str in valid_inputs:
            self.assertEqual(check_number_list(input_str), input_str)

    def test_invalid_number_list(self):
        """测试无效的数字列表"""
        invalid_inputs = [
            "abc",
            "123,abc",
            "123,456.789",
            "123,-456"
        ]
        for input_str in invalid_inputs:
            with self.assertRaises(argparse.ArgumentTypeError):
                check_number_list(input_str)


class TestCheckDymRangeString(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(check_dym_range_string(""))
        self.assertIsNone(check_dym_range_string(None))

    def test_valid_dym_range(self):
        """测试有效的动态范围字符串"""
        valid_inputs = [
            "1,2,3",
            "test_name:1,2,3",
            "test.name:1~3",
            "test-name:1,2,3"
        ]
        for input_str in valid_inputs:
            self.assertEqual(check_dym_range_string(input_str), input_str)

    def test_invalid_dym_range(self):
        """测试无效的动态范围字符串"""
        invalid_inputs = [
            "test@name:1,2,3",
            "test name:1,2,3",
            "test+name:1,2,3"
        ]
        for input_str in invalid_inputs:
            with self.assertRaises(argparse.ArgumentTypeError):
                check_dym_range_string(input_str)


class TestStr2Bool(BastCheckTestCase):
    def test_boolean_input(self):
        """测试布尔值输入"""
        self.assertTrue(str2bool(True))
        self.assertFalse(str2bool(False))

    def test_valid_true_strings(self):
        """测试表示True的有效字符串"""
        true_strings = ["yes", "true", "t", "y", "1"]
        for input_str in true_strings:
            self.assertTrue(str2bool(input_str))
            self.assertTrue(str2bool(input_str.upper()))

    def test_valid_false_strings(self):
        """测试表示False的有效字符串"""
        false_strings = ["no", "false", "f", "n", "0"]
        for input_str in false_strings:
            self.assertFalse(str2bool(input_str))
            self.assertFalse(str2bool(input_str.upper()))

    def test_invalid_strings(self):
        """测试无效字符串"""
        invalid_strings = ["maybe", "2", "invalid"]
        for input_str in invalid_strings:
            with self.assertRaises(argparse.ArgumentTypeError):
                str2bool(input_str)
