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

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import argparse

from msprobe.core.compare.offline_data_compare import (
    compare_offline_data_mode,
    call_msaccucmp,
    _check_msaccucmp_file
)
from msprobe.core.common.utils import CompareException


class TestCompareOfflineDataMode(unittest.TestCase):

    @patch('msprobe.core.compare.offline_data_compare.call_msaccucmp')
    def test_compare_offline_data_mode_with_all_params(self, mock_call_msaccucmp):
        args = argparse.Namespace(
            target_path="/path/to/target",
            golden_path="/path/to/golden",
            fusion_rule_file="/path/to/fusion",
            quant_fusion_rule_file="/path/to/quant",
            close_fusion_rule_file="/path/to/close",
            output_path="/path/to/output"
        )
        
        compare_offline_data_mode(args)
        
        # 验证 call_msaccucmp 被调用，并且参数正确
        mock_call_msaccucmp.assert_called_once()
        call_args = mock_call_msaccucmp.call_args[0][0]
        self.assertIn('-m', call_args)
        self.assertIn('/path/to/target', call_args)
        self.assertIn('-g', call_args)
        self.assertIn('/path/to/golden', call_args)
        self.assertIn('-f', call_args)
        self.assertIn('/path/to/fusion', call_args)
        self.assertIn('-q', call_args)
        self.assertIn('/path/to/quant', call_args)
        self.assertIn('-cf', call_args)
        self.assertIn('/path/to/close', call_args)
        self.assertIn('-out', call_args)
        self.assertIn('/path/to/output', call_args)

    @patch('msprobe.core.compare.offline_data_compare.call_msaccucmp')
    def test_compare_offline_data_mode_with_required_params_only(self, mock_call_msaccucmp):
        args = argparse.Namespace(
            target_path="/path/to/target",
            golden_path="/path/to/golden",
            fusion_rule_file=None,
            quant_fusion_rule_file=None,
            close_fusion_rule_file=None,
            output_path=None
        )
        
        compare_offline_data_mode(args)
        
        # 验证 call_msaccucmp 被调用
        mock_call_msaccucmp.assert_called_once()
        call_args = mock_call_msaccucmp.call_args[0][0]
        self.assertIn('-m', call_args)
        self.assertIn('/path/to/target', call_args)
        self.assertIn('-g', call_args)
        self.assertIn('/path/to/golden', call_args)
        # 验证可选参数不存在
        self.assertNotIn('-f', call_args)
        self.assertNotIn('-q', call_args)
        self.assertNotIn('-cf', call_args)
        self.assertNotIn('-out', call_args)

    @patch('msprobe.core.compare.offline_data_compare.call_msaccucmp')
    def test_compare_offline_data_mode_with_partial_params(self, mock_call_msaccucmp):
        args = argparse.Namespace(
            target_path="/path/to/target",
            golden_path="/path/to/golden",
            fusion_rule_file=None,
            quant_fusion_rule_file=None,
            close_fusion_rule_file=None,
            output_path="/path/to/output"
        )
        
        compare_offline_data_mode(args)
        
        # 验证 call_msaccucmp 被调用
        mock_call_msaccucmp.assert_called_once()
        call_args = mock_call_msaccucmp.call_args[0][0]
        self.assertIn('-m', call_args)
        self.assertIn('-g', call_args)
        self.assertIn('-out', call_args)
        self.assertIn('/path/to/output', call_args)

    @patch('msprobe.core.compare.offline_data_compare.call_msaccucmp')
    def test_compare_offline_data_mode_with_none_values(self, mock_call_msaccucmp):
        args = argparse.Namespace(
            target_path=None,
            golden_path=None,
            fusion_rule_file=None,
            quant_fusion_rule_file=None,
            close_fusion_rule_file=None,
            output_path=None
        )
        
        compare_offline_data_mode(args)
        
        # 验证 call_msaccucmp 被调用，但参数列表应该为空
        mock_call_msaccucmp.assert_called_once()
        call_args = mock_call_msaccucmp.call_args[0][0]
        # 所有参数都不应该存在
        self.assertNotIn('-m', call_args)
        self.assertNotIn('-g', call_args)
        self.assertNotIn('-f', call_args)
        self.assertNotIn('-q', call_args)
        self.assertNotIn('-cf', call_args)
        self.assertNotIn('-out', call_args)
        self.assertEqual(len(call_args), 0)


class TestCheckMsaccucmpFile(unittest.TestCase):

    @patch('os.path.exists')
    def test_check_msaccucmp_file_exists_first_path(self, mock_exists):
        """测试在第一个路径选项中找到文件"""
        def exists_side_effect(path):
            # 第一个路径存在，返回 True
            if "toolkit/tools/operator_cmp/compare" in path and path.endswith("msaccucmp.py"):
                return True
            return False
        
        mock_exists.side_effect = exists_side_effect
        cann_path = "/test/path"
        result = _check_msaccucmp_file(cann_path)
        expected_path = os.path.join(cann_path, "toolkit/tools/operator_cmp/compare", "msaccucmp.py")
        self.assertEqual(result, expected_path)

    @patch('msprobe.core.compare.offline_data_compare.logger')
    @patch('os.path.exists')
    def test_check_msaccucmp_file_not_exists(self, mock_exists, mock_logger):
        """测试所有路径选项都不存在文件"""
        mock_exists.return_value = False
        cann_path = "/test/path"
        
        with self.assertRaises(CompareException) as context:
            _check_msaccucmp_file(cann_path)
        
        self.assertEqual(context.exception.code, CompareException.INVALID_PATH_ERROR)
        # 验证日志被调用
        mock_logger.error.assert_called_once()
        # 验证尝试了所有路径选项（应该调用2次，对应两个路径选项）
        self.assertEqual(mock_exists.call_count, 2)


class TestCallMsaccucmp(unittest.TestCase):
    @patch('msprobe.core.compare.offline_data_compare._check_msaccucmp_file')
    @patch('subprocess.Popen')
    @patch('msprobe.core.compare.offline_data_compare.logger')
    def test_call_msaccucmp_success(self, mock_logger, mock_popen, mock_check_file):
        mock_check_file.return_value = "/path/to/msaccucmp.py"
        
        # 模拟 subprocess.Popen
        mock_process = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.readline = MagicMock(side_effect=[
            "2024-01-01 10:00:00 test output\n",
            "2024-01-01 10:00:01 another line\n",
            ""  # 空字符串表示结束
        ])
        mock_process.stdout = mock_stdout
        mock_process.returncode = 0
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process
        
        cmd_args = ['-m', '/path/to/target', '-g', '/path/to/golden']
        result = call_msaccucmp(cmd_args)
        
        # 验证结果
        self.assertEqual(result, mock_process)
        mock_check_file.assert_called_once()
        mock_popen.assert_called_once()
        mock_process.wait.assert_called_once()
        # 验证日志被调用
        self.assertTrue(mock_logger.info.called)
        # 验证 raw 方法被调用（对日期开头的行）
        self.assertTrue(mock_logger.raw.called)

    @patch('msprobe.core.compare.offline_data_compare._check_msaccucmp_file')
    @patch('msprobe.core.compare.offline_data_compare.logger')
    def test_call_msaccucmp_file_not_found(self, mock_logger, mock_check_file):
        """测试 msaccucmp 文件不存在的情况"""
        # _check_msaccucmp_file 会抛出异常
        mock_check_file.side_effect = CompareException(CompareException.INVALID_PATH_ERROR)
        
        cmd_args = ['-m', '/path/to/target']
        
        with self.assertRaises(CompareException) as context:
            call_msaccucmp(cmd_args)
        
        self.assertEqual(context.exception.code, CompareException.INVALID_PATH_ERROR)

    @patch('msprobe.core.compare.offline_data_compare._check_msaccucmp_file')
    @patch('subprocess.Popen')
    @patch('msprobe.core.compare.offline_data_compare.logger')
    def test_call_msaccucmp_with_return_code_2(self, mock_logger, mock_popen, mock_check_file):
        mock_check_file.return_value = "/path/to/msaccucmp.py"
        
        mock_process = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.readline = MagicMock(side_effect=["2024-01-01 10:00:00 test\n", ""])
        mock_process.stdout = mock_stdout
        mock_process.returncode = 2
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process
        
        cmd_args = ['-m', '/path/to/target']
        result = call_msaccucmp(cmd_args)
        
        # 返回码 2 应该被认为是成功的
        self.assertEqual(result, mock_process)
        # 验证成功日志被调用
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        self.assertTrue(any('successfully' in call for call in info_calls))

    @patch('msprobe.core.compare.offline_data_compare._check_msaccucmp_file')
    @patch('subprocess.Popen')
    @patch('msprobe.core.compare.offline_data_compare.logger')
    def test_call_msaccucmp_with_error_return_code(self, mock_logger, mock_popen, mock_check_file):
        mock_check_file.return_value = "/path/to/msaccucmp.py"
        
        mock_process = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.readline = MagicMock(side_effect=["2024-01-01 10:00:00 error\n", ""])
        mock_process.stdout = mock_stdout
        mock_process.returncode = 1
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process
        
        cmd_args = ['-m', '/path/to/target']
        result = call_msaccucmp(cmd_args)
        
        # 应该返回 process，但记录错误日志
        self.assertEqual(result, mock_process)
        # 验证错误日志被调用
        error_calls = [str(call) for call in mock_logger.error.call_args_list]
        self.assertTrue(any('failed' in call.lower() for call in error_calls))

    @patch('msprobe.core.compare.offline_data_compare._check_msaccucmp_file')
    @patch('subprocess.Popen')
    @patch('msprobe.core.compare.offline_data_compare.logger')
    def test_call_msaccucmp_filters_non_date_lines(self, mock_logger, mock_popen, mock_check_file):
        mock_check_file.return_value = "/path/to/msaccucmp.py"
        
        mock_process = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.readline = MagicMock(side_effect=[
            "not a date line\n",  # 应该被过滤
            "2024-01-01 10:00:00 valid line\n",  # 应该被输出
            "invalid format\n",  # 应该被过滤
            "2024-12-31 23:59:59 another valid\n",  # 应该被输出
            ""  # 结束
        ])
        mock_process.stdout = mock_stdout
        mock_process.returncode = 0
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process
        
        cmd_args = ['-m', '/path/to/target']
        call_msaccucmp(cmd_args)
        
        # 验证 raw 方法被调用（只对日期开头的行）
        # 应该只调用 2 次（两个日期开头的行）
        self.assertEqual(mock_logger.raw.call_count, 2)
        mock_process.wait.assert_called_once()

    @patch('msprobe.core.compare.offline_data_compare._check_msaccucmp_file')
    @patch('subprocess.Popen')
    @patch('msprobe.core.compare.offline_data_compare.logger')
    def test_call_msaccucmp_exception_handling(self, mock_logger, mock_popen, mock_check_file):
        mock_check_file.return_value = "/path/to/msaccucmp.py"
        
        # 模拟 Popen 抛出异常
        mock_popen.side_effect = Exception("Subprocess error")
        
        cmd_args = ['-m', '/path/to/target']
        
        with self.assertRaises(CompareException) as context:
            call_msaccucmp(cmd_args)
        
        self.assertEqual(context.exception.code, CompareException.UNKNOWN_ERROR)
        mock_logger.error.assert_called()

    @patch('msprobe.core.compare.offline_data_compare._check_msaccucmp_file')
    @patch('subprocess.Popen')
    @patch('msprobe.core.compare.offline_data_compare.logger')
    def test_call_msaccucmp_command_format(self, mock_logger, mock_popen, mock_check_file):
        mock_check_file.return_value = "/path/to/msaccucmp.py"
        
        mock_process = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.readline = MagicMock(side_effect=["2024-01-01 10:00:00 test\n", ""])
        mock_process.stdout = mock_stdout
        mock_process.returncode = 0
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process
        
        cmd_args = ['-m', '/path/to/target', '-g', '/path/to/golden', '-out', '/path/to/output']
        call_msaccucmp(cmd_args)
        
        # 验证 Popen 被调用，并且命令格式正确
        call_args = mock_popen.call_args[0][0]
        self.assertEqual(call_args[0], sys.executable)  # Python 解释器
        self.assertEqual(call_args[1], "/path/to/msaccucmp.py")  # 脚本路径
        self.assertEqual(call_args[2], "compare")  # compare 子命令
        # 验证后续参数
        self.assertIn('-m', call_args)
        self.assertIn('/path/to/target', call_args)
        self.assertIn('-g', call_args)
        self.assertIn('/path/to/golden', call_args)
        self.assertIn('-out', call_args)
        self.assertIn('/path/to/output', call_args)

    @patch('msprobe.core.compare.offline_data_compare._check_msaccucmp_file')
    @patch('subprocess.Popen')
    @patch('msprobe.core.compare.offline_data_compare.logger')
    def test_call_msaccucmp_empty_cmd_args(self, mock_logger, mock_popen, mock_check_file):
        mock_check_file.return_value = "/path/to/msaccucmp.py"
        
        mock_process = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.readline = MagicMock(side_effect=["2024-01-01 10:00:00 test\n", ""])
        mock_process.stdout = mock_stdout
        mock_process.returncode = 0
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process
        
        cmd_args = []
        call_msaccucmp(cmd_args)
        call_args = mock_popen.call_args[0][0]
        self.assertEqual(call_args[0], sys.executable)
        self.assertEqual(call_args[1], "/path/to/msaccucmp.py")
        self.assertEqual(call_args[2], "compare")
        self.assertEqual(len(call_args), 3)

    @patch('msprobe.core.compare.offline_data_compare._check_msaccucmp_file')
    @patch('subprocess.Popen')
    @patch('msprobe.core.compare.offline_data_compare.logger')
    def test_call_msaccucmp_stdout_close_called(self, mock_logger, mock_popen, mock_check_file):
        mock_check_file.return_value = "/path/to/msaccucmp.py"
        
        mock_process = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.readline = MagicMock(side_effect=["2024-01-01 10:00:00 test\n", ""])
        mock_process.stdout = mock_stdout
        mock_process.returncode = 0
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process
        
        cmd_args = ['-m', '/path/to/target']
        call_msaccucmp(cmd_args)
        
        # 验证 stdout.close() 被调用
        mock_stdout.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
