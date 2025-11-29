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

import unittest
import os
import json
import tempfile
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock

from msprobe.core.config_check.checkers.env_args_checker import (
    collect_env_data, 
    get_device_type, 
    compare_env_data,
    EnvArgsChecker
)
from msprobe.core.common.const import Const


class TestCollectEnvData(unittest.TestCase):
    """测试collect_env_data函数"""

    @patch.dict(os.environ, {'TEST_VAR': 'test_value', 'PATH': '/usr/bin'}, clear=True)
    def test_collect_env_data_with_env_vars(self):
        """测试当环境变量存在时的数据收集"""
        result = collect_env_data()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['TEST_VAR'], 'test_value')
        self.assertEqual(result['PATH'], '/usr/bin')
        self.assertEqual(len(result), 2)

    @patch.dict(os.environ, {}, clear=True)
    def test_collect_env_data_no_env_vars(self):
        """测试当没有环境变量时的数据收集"""
        result = collect_env_data()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)


class TestGetDeviceType(unittest.TestCase):
    """测试get_device_type函数"""

    def test_get_device_type_with_ascend(self):
        """测试当环境变量包含ASCEND时的设备类型判断"""
        env_json = {
            'ASCEND_VISIBLE_DEVICES': '0,1,2',
            'PATH': '/usr/bin',
            'ASCEND_RT_VISIBLE_DEVICES': '0'
        }
        
        result = get_device_type(env_json)
        self.assertEqual(result, Const.NPU_LOWERCASE)

    def test_get_device_type_without_ascend(self):
        """测试当环境变量不包含ASCEND时的设备类型判断"""
        env_json = {
            'CUDA_VISIBLE_DEVICES': '0,1,2',
            'PATH': '/usr/bin',
            'HOME': '/home/user'
        }
        
        result = get_device_type(env_json)
        self.assertEqual(result, Const.GPU_LOWERCASE)

    def test_get_device_type_empty_dict(self):
        """测试当输入为空字典时的设备类型判断"""
        env_json = {}
        
        result = get_device_type(env_json)
        self.assertEqual(result, Const.GPU_LOWERCASE)


class TestCompareEnvData(unittest.TestCase):
    """测试compare_env_data函数"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.npu_path = os.path.join(self.temp_dir, 'npu_env.json')
        self.bench_path = os.path.join(self.temp_dir, 'bench_env.json')
        
        # 创建测试用的npu环境数据
        self.npu_env_data = {
            'ASCEND_VISIBLE_DEVICES': '0,1,2',
            'ASCEND_LAUNCH_BLOCKING': '0',
            'ASCEND_RT_VISIBLE_DEVICES': '0,1'
        }
        
        # 创建测试用的bench环境数据
        self.bench_env_data = {
            'CUDA_VISIBLE_DEVICES': '0,1,2',
            'CUDA_LAUNCH_BLOCKING': '0',
            'CUDA_VISIBLE_DEVICES': '0,1'
        }
        
        with open(self.npu_path, 'w') as f:
            json.dump(self.npu_env_data, f)
            
        with open(self.bench_path, 'w') as f:
            json.dump(self.bench_env_data, f)

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('msprobe.core.config_check.checkers.env_args_checker.load_yaml')
    @patch('msprobe.core.config_check.checkers.env_args_checker.load_json')
    def test_compare_env_data_same_values(self, mock_load_json, mock_load_yaml):
        """测试当环境变量值相同时的比较"""
        # Mock load_yaml返回配置数据
        mock_load_yaml.return_value = {
            'ASCEND_LAUNCH_BLOCKING': {
                'npu': {'name': 'ASCEND_LAUNCH_BLOCKING', 'default_value': '0'},
                'gpu': {'name': 'CUDA_LAUNCH_BLOCKING', 'default_value': '0'}
            }
        }
        
        # Mock load_json返回相同的环境变量值
        mock_load_json.side_effect = [
            {'ASCEND_LAUNCH_BLOCKING': '0'},  # NPU数据
            {'CUDA_LAUNCH_BLOCKING': '0'}   # GPU数据
        ]
        
        result = compare_env_data(self.npu_path, self.bench_path)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)  # 没有差异

    @patch('msprobe.core.config_check.checkers.env_args_checker.load_yaml')
    @patch('msprobe.core.config_check.checkers.env_args_checker.load_json')
    def test_compare_env_data_different_values(self, mock_load_json, mock_load_yaml):
        """测试当环境变量值不同时的比较"""
        # Mock load_yaml返回配置数据
        mock_load_yaml.return_value = {
            'ASCEND_LAUNCH_BLOCKING': {
                'npu': {'name': 'ASCEND_LAUNCH_BLOCKING', 'default_value': '0'},
                'gpu': {'name': 'CUDA_LAUNCH_BLOCKING', 'default_value': '0'}
            }
        }
        
        # Mock load_json返回不同的环境变量值
        mock_load_json.side_effect = [
            {'ASCEND_LAUNCH_BLOCKING': '1'},  # NPU数据
            {'CUDA_LAUNCH_BLOCKING': '0'}   # GPU数据
        ]
        
        result = compare_env_data(self.npu_path, self.bench_path)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['level'], Const.CONFIG_CHECK_ERROR)


class TestEnvArgsChecker(unittest.TestCase):
    """测试EnvArgsChecker类"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.pack_input = MagicMock()
        self.pack_input.output_zip_path = os.path.join(self.temp_dir, 'test.zip')

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('msprobe.core.config_check.checkers.env_args_checker.collect_env_data')
    @patch('msprobe.core.config_check.checkers.env_args_checker.create_file_in_zip')
    @patch('msprobe.core.config_check.checkers.env_args_checker.config_checking_print')
    def test_pack_success(self, mock_print, mock_create_file, mock_collect_env):
        """测试pack方法成功情况"""
        # Mock环境数据收集
        mock_env_data = {'TEST_VAR': 'test_value', 'PATH': '/usr/bin'}
        mock_collect_env.return_value = mock_env_data
        
        EnvArgsChecker.pack(self.pack_input)
        
        # 验证collect_env_data被调用
        mock_collect_env.assert_called_once()
        
        # 验证create_file_in_zip被调用，并检查参数
        mock_create_file.assert_called_once()
        call_args = mock_create_file.call_args[0]
        self.assertEqual(call_args[0], self.pack_input.output_zip_path)
        self.assertEqual(call_args[1], "env")
        
        # 验证写入的JSON数据
        written_data = json.loads(call_args[2])
        self.assertEqual(written_data, mock_env_data)
        
        # 验证打印信息
        mock_print.assert_called_once_with("add env args to zip")

    @patch('msprobe.core.config_check.checkers.env_args_checker.compare_env_data')
    @patch('msprobe.core.config_check.checkers.env_args_checker.process_pass_check')
    def test_compare_success(self, mock_process_pass, mock_compare_env):
        """测试compare方法成功情况"""
        # Mock比较数据
        mock_df = pd.DataFrame({
            'bench_env_name': ['TEST_ENV'],
            'cmp_env_name': ['TEST_ENV'],
            'bench_value': ['value1'],
            'cmp_value': ['value2'],
            'level': [Const.CONFIG_CHECK_ERROR]
        })
        mock_compare_env.return_value = mock_df
        mock_process_pass.return_value = False
        
        bench_dir = os.path.join(self.temp_dir, 'bench')
        cmp_dir = os.path.join(self.temp_dir, 'cmp')
        output_path = os.path.join(self.temp_dir, 'output')
        
        # 创建必要的目录和文件
        os.makedirs(bench_dir, exist_ok=True)
        os.makedirs(cmp_dir, exist_ok=True)
        
        with open(os.path.join(bench_dir, 'env'), 'w') as f:
            f.write('bench_data')
        with open(os.path.join(cmp_dir, 'env'), 'w') as f:
            f.write('cmp_data')
        
        result = EnvArgsChecker.compare(bench_dir, cmp_dir, output_path)
        
        # 验证返回结果
        self.assertEqual(result[0], "env")
        self.assertEqual(result[1], False)
        self.assertIsInstance(result[2], pd.DataFrame)
        
        # 验证compare_env_data被调用
        mock_compare_env.assert_called_once()

    @patch('msprobe.core.config_check.checkers.env_args_checker.compare_env_data')
    @patch('msprobe.core.config_check.checkers.env_args_checker.process_pass_check')
    def test_compare_with_pass_check(self, mock_process_pass, mock_compare_env):
        """测试compare方法当检查通过时"""
        # Mock比较数据（只有警告，没有错误）
        mock_df = pd.DataFrame({
            'bench_env_name': ['TEST_ENV'],
            'level': [Const.CONFIG_CHECK_WARNING]
        })
        mock_compare_env.return_value = mock_df
        mock_process_pass.return_value = True
        
        bench_dir = os.path.join(self.temp_dir, 'bench')
        cmp_dir = os.path.join(self.temp_dir, 'cmp')
        output_path = os.path.join(self.temp_dir, 'output')
        
        # 创建必要的目录和文件
        os.makedirs(bench_dir, exist_ok=True)
        os.makedirs(cmp_dir, exist_ok=True)
        
        with open(os.path.join(bench_dir, 'env'), 'w') as f:
            f.write('bench_data')
        with open(os.path.join(cmp_dir, 'env'), 'w') as f:
            f.write('cmp_data')
        
        result = EnvArgsChecker.compare(bench_dir, cmp_dir, output_path)
        
        # 验证返回结果
        self.assertEqual(result[0], "env")
        self.assertEqual(result[1], True)  # 检查通过
        self.assertIsInstance(result[2], pd.DataFrame)
