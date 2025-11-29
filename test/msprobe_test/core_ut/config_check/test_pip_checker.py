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
import tempfile
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock

from msprobe.core.config_check.checkers.pip_checker import (
    load_pip_txt, 
    collect_pip_data, 
    compare_pip_data,
    PipPackageChecker
)
from msprobe.core.common.const import Const


class TestLoadPipTxt(unittest.TestCase):
    """测试load_pip_txt函数"""

    def test_load_pip_txt_normal_format(self):
        """测试正常格式的pip文件加载"""
        mock_content = """transformers=4.21.0
        torch=1.13.1
        numpy=1.21.6
        datasets=2.5.0"""
        
        with patch('msprobe.core.config_check.checkers.pip_checker.FileOpen', 
                   mock_open(read_data=mock_content)):
            result = load_pip_txt('dummy_path.txt')
            
            expected = {
                'transformers': '4.21.0',
                'torch': '1.13.1',
                'numpy': '1.21.6',
                'datasets': '2.5.0'
            }
            self.assertEqual(result, expected)


class TestCollectPipData(unittest.TestCase):
    """测试collect_pip_data函数"""

    @patch('msprobe.core.config_check.checkers.pip_checker.metadata.distributions')
    def test_collect_pip_data_success(self, mock_distributions):
        """测试成功收集pip数据"""
        # Mock包数据
        mock_pkg1 = MagicMock()
        mock_pkg1.metadata.get.return_value = 'transformers'
        mock_pkg1.version = '4.21.0'
        
        mock_pkg2 = MagicMock()
        mock_pkg2.metadata.get.return_value = 'torch'
        mock_pkg2.version = '1.13.1'
        
        mock_distributions.return_value = [mock_pkg1, mock_pkg2]
        
        result = collect_pip_data()
        
        expected = "transformers=4.21.0\ntorch=1.13.1\n"
        self.assertEqual(result, expected)

    @patch('msprobe.core.config_check.checkers.pip_checker.metadata.distributions')
    def test_collect_pip_data_empty(self, mock_distributions):
        """测试没有包的情况"""
        mock_distributions.return_value = []
        
        result = collect_pip_data()
        
        self.assertEqual(result, "")

    @patch('msprobe.core.config_check.checkers.pip_checker.metadata.distributions')
    def test_collect_pip_data_import_error(self, mock_distributions):
        """测试导入错误处理"""
        # 模拟ImportError异常
        mock_distributions.side_effect = ImportError("No module named 'importlib.metadata'")
        
        with self.assertRaises(ImportError):
            collect_pip_data()


class TestComparePipData(unittest.TestCase):
    """测试compare_pip_data函数"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.bench_path = os.path.join(self.temp_dir, 'bench_pip.txt')
        self.cmp_path = os.path.join(self.temp_dir, 'cmp_pip.txt')

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('msprobe.core.config_check.checkers.pip_checker.load_yaml')
    def test_compare_pip_data_same_versions(self, mock_load_yaml):
        """测试包版本相同的情况"""
        # Mock依赖配置
        mock_load_yaml.return_value = {
            "dependency": ["transformers", "torch", "numpy"]
        }
        
        # 创建测试文件
        with open(self.bench_path, 'w') as f:
            f.write("""transformers=4.21.0 
            torch=1.13.1
            numpy=1.21.6""")
        
        with open(self.cmp_path, 'w') as f:
            f.write("""transformers=4.21.0
            torch=1.13.1
            numpy=1.21.6""")
        
        result = compare_pip_data(self.bench_path, self.cmp_path)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)  # 没有差异

    @patch('msprobe.core.config_check.checkers.pip_checker.load_yaml')
    def test_compare_pip_data_different_versions(self, mock_load_yaml):
        """测试包版本不同的情况"""
        # Mock依赖配置
        mock_load_yaml.return_value = {
            "dependency": ["transformers", "torch"]
        }
        
        # 创建测试文件
        with open(self.bench_path, 'w') as f:
            f.write("""transformers=4.21.0
            torch=1.13.1""")
        
        with open(self.cmp_path, 'w') as f:
            f.write("""transformers=4.30.0
            torch=2.0.0""")
        
        result = compare_pip_data(self.bench_path, self.cmp_path)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # 两个包版本不同
        self.assertEqual(result.iloc[0]['level'], Const.CONFIG_CHECK_ERROR)
        self.assertEqual(result.iloc[1]['level'], Const.CONFIG_CHECK_ERROR)

    @patch('msprobe.core.config_check.checkers.pip_checker.load_yaml')
    def test_compare_pip_data_empty_dependency_list(self, mock_load_yaml):
        """测试空依赖列表"""
        # Mock依赖配置
        mock_load_yaml.return_value = {
            "dependency": []
        }
        
        # 创建测试文件
        with open(self.bench_path, 'w') as f:
            f.write("""transformers=4.21.0
            torch=1.13.1""")
        
        with open(self.cmp_path, 'w') as f:
            f.write("""transformers=4.21.0
            torch=1.13.1""")
        
        result = compare_pip_data(self.bench_path, self.cmp_path)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)  # 没有依赖需要比较


class TestPipPackageChecker(unittest.TestCase):
    """测试PipPackageChecker类"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.pack_input = MagicMock()
        self.pack_input.output_zip_path = os.path.join(self.temp_dir, 'test.zip')

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_class_attributes(self):
        """测试类的静态属性"""
        self.assertEqual(PipPackageChecker.target_name_in_zip, "pip")
        self.assertEqual(PipPackageChecker.result_header, 
                        ['package', 'bench version', 'cmp version', 'level'])

    @patch('msprobe.core.config_check.checkers.pip_checker.collect_pip_data')
    @patch('msprobe.core.config_check.checkers.pip_checker.create_file_in_zip')
    @patch('msprobe.core.config_check.checkers.pip_checker.config_checking_print')
    def test_pack_success(self, mock_print, mock_create_file, mock_collect_pip):
        """测试pack方法成功情况"""
        # Mock pip数据收集
        mock_pip_data = "transformers=4.21.0\ntorch=1.13.1\n"
        mock_collect_pip.return_value = mock_pip_data
        
        PipPackageChecker.pack(self.pack_input)
        
        # 验证collect_pip_data被调用
        mock_collect_pip.assert_called_once()
        
        # 验证create_file_in_zip被调用，并检查参数
        mock_create_file.assert_called_once()
        call_args = mock_create_file.call_args[0]
        self.assertEqual(call_args[0], self.pack_input.output_zip_path)
        self.assertEqual(call_args[1], "pip")
        self.assertEqual(call_args[2], mock_pip_data)
        
        # 验证打印信息
        mock_print.assert_called_once_with("add pip info to zip")

    @patch('msprobe.core.config_check.checkers.pip_checker.compare_pip_data')
    @patch('msprobe.core.config_check.checkers.pip_checker.process_pass_check')
    def test_compare_success(self, mock_process_pass, mock_compare_pip):
        """测试compare方法成功情况"""
        # Mock比较数据
        mock_df = pd.DataFrame({
            'package': ['transformers', 'torch'],
            'bench version': ['4.21.0', '1.13.1'],
            'cmp version': ['4.30.0', '2.0.0'],
            'level': [Const.CONFIG_CHECK_ERROR, Const.CONFIG_CHECK_ERROR]
        })
        mock_compare_pip.return_value = mock_df
        mock_process_pass.return_value = False
        
        bench_dir = os.path.join(self.temp_dir, 'bench')
        cmp_dir = os.path.join(self.temp_dir, 'cmp')
        output_path = os.path.join(self.temp_dir, 'output')
        
        # 创建必要的目录和文件
        os.makedirs(bench_dir, exist_ok=True)
        os.makedirs(cmp_dir, exist_ok=True)
        
        with open(os.path.join(bench_dir, 'pip'), 'w') as f:
            f.write('bench_data')
        with open(os.path.join(cmp_dir, 'pip'), 'w') as f:
            f.write('cmp_data')
        
        result = PipPackageChecker.compare(bench_dir, cmp_dir, output_path)
        
        # 验证返回结果
        self.assertEqual(result[0], "pip")
        self.assertEqual(result[1], False)
        self.assertIsInstance(result[2], pd.DataFrame)
        
        # 验证compare_pip_data被调用
        mock_compare_pip.assert_called_once()

    @patch('msprobe.core.config_check.checkers.pip_checker.compare_pip_data')
    @patch('msprobe.core.config_check.checkers.pip_checker.process_pass_check')
    def test_compare_with_pass_check(self, mock_process_pass, mock_compare_pip):
        """测试compare方法当检查通过时"""
        # Mock比较数据（没有错误）
        mock_df = pd.DataFrame({
            'package': ['transformers', 'torch'],
            'bench version': ['4.21.0', '1.13.1'],
            'cmp version': ['4.21.0', '1.13.1'],
            'level': [Const.CONFIG_CHECK_PASS, Const.CONFIG_CHECK_PASS]
        })
        mock_compare_pip.return_value = mock_df
        mock_process_pass.return_value = True
        
        bench_dir = os.path.join(self.temp_dir, 'bench')
        cmp_dir = os.path.join(self.temp_dir, 'cmp')
        output_path = os.path.join(self.temp_dir, 'output')
        
        # 创建必要的目录和文件
        os.makedirs(bench_dir, exist_ok=True)
        os.makedirs(cmp_dir, exist_ok=True)
        
        with open(os.path.join(bench_dir, 'pip'), 'w') as f:
            f.write('bench_data')
        with open(os.path.join(cmp_dir, 'pip'), 'w') as f:
            f.write('cmp_data')
        
        result = PipPackageChecker.compare(bench_dir, cmp_dir, output_path)
        
        # 验证返回结果
        self.assertEqual(result[0], "pip")
        self.assertEqual(result[1], True)  # 检查通过
        self.assertIsInstance(result[2], pd.DataFrame)
