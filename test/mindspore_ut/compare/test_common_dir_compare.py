import unittest
import tempfile
import os
import numpy as np
import pandas as pd
from pathlib import Path
from python.msprobe.mindspore.compare.common_dir_compare import common_dir_compare

class TestCommonDirCompare(unittest.TestCase):
    def setUp(self):
        # 创建临时目录
        self.npu_dir = tempfile.mkdtemp()
        self.bench_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # 清理临时目录
        for dir_path in [self.npu_dir, self.bench_dir, self.output_dir]:
            for root, dirs, files in os.walk(dir_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(dir_path)

    def test_simple_directory_comparison(self):
        """测试简单目录结构比对"""
        # 创建测试npy文件
        np.save(os.path.join(self.npu_dir, "x_float32_0.npy"), np.random.rand(10, 10).astype(np.float32))
        np.save(os.path.join(self.npu_dir, "x_float32_1.npy"), np.random.rand(10, 10).astype(np.float32))
        np.save(os.path.join(self.bench_dir, "x_float32_0.npy"), np.random.rand(10, 10).astype(np.float32))
        np.save(os.path.join(self.bench_dir, "x_float32_1.npy"), np.random.rand(10, 10).astype(np.float32))
        
        # 执行比对
        input_params = {'npu_path': self.npu_dir, 'bench_path': self.bench_dir}
        result = common_dir_compare(input_params, self.output_dir)
        
        # 验证输出目录结构
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'result.csv')))
        
        # 验证结果文件内容
        result_df = pd.read_csv(os.path.join(self.output_dir, 'result.csv'))
        self.assertEqual(len(result_df), 2)
        self.assertIn('x_0', result_df['Name'].values)
        self.assertIn('x_1', result_df['Name'].values)

    def test_nested_directory_comparison(self):
        """测试嵌套目录结构比对"""
        # 创建嵌套目录
        os.makedirs(os.path.join(self.npu_dir, "rank0"))
        os.makedirs(os.path.join(self.bench_dir, "rank0"))
        
        # 创建测试npy文件
        np.save(os.path.join(self.npu_dir, "rank0", "y_float32_0.npy"), np.random.rand(5, 5).astype(np.float32))
        np.save(os.path.join(self.bench_dir, "rank0", "y_float32_0.npy"), np.random.rand(5, 5).astype(np.float32))
        
        input_params = {'npu_path': self.npu_dir, 'bench_path': self.bench_dir}
        result = common_dir_compare(input_params, self.output_dir)
        
        # 验证输出目录结构
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'rank0', 'result.csv')))
        
        # 验证结果文件内容
        result_df = pd.read_csv(os.path.join(self.output_dir, 'rank0', 'result.csv'))
        self.assertEqual(len(result_df), 1)
        self.assertIn('y_0', result_df['Name'].values)

    def test_filename_mapping(self):
        """测试文件名映射功能"""
        # 创建不同名称但通过映射关联的文件
        np.save(os.path.join(self.npu_dir, "a_float32_0.npy"), np.random.rand(4, 4).astype(np.float32))
        np.save(os.path.join(self.bench_dir, "b_float32_0.npy"), np.random.rand(4, 4).astype(np.float32))
        
        input_params = {'npu_path': self.npu_dir, 'bench_path': self.bench_dir, 'map_dict': {'a': 'b'}}
        result = common_dir_compare(input_params, self.output_dir)
        
        # 验证结果文件生成
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'result.csv')))
        
        # 验证结果文件内容
        result_df = pd.read_csv(os.path.join(self.output_dir, 'result.csv'))
        self.assertEqual(len(result_df), 1)
        self.assertIn('a_0', result_df['Name'].values)

    def test_large_number_of_files(self):
        """测试大量文件比对"""
        # 创建100对npy文件
        for i in range(100):
            np.save(os.path.join(self.npu_dir, f"data_float32_{i}.npy"), np.random.rand(20, 20).astype(np.float32))
            np.save(os.path.join(self.bench_dir, f"data_float32_{i}.npy"), np.random.rand(20, 20).astype(np.float32))
        
        input_params = {'npu_path': self.npu_dir, 'bench_path': self.bench_dir}
        result = common_dir_compare(input_params, self.output_dir)
        
        # 验证所有结果都被处理
        result_df = pd.read_csv(os.path.join(self.output_dir, 'result.csv'))
        self.assertEqual(len(result_df), 100)

    def test_empty_directory(self):
        """测试空目录"""
        input_params = {'npu_path': self.npu_dir, 'bench_path': self.bench_dir}
        result = common_dir_compare(input_params, self.output_dir)
        
        # 应该没有结果文件生成
        self.assertEqual(len(os.listdir(self.output_dir)), 0)

    def test_different_data_types(self):
        """测试不同数据类型的npy文件"""
        np.save(os.path.join(self.npu_dir, "type_float32_0.npy"), np.random.rand(2, 2).astype(np.float32))
        np.save(os.path.join(self.bench_dir, "type_float64_0.npy"), np.random.rand(2, 2).astype(np.float64))
        
        input_params = {'npu_path': self.npu_dir, 'bench_path': self.bench_dir, 'map_dict': {'type_float32': 'type_float64'}}
        result = common_dir_compare(input_params, self.output_dir)
        
        # 验证数据类型被正确记录
        result_df = pd.read_csv(os.path.join(self.output_dir, 'result.csv'))
        self.assertEqual(result_df.iloc[0]['NPU Dtype'], 'float32')
        self.assertEqual(result_df.iloc[0]['Bench Dtype'], 'float64')

if __name__ == '__main__':
    unittest.main()