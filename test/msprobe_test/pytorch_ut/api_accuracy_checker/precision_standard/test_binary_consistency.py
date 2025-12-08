import unittest
import numpy as np
import torch

from msprobe.pytorch.api_accuracy_checker.precision_standard.binary_consistency import BinaryCompare


class InputData:
    """测试数据类"""
    def __init__(self, bench_output, device_output, compare_column, dtype):
        self.bench_output = bench_output
        self.device_output = device_output
        self.dtype = dtype
        self.compare_column = compare_column


class TestBinaryCompare(unittest.TestCase):
    def setUp(self):
        # 创建实际的测试数据
        self.bench_output = np.array([True, False, True, False])
        self.device_output = np.array([True, False, False, False])
        self.compare_column = {}
        self.dtype = torch.bool

        
        self.input_data = InputData(
            bench_output=self.bench_output,
            device_output=self.device_output,
            compare_column=self.compare_column,
            dtype=self.dtype
        )

    def test_binary_compare(self):
        """测试二进制比较的基本功能"""
        binary_compare = BinaryCompare(self.input_data)
        metrics = binary_compare._compute_metrics()

        # 在这个例子中，4个元素中有1个不匹配，所以错误率应该是0.25
        self.assertAlmostEqual(metrics['error_rate'], 0.25)

    def test_binary_compare_all_match(self):
        """测试完全匹配的情况"""
        input_data = InputData(
            bench_output=np.array([True, False, True]),
            device_output=np.array([True, False, True]),
            compare_column=self.compare_column,
            dtype=self.dtype
        )
        
        binary_compare = BinaryCompare(input_data)
        metrics = binary_compare._compute_metrics()
        
        self.assertAlmostEqual(metrics['error_rate'], 0.0)

    def test_binary_compare_no_match(self):
        """测试完全不匹配的情况"""
        input_data = InputData(
            bench_output=np.array([True, True, True]),
            device_output=np.array([False, False, False]),
            compare_column=self.compare_column,
            dtype=self.dtype
        )
        
        binary_compare = BinaryCompare(input_data)
        metrics = binary_compare._compute_metrics()
        
        self.assertAlmostEqual(metrics['error_rate'], 1.0)

    def test_binary_compare_multidimensional(self):
        """测试多维数组的情况"""
        input_data = InputData(
            bench_output=np.array([[True, False], [True, True]]),
            device_output=np.array([[True, False], [False, True]]),
            compare_column=self.compare_column,
            dtype=self.dtype
        )
        
        binary_compare = BinaryCompare(input_data)
        metrics = binary_compare._compute_metrics()
        
        # 4个元素中有1个不匹配，错误率应该是0.25
        self.assertAlmostEqual(metrics['error_rate'], 0.25)


if __name__ == '__main__':
    unittest.main()