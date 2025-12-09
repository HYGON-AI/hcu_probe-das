import unittest
import torch

from msprobe.pytorch.bench_functions.mish import npu_mish


class TestNPUMish(unittest.TestCase):
    def setUp(self):
        self.input0 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.eps = 1e-5

    def test_npu_mish_positive(self):
        # 调用 npu_mish 函数
        result = npu_mish(self.input0)
        
        # 预期的结果
        expected_result = torch.tensor([[[[0.8651, 1.9440], [2.9865, 3.9974]]]])
        
        # 使用 torch.allclose 进行近似比较
        self.assertTrue(torch.allclose(result[0], expected_result, atol=1e-4))
