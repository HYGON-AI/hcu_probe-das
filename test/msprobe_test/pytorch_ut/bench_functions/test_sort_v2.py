import unittest
import torch

from msprobe.pytorch.bench_functions.sort_v2 import npu_sort_v2


class TestSortV2(unittest.TestCase):
    def setUp(self):
        self.input0 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.dim = -1
        self.descending = False
        self.out = None

    def test_npu_sort_v2(self):
        # 调用 npu_sort_v2 函数
        result = npu_sort_v2(self.input0, self.dim, self.descending, self.out)
        
        # 预期的结果
        expected_result = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        
        # 使用 torch.allclose 进行近似比较
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4))
