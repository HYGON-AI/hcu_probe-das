import unittest
import torch

from msprobe.pytorch.bench_functions.moe_gating_top_k_softmax import npu_moe_gating_top_k_softmax, softmax_func


class TestNPUMoEGatingTopKSoftmax(unittest.TestCase):
    def setUp(self):
        self.input0 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.finished_optional = None
        self.k = 2

    def test_npu_moe_gating_top_k_softmax(self):
        # 调用 npu_moe_gating_top_k_softmax 函数
        result = npu_moe_gating_top_k_softmax(self.input0, self.finished_optional, self.k)
        
        # 预期的结果
        expected_result = (
            torch.tensor([[[[0.7311, 0.2689], [0.7311, 0.2689]]]]),
            torch.tensor([[[[1, 0], [1, 0]]]]),
            torch.tensor([[0]])
        )
        
        # 使用 torch.allclose 进行近似比较
        self.assertTrue(torch.allclose(result[0], expected_result[0], atol=1e-4))
        self.assertTrue(torch.allclose(result[1], expected_result[1], atol=1e-4))
        self.assertTrue(torch.allclose(result[2], expected_result[2], atol=1e-4))

    def test_softmax_func(self):
        # 调用 softmax_func 函数
        result = softmax_func(self.input0, -1)
        
        # 预期的结果
        expected_result = torch.tensor([[[[0.2689, 0.7311], [0.2689, 0.7311]]]])
        
        # 使用 torch.allclose 进行近似比较
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4))
