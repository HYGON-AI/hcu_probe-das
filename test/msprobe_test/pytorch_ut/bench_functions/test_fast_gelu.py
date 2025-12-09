import unittest
import torch

from msprobe.pytorch.bench_functions.fast_gelu import npu_fast_gelu, npu_fast_gelu_backward


class TestNPUFastGELU(unittest.TestCase):
    def setUp(self):
        self.input0 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.grad = torch.ones_like(self.input0)

    def test_npu_fast_gelu_positive_and_negative(self):
        result = npu_fast_gelu(self.input0)
        expected_result = torch.tensor([
            -0.06434137, -0.15420423,  0.00000000,  0.84579575,  1.93565857
        ])
        self.assertTrue(torch.allclose(result, expected_result))

    def test_npu_fast_gelu_backward_positive_and_negative(self):
        grad = torch.ones_like(self.input0)
        result = npu_fast_gelu_backward(grad, self.input0)
        expected_result = torch.tensor([
            -0.07381535, -0.06777961,  0.50000000,  1.06777954,  1.07381523
        ])
        self.assertTrue(torch.allclose(result, expected_result))
