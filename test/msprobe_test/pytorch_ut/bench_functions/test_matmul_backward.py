import torch
import unittest

from msprobe.pytorch.bench_functions.matmul_backward import matmul_backward


class TestMatmulBackward(unittest.TestCase):
    def test_single_single(self):
        # 单维张量与单维张量的乘法
        grad = torch.tensor([1.0])
        self_tensor = torch.tensor([2.0])
        other = torch.tensor([3.0])
        mask = [True, True]
        expected_grad_self, expected_grad_other = torch.tensor([3.0]), torch.tensor([2.0])
        grad_self, grad_other = matmul_backward(grad, self_tensor, other, mask)
        self.assertTrue(torch.allclose(grad_self, expected_grad_self))
        self.assertTrue(torch.allclose(grad_other, expected_grad_other))
