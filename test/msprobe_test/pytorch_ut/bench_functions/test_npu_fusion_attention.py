import pytest
import torch
import unittest

from msprobe.pytorch.bench_functions.npu_fusion_attention import npu_fusion_attention, npu_fusion_attention_grad, \
    broadcast_kv, convert_from_bsnd, convert_to_bsnd, rearrange


class TestNpuFusionAttention(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.N1 = 3
        self.N2 = 3
        self.S1 = 4
        self.S2 = 4
        self.D = 64
        self.query = torch.randn(self.B, self.S1, self.N1, self.D)
        self.key = torch.randn(self.B, self.S2, self.N2, self.D)
        self.value = torch.randn(self.B, self.S2, self.N2, self.D)
        self.atten_mask = torch.randn(self.B, 1, self.S1, self.S2)
        self.batch_size = 2
        self.seq_len = 3
        self.num_heads = 4
        self.head_dim = 5
        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim)

    def test_convert_from_bsnd(self):
        # 测试从 bsnd 转换到 BSH
        converted_tensor = convert_from_bsnd(self.input_tensor, "BSH")
        self.assertEqual(converted_tensor.shape, (self.batch_size, self.seq_len, self.num_heads * self.head_dim))

        # 测试从 bsnd 转换到 SBH
        converted_tensor = convert_from_bsnd(self.input_tensor, "SBH")
        self.assertEqual(converted_tensor.shape, (self.seq_len, self.batch_size, self.num_heads * self.head_dim))

        # 测试从 bsnd 转换到 BNSD
        converted_tensor = convert_from_bsnd(self.input_tensor, "BNSD")
        self.assertEqual(converted_tensor.shape, (self.batch_size, self.num_heads, self.seq_len, self.head_dim))

    def test_convert_to_bsnd(self):
        # 测试从 BSH 转换回 bsnd
        converted_tensor = convert_to_bsnd(rearrange(self.input_tensor, 'b s n d -> b s (n d)'), self.num_heads, "BSH")
        self.assertEqual(converted_tensor.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))

        # 测试从 SBH 转换回 bsnd
        converted_tensor = convert_to_bsnd(rearrange(self.input_tensor, 'b s n d -> s b (n d)'), self.num_heads, "SBH")
        self.assertEqual(converted_tensor.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        
        # 测试从 BNSD 转换回 bsnd
        converted_tensor = convert_to_bsnd(rearrange(self.input_tensor, 'b s n d -> b n s d'), self.num_heads, "BNSD")
        self.assertEqual(converted_tensor.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))

    def test_basic_forward_input_layout_is_BSND(self):
        # 基本前向传播测试
        out, _, _ = npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="BSND")
        self.assertEqual(out.shape, (self.B, self.S1, self.N1, self.D))

    def test_basic_forward_input_layout_is_BNSD(self):
        # 基本前向传播测试
        self.query = torch.randn(self.B, self.N1, self.S1, self.D)
        self.key = torch.randn(self.B, self.N2, self.S2, self.D)
        self.value = torch.randn(self.B, self.N2, self.S2, self.D)
        out, _, _ = npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="BNSD")
        self.assertEqual(out.shape, (self.B, self.N1, self.S1, self.D))

    def test_basic_backward(self):
        # 基本反向传播测试
        dx = torch.randn(self.B, self.S1, self.N1, self.D)
        out, _, _ = npu_fusion_attention_grad(dx, self.query, self.key, self.value, self.N1, "BSND")
        self.assertEqual(out.shape, (self.B,  self.S1, self.N1, self.D))  # 检查dq形状

    def test_different_input_layout(self):
        # BSH
        self.query = torch.randn(self.B, self.S1, self.N1*self.D)
        self.key = torch.randn(self.B, self.S2, self.N2*self.D)
        self.value = torch.randn(self.B, self.S2, self.N2*self.D)
        out, _, _ = npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="BSH")
        self.assertEqual(out.shape, (self.B, self.S1, self.N1 * self.D))

        # SBH
        self.query = torch.randn(self.S1, self.B, self.N1*self.D)
        self.key = torch.randn(self.S2, self.B, self.N2*self.D)
        self.value = torch.randn(self.S2, self.B, self.N2*self.D)
        out, _, _ = npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="SBH")
        self.assertEqual(out.shape, (self.S1, self.B, self.N1 * self.D))

    def test_with_attention_mask(self):
        # 带注意力掩码的测试
        out, _, _ = npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="BSND", atten_mask=self.atten_mask)
        self.assertIsNotNone(out)

    def test_sparse_mode(self):
        # 稀疏模式测试
        out, _, _ = npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="BSND", sparse_mode=2)
        self.assertIsNotNone(out)

    def test_invalid_input(self):
        # 无效输入测试
        with self.assertRaises(ValueError):
            npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="INVALID")

    def test_mismatch_dims(self):
        # 维度不匹配测试
        self.key = torch.randn(self.B, self.N1 + 1, self.S2, self.D)  # 故意制造维度不匹配
        with self.assertRaises(ValueError):
            npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="BSND")

    def test_input_layout_is_TND(self):
        with self.assertRaises(ValueError):
            npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="TND")

    def test_broadcast_kv(self):
        # valid input
        num_heads = 4
        num_kv_heads = 2
        kv_tensor = torch.randn(1, num_kv_heads, 10, 10)
        result = broadcast_kv(num_heads, num_kv_heads, kv_tensor, torch.float32)
        self.assertEqual(result.shape, (1, num_heads, 10, 10))

        # invalid_input
        num_heads = 4
        kv_tensor = torch.randn(1, num_kv_heads, 10, 10)
        num_kv_heads = 0
        with pytest.raises(ValueError):
            broadcast_kv(num_heads, num_kv_heads, kv_tensor, torch.float32)
        num_kv_heads = 5
        with pytest.raises(ValueError):
            broadcast_kv(num_heads, num_kv_heads, kv_tensor, torch.float32)

        # num_heads equals to num_kv_heads
        num_heads = 4
        num_kv_heads = 4
        kv_tensor = torch.randn(1, num_kv_heads, 10, 10)
        result = broadcast_kv(num_heads, num_kv_heads, kv_tensor, torch.float32)

        self.assertEqual(result.shape, (1, num_heads, 10, 10))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result, kv_tensor.expand(-1, num_heads, -1, -1)))
