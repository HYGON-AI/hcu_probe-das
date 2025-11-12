# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

"""
# 前向函数声明对比
标杆实现:fusion_attention_forward: q, k, v, drop_mask, atten_mask, pse, scale, keep_prob
融合算子:npu_fusion_attention_forward: query, key, value, head_num, input_layout, *, pse=None, padding_mask=None,
                                      atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647,
                                      next_tockens=2147483647, inner_precise=0, prefix=None, sparse_mode=0,
                                      gen_mask_parallel=True, sync=False

# 反向函数声明对比
标杆实现:fusion_attention_backward: dx, q, k, v, softmax_res, drop_mask, pse, scale, keep_prob
融合算子:npu_fusion_attention_backward: query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None,
                                       atten_mask=None, softmax_max=None, softmax_sum=None, softmax_in=None,
                                       attention_in=None, scale_value=1.0, keep_prob=1.0, pre_tockens=2147483647,
                                       next_tockens=2147483647, inner_precise=0, seed=0, offset=0,
                                       numels=0, prefix=None, sparse_mode=0, gen_mask_parallel=True, sync=False
"""

from collections import namedtuple
import torch
import numpy as np
from einops import rearrange

try:
    import torch_npu
except ImportError:
    is_gpu = True
    try:
        # flash_attn为gpu的fa三方库
        from flash_attn import flash_attn_func
    except ImportError:
        # 如果为cpu的ut环境，则不做任何处理
        pass
else:
    is_gpu = False

from msprobe.pytorch.common.utils import logger
from msprobe.core.common.const import Const, CompareConst

GTYPE = torch.float64  # arm host必须选择float64，x86环境选择float32即可，64也行。arm计算很慢，s=8k的场景建议使用x86
SOFTMAX_BUILD_MODE = "QKV"  # "MAX_SUM"


FaForwardParams = namedtuple("FaForwardParams",
                            ["q", "k", "v", "drop_mask", "atten_mask", "pse", "scale", "keep_prob"])
FaBackwardParams = namedtuple("FaBackwardParams",
                            ["dx", "q", "k", "v", "softmax_res", "drop_mask", "pse", "scale", "keep_prob"])
RebuildSoftmaxParams = namedtuple("RebuildSoftmaxParams",
                                ["q", "k", "atten_mask", "pse", "scale", "softmax_max", "softmax_sum"])


def softmax_forward(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    res = y.div(x_sum)
    return res, x_max, x_sum


def softmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res


def broadcast_kv(num_heads, num_kv_heads, kv_tensor, dtype):
    if num_kv_heads == 0 or num_kv_heads > num_heads:
        raise ValueError(f"num_kv_heads must be non-zero and bigger than num_heads.")

    factor = num_heads // num_kv_heads
    kv_shape = kv_tensor.shape
    b = kv_shape[0]
    s = kv_shape[2]
    d = kv_shape[3]
    kv_res = torch.zeros([b, num_heads, s, d]).to(dtype)
    for i in range(num_heads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res


def calculate_qk(q, k, atten_mask, pse, scale):
    if pse is None or len(pse.shape) == 0:
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(scale)
    else:
        qk = (torch.matmul(q, k.permute(0, 1, 3, 2)) + pse).mul(scale)
    if atten_mask is None or len(atten_mask.shape) == 0:
        return qk
    else:
        qk = qk + atten_mask.bool() * (-40000.0)  # -10000
    return qk


def fusion_attention_forward(forward_params):
    q = forward_params.q
    k = forward_params.k
    v = forward_params.v
    drop_mask = forward_params.drop_mask
    atten_mask = forward_params.atten_mask
    pse = forward_params.pse
    scale = forward_params.scale
    keep_prob = forward_params.keep_prob

    # 除零风险拦截：keep_prob 为 0 时会导致除零错误
    if keep_prob == 0:
        raise ValueError("fusion_attention_forward: keep_prob cannot be zero to avoid division by zero.")


    qk = calculate_qk(q, k, atten_mask, pse, scale)
    softmax_res, softmax_max, softmax_sum = softmax_forward(qk)
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res
    else:
        drop_res = softmax_res * drop_mask * (1.0 / keep_prob)
    y = torch.matmul(drop_res, v)
    return y, softmax_max, softmax_sum


def fusion_attention_backward(backward_params):
    dx = backward_params.dx
    q = backward_params.q
    k = backward_params.k
    v = backward_params.v
    softmax_res = backward_params.softmax_res
    drop_mask = backward_params.drop_mask
    pse = backward_params.pse
    scale = backward_params.scale
    keep_prob = backward_params.keep_prob

    # 除零风险拦截：keep_prob 为 0 时会导致除零错误
    if keep_prob == 0:
        raise ValueError("fusion_attention_backward: keep_prob cannot be zero to avoid division by zero.")

    dp = torch.matmul(dx, v.permute(0, 1, 3, 2))
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res.permute(0, 1, 3, 2)
        dp_drop = dp
    else:
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / keep_prob).permute(0, 1, 3, 2)
        dp_drop = dp * drop_mask * (1.0 / keep_prob)
    dv = torch.matmul(drop_res, dx)
    softmax_grad_res = (softmax_grad(dp_drop, softmax_res) * scale)
    dq = torch.matmul(softmax_grad_res, k)
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 3, 2), q)
    return dq, dk, dv


def parse_bsnd_args(query, key, head_num, input_layout):
    supported_input_layout = ["BSH", "SBH", "BSND", "BNSD", "TND"]
    b, s1, s2, n1, n2, d, h1, h2 = None, None, None, head_num, None, None, None, None

    if not isinstance(input_layout, str) or input_layout not in supported_input_layout:
        raise ValueError(f"Invalid input_layout arg which must be one of {supported_input_layout}.")

    if input_layout == "TND":
        raise ValueError(f"input_layout {input_layout} does not supported for now.")
    try:
        if input_layout == "BSH":
            b, s1, h1 = query.shape
            _, s2, h2 = key.shape
            if n1 == 0:
                raise ValueError("parse_bsnd_args: head_num (n1) cannot be zero to avoid division by zero.")
            d = h1 // n1
            if d == 0:
                raise ValueError("parse_bsnd_args: computed head dimension (d) is zero, division by zero risk.")
            n2 = h2 // d
        elif input_layout == "SBH":
            s1, b, h1 = query.shape
            s2, _, h2 = key.shape
            if n1 == 0:
                raise ValueError("parse_bsnd_args: head_num (n1) cannot be zero to avoid division by zero.")
            d = h1 // n1
            if d == 0:
                raise ValueError("parse_bsnd_args: computed head dimension (d) is zero, division by zero risk.")
            n2 = h2 // d
        elif input_layout == "BSND":
            b, s1, n1, d = query.shape
            _, s2, n2, _ = key.shape
            h1 = n1 * d
            h2 = n2 * d
            if d == 0:
                raise ValueError("parse_bsnd_args: head dimension (d) is zero, division by zero risk.")
        elif input_layout == "BNSD":
            b, n1, s1, d = query.shape
            _, n2, s2, _ = key.shape
            h1 = n1 * d
            h2 = n2 * d
            if d == 0:
                raise ValueError("parse_bsnd_args: head dimension (d) is zero, division by zero risk.")
    except Exception as e:
        raise ValueError(f"query.shape: {query.shape}, key.shape: {key.shape}, parse_bsnd_args error: {e}") from e

    if d == 0:
        raise ValueError(f"Value d must be non-zero.")
    _dtype = query.dtype
    ret = (b, s1, s2, n1, n2, d, h1, h2, _dtype)
    return ret


def convert_from_bnsd(_input, input_layout):
    """
    transform qkv from bnsd to input_layout.
    B: batch_size
    S: sequence_length
    N: num_heads
    D: head_dim
    Args:
       _input (torch.Tensor): tensor of shape (B,N,S,D)
        input_layout (str): "BSH" or "SBH" or "BSND" or "BNSD" or "TND"
    Returns:
        tensor of shape (B,N,S,D) or (B,S,N,D) or (S,B,H) or (B,S,H)
    """
    if input_layout == "BSH":
        # (B,N,S,D)=>(B,S,N*D)
        out = rearrange(_input, 'b n s d -> b s (n d)').contiguous()
    elif input_layout == "SBH":
        # (B,N,S,D)=>(S,B,N*D)
        out = rearrange(_input, 'b n s d -> s b (n d)').contiguous()
    elif input_layout == "BSND":
        # (B,N,S,D)=>(B,S,N,D)
        out = rearrange(_input, 'b n s d -> b s n d').contiguous()
    elif input_layout == "TND":
        raise ValueError(f"input_layout {input_layout} does not supported for now.")
    else:
        out = _input
    return out


def convert_to_bnsd(_input, n, input_layout):
    """
    transform qkv from input_layout to bnsd.
    B: batch_size
    S: sequence_length
    N: num_heads
    D: head_dim
    Args:
        _input (torch.Tensor): tensor of shape (B,N,S,D) or (B,S,N,D) or (S,B,H) or (B,S,H)
        n (int): num_heads
        input_layout (str):"BSH" or "SBH" or "BSND" or "BNSD" or "TND"
    Returns:
        tensor of shape (B,N,S,D)
    """
    if input_layout == "BSH":
        # (B,S,N*D)=>(B,N,S,D)
        out = rearrange(_input, 'b s (n d) -> b n s d', n=n)
    elif input_layout == "SBH":
        # (S,B,N*D)=>(B,N,S,D)
        out = rearrange(_input, 's b (n d) -> b n s d', n=n)
    elif input_layout == "BSND":
        # (B,S,N,D)=>(B,N,S,D)
        out = rearrange(_input, 'b s n d -> b n s d', n=n)
    elif input_layout == "TND":
        raise ValueError(f"input_layout {input_layout} does not supported for now.")
    else:
        out = _input
    if out.dim() != 4:
        raise ValueError(f"convert qkv format failed with input_layout {input_layout}.")
    return out.to(GTYPE)


def convert_from_bsnd(_input, input_layout):
    """
    transform qkv from bsnd to input_layout.
    B: batch_size
    S: sequence_length
    N: num_heads
    D: head_dim
    Args:
       _input (torch.Tensor): tensor of shape (B,S,N,D)
        input_layout (str): "BSH" or "SBH" or "BSND" or "BNSD" or "TND"
    Returns:
        tensor of shape (B,N,S,D) or (B,S,N,D) or (S,B,H) or (B,S,H)
    """
    if input_layout == "BSH":
        # (B,S,N,D)=>(B,S,N*D)
        out = rearrange(_input, 'b s n d -> b s (n d)').contiguous()
    elif input_layout == "SBH":
        # (B,S,N,D)=>(S,B,N*D)
        out = rearrange(_input, 'b s n d -> s b (n d)').contiguous()
    elif input_layout == "BNSD":
        # (B,S,N,D)=>(B,N,S,D)
        out = rearrange(_input, 'b s n d -> b n s d').contiguous()
    elif input_layout == "TND":
        raise ValueError(f"input_layout {input_layout} does not supported for now.")
    else:
        out = _input
    return out


def convert_to_bsnd(_input, n, input_layout):
    """
    transform qkv from input_layout to bsnd.
    B: batch_size
    S: sequence_length
    N: num_heads
    D: head_dim
    Args:
        _input (torch.Tensor): tensor of shape (B,N,S,D) or (B,S,N,D) or (S,B,H) or (B,S,H)
        n (int): num_heads
        input_layout (str):"BSH" or "SBH" or "BSND" or "BNSD" or "TND"
    Returns:
        tensor of shape (B,S,N,D)
    """
    if input_layout == "BSH":
        # (B,S,N*D)=>(B,S,N,D)
        out = rearrange(_input, 'b s (n d) -> b s n d', n=n)
    elif input_layout == "SBH":
        # (S,B,N*D)=>(B,S,N,D)
        out = rearrange(_input, 's b (n d) -> b s n d', n=n)
    elif input_layout == "BNSD":
        # (B,N,S,D)=>(B,S,N,D)
        out = rearrange(_input, 'b n s d -> b s n d', n=n)
    elif input_layout == "TND":
        raise ValueError(f"input_layout {input_layout} does not supported for now.")
    else:
        out = _input
    if out.dim() != 4:
        raise ValueError(f"convert qkv format failed with input_layout {input_layout}.")
    return out


def generate_atten_mask(*args):
    """
    # 当sparse_mode=2、3、4时小算子到融合算子会走这个优化，反过来看就要拆解回原来的基本实现
    ===> atten_mask = torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1)).to(dtype)
    """

    sparse_mode, atten_mask, b, n1, s1, s2, pre_tocken, next_tocken, dtype = args
    shape = [s1, s2]

    if atten_mask is not None:
        # 当FA的输入已经包含atten_mask时，可以认为已经是转换之后的mask矩阵了，有三种特殊场景，即稀疏矩阵场景，需要进行逆向还原
        if sparse_mode == 2 or sparse_mode == 3 or sparse_mode == 4:
            logger.info(f"s1: {s1}, s2:{s2}, atten_mask.shape:{atten_mask.shape}, atten_mask.dtype:{atten_mask.dtype}")

            if atten_mask.dim() == 2 and atten_mask.shape[0] == 2048 and atten_mask.shape[1] == 2048:
                if atten_mask.equal(torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1)).to(atten_mask.dtype)):
                    if sparse_mode == 2:
                        atten_mask = torch.from_numpy(np.triu(np.ones(shape), k=1))
                    elif sparse_mode == 3:
                        atten_mask = torch.from_numpy(np.triu(np.ones(shape), k=s2 - s1 + 1))
                    elif sparse_mode == 4:
                        atten_mask_u = torch.from_numpy(np.triu(np.ones(shape), k=next_tocken + 1))
                        atten_mask_l = torch.from_numpy(np.tril(np.ones(shape), k=-pre_tocken - 1))
                        atten_mask = atten_mask_u + atten_mask_l
                    logger.debug(f"反向转换atten_mask {atten_mask.shape}")
                    return atten_mask.to(dtype)

        return atten_mask.to(dtype)

    if atten_mask is not None:
        if atten_mask.dim() == 2:
            if atten_mask.shape[0] != s1 or atten_mask.shape[1] != s2:
                raise ValueError(f"Invalid atten_mask shape `SS` {atten_mask.shape}")
            shape = [s1, s2]
        elif atten_mask.dim() == 4:
            if atten_mask.shape[1] == 1:
                shape = [b, 1, s1, s2] if b != 1 else [1, 1, s1, s2]
            else:
                shape = [b, n1, s1, s2] if b != 1 else [1, n1, s1, s2]

    if sparse_mode == 0:
        atten_mask_u = torch.from_numpy(np.triu(np.ones(shape), k=next_tocken + 1))
        atten_mask_l = torch.from_numpy(np.tril(np.ones(shape), k=-pre_tocken - 1))
        atten_mask = atten_mask_u + atten_mask_l
    elif sparse_mode == 1:  # no sparse
        atten_mask = torch.from_numpy(np.zeros(shape))
    elif sparse_mode == 2:
        atten_mask = torch.from_numpy(np.triu(np.ones(shape), k=1))
    elif sparse_mode == 3:
        atten_mask = torch.from_numpy(np.triu(np.ones(shape), k=s2 - s1 + 1))
    elif sparse_mode == 4:
        atten_mask_u = torch.from_numpy(np.triu(np.ones(shape), k=next_tocken + 1))
        atten_mask_l = torch.from_numpy(np.tril(np.ones(shape), k=-pre_tocken - 1))
        atten_mask = atten_mask_u + atten_mask_l
    # 注:不会出现sparse_mode=5的情况，该情况要求必须要传入atten_mask，且atten_mask矩阵数据格式须为BNSS或B1SS，
    # 因此可以认为FA的输入已经是正确的atten_mask了
    return atten_mask.to(dtype)


def generate_kv(key, value, n1, n2):
    # N不等长适配by cdy
    if not (n1 == n2):
        k_new = broadcast_kv(n1, n2, key, key.dtype)
        v_new = broadcast_kv(n1, n2, value, value.dtype)
    else:
        k_new = key
        v_new = value
    return k_new, v_new


def rebuid_softmax_by_qkv(q, k, atten_mask, pse, scale):
    """
    attention = softmax(QK^T/sqrt(d))V
    softmax(x_i) = e^(x_i - x_max) / sum(e^(x_i - x_max))
    """
    logger.info("Using QKV to rebuild original softmax")
    qk = calculate_qk(q, k, atten_mask, pse, scale)
    softmax_res, _, _ = softmax_forward(qk)
    return softmax_res


def rebuild_softmax_by_max_sum(softmax_params):
    """
    attention = softmax(QK^T/sqrt(d))V
    softmax(x_i) = e^(x_i - x_max_i) / x_sum_i)
    """
    q = softmax_params.q
    k = softmax_params.k
    atten_mask = softmax_params.atten_mask
    pse = softmax_params.pse
    scale = softmax_params.scale
    softmax_max = softmax_params.softmax_max
    softmax_sum = softmax_params.softmax_sum
    logger.info("Using softmax_max and softmax_sum to rebuild original softmax")
    qk = calculate_qk(q, k, atten_mask, pse, scale)
    if softmax_max.shape[-1] == 0:
        raise ValueError(f"softmax_max.shape[-1] must be non-zero, softmax_max.shape: {softmax_max.shape}")
    repeat_dim = qk.shape[-1] // softmax_max.shape[-1]
    softmax_res = torch.exp(qk.sub(softmax_max.repeat(1, 1, 1, repeat_dim))).div(
        softmax_sum.repeat(1, 1, 1, repeat_dim))
    return softmax_res


def get_head_num(*args, **kwargs):
    if kwargs.get("head_num", None):
        head_num = kwargs.get("head_num")
    elif len(args) >= 4:
        head_num = args[3]
    else:
        raise ValueError(f"Unsupported npu_fusion_attention args {args}.")
    return head_num


def get_input_layout(*args, **kwargs):
    if kwargs.get("input_layout", None):
        input_layout = kwargs.get("input_layout")
    elif len(args) >= 5:
        input_layout = args[4]
    else:
        raise ValueError(f"Unsupported npu_fusion_attention args {args}.")
    return input_layout


def npu_fusion_attention_forward_patch(*args, **kwargs):

    if len(args) < 2:
        raise RuntimeError("npu_fusion_attention_forward_patch: length of args should greater than or equal to 2.")

    # query, key, value, head_num, input_layout
    head_num = get_head_num(*args, **kwargs)
    input_layout = get_input_layout(*args, **kwargs)

    b, s1, s2, n1, n2, d, h1, h2, dtype = parse_bsnd_args(args[0], args[1], head_num, input_layout)
    if d == 0:
        raise ValueError("npu_fusion_attention_forward_patch: head dimension (d) is zero, division by zero risk.")
    if n1 == n2 and s1 == s2:
        logger.debug(f"running case : BNSD = {b}_{n1}_{s1}_{d}, sparse = {kwargs.get('sparse_mode', 0)}")
    else:
        logger.debug(f"running case: BNSD = {b}_{n1}({n2})_{s1}({s2})_{d}, sparse = {kwargs.get('sparse_mode', 0)}")
    if not (n1 % n2 == 0 and n1 >= n2):
        raise ValueError(f"N1与N2不匹配,请检查: n1 = {n1}, n2 = {n2}.")

    dims_kwargs = {
        "b": b, "s1": s1, "s2": s2, "n1": n1, "n2": n2,
        "d": d, "h1": h1, "h2": h2, "dtype": dtype
    }

    new_kwargs = {
        "keep_prob": 1,
        "scale": kwargs.get("scale", 1 / (d ** 0.5)),
        "sparse_mode": kwargs.get("sparse_mode", 0),
        "prefix": kwargs.get("prefix"),
        "pre_tockens": kwargs.get("pre_tockens", 2147483647),
        "next_tockens": kwargs.get("next_tockens", 2147483647),
        "pse": kwargs.get("pse"),
        "padding_mask": kwargs.get("padding_mask"),
        "atten_mask": kwargs.get("atten_mask")
    }

    return args, dims_kwargs, new_kwargs


def npu_fusion_attention_backward_patch(*args, **kwargs):
    if len(args) != 6:
        raise ValueError(f"Unsupported npu_fusion_attention_grad args {args}.")

    b, s1, s2, n1, n2, d, h1, h2, dtype = parse_bsnd_args(args[0], args[1], args[4], args[5])
    if d == 0:
        raise ValueError("npu_fusion_attention_backward_patch: head dimension (d) is zero, division by zero risk.")
    if n1 == n2 and s1 == s2:
        logger.info(f"running case : bnsd = {b}_{n1}_{s1}_{d}, sparse = {kwargs.get('sparse_mode', 0)}")
    else:
        logger.info(f"running case: bnsd = {b}_{n1}({n2})_{s1}({s2})_{d}, sparse = {kwargs.get('sparse_mode', 0)}")
    if not (n1 % n2 == 0 and n1 >= n2):
        raise ValueError(f"N1与N2不匹配,请检查: n1 = {n1}, n2 = {n2}.")

    dims_kwargs = {
        "b": b, "s1": s1, "s2": s2, "n1": n1, "n2": n2,
        "d": d, "h1": h1, "h2": h2, "dtype": dtype
    }

    new_kwargs = {
        "keep_prob": 1,
        "scale_value": kwargs.get("scale_value", 1 / (d ** 0.5)),
        "sparse_mode": kwargs.get("sparse_mode", 0),
        "prefix": kwargs.get("prefix"),
        "pre_tockens": kwargs.get("pre_tockens", 2147483647),
        "next_tockens": kwargs.get("next_tockens", 2147483647),
        "pse": kwargs.get("pse"),
        "padding_mask": kwargs.get("padding_mask"),
        "softmax_max": kwargs.get("softmax_max"),
        "softmax_sum": kwargs.get("softmax_sum"),
        "softmax_in": kwargs.get("softmax_in"),
        "attention_in": kwargs.get("attention_in"),
        "seed": kwargs.get("seed", 0),
        "offset": kwargs.get("offset", 0),
        "numels": kwargs.get("numels", 0),
        "atten_mask": kwargs.get("atten_mask")
    }

    return args, dims_kwargs, new_kwargs


def npu_fusion_attention(*args, **kwargs):
    new_args, dims_kwargs, new_kwargs = npu_fusion_attention_forward_patch(*args, **kwargs)
    query, key, value = new_args[0], new_args[1], new_args[2]
    input_layout = get_input_layout(*args, **kwargs)
    n1 = dims_kwargs.get("n1")
    n2 = dims_kwargs.get("n2")
    s1 = dims_kwargs.get("s1")
    s2 = dims_kwargs.get("s2")
    b = dims_kwargs.get("b")
    dtype = dims_kwargs.get("dtype")
    atten_mask = new_kwargs.get("atten_mask")
    keep_prob = new_kwargs.get("keep_prob")
    sparse_mode = new_kwargs.get("sparse_mode")
    pre_tockens = new_kwargs.get("pre_tockens")
    next_tockens = new_kwargs.get("next_tockens")
    pse = new_kwargs.get("pse")
    scale = new_kwargs.get("scale")
    args_temp = [sparse_mode, atten_mask, b, n1, s1, s2, pre_tockens, next_tockens, dtype]
    atten_mask = generate_atten_mask(*args_temp)
    query = convert_to_bnsd(query, n1, input_layout)
    key = convert_to_bnsd(key, n2, input_layout)
    value = convert_to_bnsd(value, n2, input_layout)
    k_new, v_new = generate_kv(key, value, n1, n2)
    forward_params = FaForwardParams(query, k_new, v_new, None, atten_mask, pse, scale, keep_prob)
    out_golden, softmax_max, softmax_sum = fusion_attention_forward(forward_params)
    if out_golden.dim() == 5:
        out_golden = out_golden.reshape(out_golden.size(0), out_golden.size(1) * out_golden.size(2), out_golden.size(3),
                                        out_golden.size(4))
    out_golden = convert_from_bnsd(out_golden, input_layout)

    return out_golden.cpu(), softmax_max.repeat(1, 1, 1, 8).cpu(), softmax_sum.repeat(1, 1, 1, 8).cpu()


def npu_fusion_attention_grad(*args, **kwargs):
    # dx, q, k, v, softmax_res, drop_mask, pse, scale, keep_prob
    new_args, dims_kwargs, new_kwargs = npu_fusion_attention_backward_patch(*args, **kwargs)
    query, key, value, dx, input_layout = new_args[0], new_args[1], new_args[2], new_args[3], new_args[5]
    n1 = dims_kwargs.get("n1")
    n2 = dims_kwargs.get("n2")
    s1 = dims_kwargs.get("s1")
    s2 = dims_kwargs.get("s2")
    b = dims_kwargs.get("b")
    d = dims_kwargs.get("d")
    dtype = dims_kwargs.get("dtype")
    atten_mask = new_kwargs.get("atten_mask")
    keep_prob = new_kwargs.get("keep_prob")
    sparse_mode = new_kwargs.get("sparse_mode")
    pre_tockens = new_kwargs.get("pre_tockens")
    next_tockens = new_kwargs.get("next_tockens")
    pse = new_kwargs.get("pse")
    softmax_max = new_kwargs.get("softmax_max")
    softmax_sum = new_kwargs.get("softmax_sum")
    scale_value = new_kwargs.get("scale_value")

    args_temp = [sparse_mode, atten_mask, b, n1, s1, s2, pre_tockens, next_tockens, dtype]
    atten_mask = generate_atten_mask(*args_temp)
    query = convert_to_bnsd(query, n1, input_layout)
    dx = convert_to_bnsd(dx, n1, input_layout)
    key = convert_to_bnsd(key, n2, input_layout)
    value = convert_to_bnsd(value, n2, input_layout)
    k_new, v_new = generate_kv(key, value, n1, n2)

    if SOFTMAX_BUILD_MODE == "QKV":
        softmax_res = rebuid_softmax_by_qkv(query, k_new, atten_mask, pse, scale_value)
    else:
        softmax_params = RebuildSoftmaxParams(query, k_new, atten_mask, pse, scale_value, softmax_max, softmax_sum)
        softmax_res = rebuild_softmax_by_max_sum(softmax_params)
    backward_params = FaBackwardParams(dx, query, k_new, v_new, softmax_res, None, pse, scale_value, keep_prob)
    dq, dk, dv = fusion_attention_backward(backward_params)

    # N不等长适配by cdy
    if not (n1 == n2):
        if n2 == 0:
            raise ValueError("dims_kwargs.n2 must be non-zero.")
        g = int(n1 / n2)
        dk = torch.sum(dk.reshape(b, n2, g, s2, d), dim=2, keepdim=True).reshape(b, n2, s2, d)
        dv = torch.sum(dv.reshape(b, n2, g, s2, d), dim=2, keepdim=True).reshape(b, n2, s2, d)

    if dq.dim() == 5:
        dq = dq.reshape(dq.size(0), dq.size(1) * dq.size(2), dq.size(3), dq.size(4))
    if dk.dim() == 5:
        dk = dk.reshape(dk.size(0), dk.size(1) * dk.size(2), dk.size(3), dk.size(4))
    if dv.dim() == 5:
        dv = dv.reshape(dv.size(0), dv.size(1) * dv.size(2), dv.size(3), dv.size(4))

    dq = convert_from_bnsd(dq, input_layout)
    dk = convert_from_bnsd(dk, input_layout)
    dv = convert_from_bnsd(dv, input_layout)

    return dq.cpu(), dk.cpu(), dv.cpu()


def is_attention_off_due_to_mask(atten_mask_dtype):
    return not atten_mask_dtype


def is_attention_off_in_sparse_mode_4(sparse_mode, next_tockens, pre_tockens, s1):
    return sparse_mode == 4 and (next_tockens != 0 or pre_tockens < s1)


def is_attention_off_in_sparse_mode_0(sparse_mode, pre_tockens, next_tockens, s1, s2):
    return sparse_mode == 0 and pre_tockens >= s1 and next_tockens >= s2


def gpu_fusion_attention(*args, **kwargs):
    deterministic = False
    new_args, dims_kwargs, new_kwargs = npu_fusion_attention_forward_patch(*args, **kwargs)
    query, key, value = new_args[0], new_args[1], new_args[2]
    keep_prob = new_kwargs.get("keep_prob", 1.0)
    scale = new_kwargs.get("scale")
    n1 = dims_kwargs.get("n1")
    n2 = dims_kwargs.get("n2")
    s1 = dims_kwargs.get("s1")
    s2 = dims_kwargs.get("s2")
    b = dims_kwargs.get("b")
    pse = new_kwargs.get("pse")
    sparse_mode = new_kwargs.get("sparse_mode")
    pre_tockens = new_kwargs.get("pre_tockens")
    next_tockens = new_kwargs.get("next_tockens")
    attn_mask = new_kwargs.get("atten_mask")
    atten_mask_dtype = attn_mask.dtype if new_kwargs.get("atten_mask") is not None else None
    pre_tockens = min(CompareConst.MAX_TOKENS, pre_tockens)
    next_tockens = min(CompareConst.MAX_TOKENS, next_tockens)
    atten_off = (is_attention_off_due_to_mask(atten_mask_dtype) or
                 is_attention_off_in_sparse_mode_4(sparse_mode, next_tockens, pre_tockens, s1) or
                 is_attention_off_in_sparse_mode_0(sparse_mode, pre_tockens, next_tockens, s1, s2))
    causal_switch = not atten_off
    if sparse_mode == CompareConst.SPECIAL_SPARSE_MOED:
        window_left = pre_tockens
        window_right = next_tockens
    else:
        pre_tockens = next_tockens = CompareConst.MAX_TOKENS
        window_left = pre_tockens - s1 + s2
        window_right = next_tockens + s1 - s2

    if pse is not None:
        alibi_slopes = torch.rand(b, n1, dtype=torch.float32) * 0.3
    else:
        alibi_slopes = None

    input_layout = get_input_layout(*args, **kwargs)
    query = convert_to_bsnd(query, n1, input_layout)
    key = convert_to_bsnd(key, n2, input_layout)
    value = convert_to_bsnd(value, n2, input_layout)
    out = flash_attn_func(
        query, key, value, dropout_p=(1 - keep_prob), softmax_scale=scale, causal=causal_switch,
        window_size=(window_left, window_right), alibi_slopes=alibi_slopes, deterministic=deterministic
    )
    out = convert_from_bsnd(out, input_layout)
    return out, Const.NONE, Const.NONE
