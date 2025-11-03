# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

from mindspore import mint, ops, _no_grad
from mindspore import Tensor
from mindspore import dtype as mstype

from msprobe.core.common.log import logger


@_no_grad()
def square_sum(x: Tensor):
    return (x * x).sum()


@_no_grad()
def get_min(x: Tensor):
    return mint.min(x)


@_no_grad()
def get_mean(x: Tensor):
    return mint.mean(x.astype(mstype.float32))


@_no_grad()
def get_norm(x: Tensor):
    norm_func = mint.norm if hasattr(mint, "norm") else ops.norm
    return norm_func(x.astype(mstype.float32))


@_no_grad()
def get_max(x: Tensor):
    return mint.max(x)


@_no_grad()
def get_zeros(x: Tensor, eps: float):
    if x.numel() == 0:
        return Tensor(float('nan'))
    return mint.sum(mint.abs(x) < eps) / x.numel()


@_no_grad()
def get_nans(t):
    return ops.isnan(t.astype(mstype.float32)).sum()


def get_shape(t):
    return t.shape


def get_dtype(t):
    return t.dtype


FUNC_MAP = {
    "min": get_min,
    "max": get_max,
    "mean": get_mean,
    "norm": get_norm,
    "nans": get_nans,
    "zeros": get_zeros,
    "shape": get_shape,
    "dtype": get_dtype
}


def max_eigenvalue(input_tensor: Tensor, num_iterations=3):
    input_tensor = input_tensor.float()
    try:
        check_tensor_dim(input_tensor, 2)
    except (TypeError, ValueError) as e:
        logger.warning(f"calcute max eigenvalue failed, {e}")
        return Tensor(0)
    in_features = input_tensor.shape[1]
    u_tensor = ops.randn(in_features)
    u_norm = u_tensor.norm()
    if u_norm == 0:
        return Tensor(0)
    u_tensor /= u_tensor.norm()
    input_seq = ops.matmul(input_tensor.T, input_tensor)
    for _ in range(num_iterations):
        v_tensor = ops.matmul(input_seq, u_tensor)
        spectral_norm = ops.matmul(v_tensor.T, u_tensor)
        v_norm = v_tensor.norm()
        if v_norm > 0:
            u_tensor = v_tensor / v_norm
        else:
            spectral_norm = Tensor(0)
            break
    return spectral_norm.sqrt()


def check_tensor_dim(tensor, n):
    if not isinstance(tensor, Tensor):
        raise TypeError(
            f"Input must be a mindspore Tensor, but got {type(tensor)} instead."
            )
    if len(tensor.shape) < n:
        raise ValueError(
            f"tensor dim must be at least {n} dimensions."
            f"Got shape: {tuple(tensor.shape)} with {tensor.dim()} dims"
            )


def cal_entropy(qk_tensor: Tensor, mask=None):
    try:
        check_tensor_dim(qk_tensor, 2)
    except (TypeError, ValueError) as e:
        logger.warning(f"calculate entropy failed, {e}")
        return Tensor(0), Tensor(0)
    if mask is None:
        mask = ops.tril(ops.ones((qk_tensor.shape[1], qk_tensor.shape[1])))
    qk_tensor = qk_tensor - ops.amax(qk_tensor, axis=1, keepdims=True)
    qk_tensor = qk_tensor.masked_fill(mask == 0, float('-inf'))
    softmax_qkt = ops.softmax(qk_tensor.float(), axis=1)
    softmax_max = ops.mean(ops.amax(softmax_qkt, axis=1))
    entropy = ops.mean(-ops.nansum(softmax_qkt * ops.log(softmax_qkt), axis=1))
    return entropy, softmax_max


def cal_stable_rank(weight: Tensor):
    eig = max_eigenvalue(weight)
    if eig == Tensor(0):
        return Tensor(0), Tensor(0)
    f_norm = ops.norm(weight, ord='fro')
    return f_norm / eig, eig


def cal_qkt(q_h: Tensor, k_h: Tensor, order="s,b,h,d"):
    # q_h shape is (s, b, h, d)
    try:
        check_tensor_dim(q_h, 4)
        check_tensor_dim(k_h, 4)
    except (TypeError, ValueError) as e:
        logger.warning(f"calculatee qkt failed, {e}")
        return Tensor(0)
    if order == "s,b,h,d":
        qkt = ops.matmul(q_h[:, 0, 0, :], k_h[:, 0, 0, :].t()) / q_h.shape[-1] ** 0.5
    elif order == "b,s,h,d":
        qkt = ops.matmul(q_h[0, :, 0, :], k_h[0, :, 0, :].t()) / q_h.shape[-1] ** 0.5
    else:
        logger.warning(f"Calculate qk tensor failed: Order unsupported.")
        qkt = Tensor(0)
    return qkt
