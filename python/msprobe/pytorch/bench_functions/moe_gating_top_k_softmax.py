# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import torch
import numpy as np


def softmax_func(x, axis=None):
    x = x.float()
    x_max = x.max(dim=axis, keepdims=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=axis, keepdims=True)
    ans = 0 if (x_sum == 0).any() else y / x_sum
    return ans


def npu_moe_gating_top_k_softmax(x, finished_optional, k):
    input_dtype = x.dtype
    if x.dim() < 1:
        raise ValueError("Input x must have at least 1 dimensions.")
    num_expert = x.shape[-1]
    softmax = softmax_func(x, -1)
    softmax = softmax.to(input_dtype)
    expert_idx = torch.argsort(-softmax, dim=-1, stable=True)
    expert_idx = expert_idx[:, :k]
    y = torch.gather(softmax, index=expert_idx, dim=-1)
    if finished_optional is not None:
        if finished_optional.dim() < 1:
            raise ValueError("Finished_optional must have at least 1 dimensions.")
        finished_optional = finished_optional.view(finished_optional.shape[0], 1)
        finished_optional = finished_optional.expand(-1, k)
        expert_idx = torch.where(finished_optional, num_expert, expert_idx)
    if y.dim() < 2:
        raise ValueError("Variable y must have at least 2 dimensions.")
    row_idx = torch.arange(y.shape[0] * y.shape[1]).reshape(y.shape[1], y.shape[0]).t()

    return y, expert_idx, row_idx
