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
