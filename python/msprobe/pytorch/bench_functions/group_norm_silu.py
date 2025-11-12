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


def npu_group_norm_silu(x, gama, beta, group, eps):
    if len(x.shape) != 4:
        raise ValueError("x shape should be (N, C, H, W)")
    res = torch.ops.aten.native_group_norm(x, gama, beta, x.shape[0], x.shape[1], x.shape[2] * x.shape[3], group, eps)
    res = list(res)
    if not res:
        raise ValueError("run native_group_norm failed")
    res[0] = torch.nn.functional.silu(res[0])
    return res
