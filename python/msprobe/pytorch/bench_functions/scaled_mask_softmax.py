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

import torch


def npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask):
    if fixed_triu_mask:
        mask = (torch.triu(torch.ones(mask.shape), k=1)).bool().to(mask.device)
    dtype = x.dtype
    x = (x * scale).masked_fill(mask, value=-10000)
    x = x - torch.max(x, dim=-1, keepdims=True)[0]
    x = torch.exp(x.float())
    y = torch.div(x, torch.sum(x, dim=-1, keepdims=True))
    return y.to(dtype)


def npu_scaled_masked_softmax_backward(y_grad, y, mask, scale, fixed_triu_mask):
    if fixed_triu_mask:
        mask = (torch.triu(torch.ones(mask.shape), k=1)).bool().to(mask.device)
    dtype = y_grad.dtype
    y_grad = y_grad.float()
    y = y.float()
    x_grad = y_grad * y
    x_grad = y_grad - torch.sum(x_grad, dim=-1, keepdims=True)
    x_grad = x_grad * y
    x_grad = x_grad * scale
    x_grad = x_grad.masked_fill(mask, value=0)
    return x_grad.to(dtype).cpu()
