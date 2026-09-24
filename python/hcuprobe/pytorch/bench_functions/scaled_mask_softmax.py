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
