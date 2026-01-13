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


def npu_rms_norm(x, gamma, epsilon=1e-5):
    rstd = torch.rsqrt(torch.mean(torch.pow(x, 2), axis=-1, keepdim=True) + epsilon)
    res = x * rstd * gamma
    return res, rstd.float()


def npu_rms_norm_backward(grad, x, gamma, rstd):
    mean_gy = (grad * x * gamma * rstd).mean(dim=-1, keepdim=True)
    grad_x = (grad * gamma - x * rstd * mean_gy) * rstd
    grad_gamma = x * grad * rstd
    return grad_x.cpu(), grad_gamma.cpu()

