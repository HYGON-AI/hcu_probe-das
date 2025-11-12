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


def npu_apply_adam_w(beta1_power, beta2_power, lr, weight_decay,
                     beta1, beta2, eps, grad, max_grad_norm, amsgrad, maximize, out):
    var, m, v = out
    if amsgrad:
        max_grad_norm = (torch.rand(var.shape) * 10.0 - 5.0).to(var.dtype)
    beta1_power_out = beta1_power * beta1
    beta2_power_out = beta2_power * beta2
    var_t = var * (1 + (-lr * weight_decay))
    gt = -grad if maximize else grad
    m_out = m * beta1 - (beta1 + (-1)) * gt
    v_out = v * beta2 - (beta2 + (-1)) * gt * gt

    if amsgrad:
        max_grad_norm_out = torch.max(max_grad_norm, v_out)
        if (1 - beta2_power_out) == 0:
            beta2_power_out -= eps
        denom = torch.sqrt(torch.div(max_grad_norm_out, (1 - beta2_power_out))) + eps
    else:
        vraintain = torch.div(v_out, (1 - beta2_power_out))
        denom = torch.sqrt(vraintain) + eps

    if (1 - beta1_power_out) == 0:
        beta1_power_out -= eps
    var_out = var_t + torch.div(-lr * m_out, (1 - beta1_power_out)).div(denom)
    return var_out, m_out, v_out
