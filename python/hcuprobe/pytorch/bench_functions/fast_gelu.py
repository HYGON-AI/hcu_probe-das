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


def npu_fast_gelu(input0):
    attr = 1.702
    const_0 = 0 - attr
    const_1 = 1
    const_2 = attr / 2

    abs_x = torch.abs(input0)
    mul_abs_x = abs_x * const_0
    exp_abs_x = torch.exp(mul_abs_x)
    div_down = exp_abs_x + const_1

    pn_x = input0 - abs_x
    mul_pn_x = pn_x * const_2
    exp_pn_x = torch.exp(mul_pn_x)
    div_up = input0 * exp_pn_x
    div_down_rec = torch.reciprocal(div_down)
    result = div_up * div_down_rec

    return result


def npu_fast_gelu_backward(grad, input_x):
    const_2 = 1.702
    const_3 = 1.0
    const_1 = 0.0 - const_2

    # e^(-1.702x)
    abs_x = torch.abs(input_x)
    mul_abs_x = abs_x * const_1
    exp_x = torch.exp(mul_abs_x)

    # 1.702xe^(-1.702x)
    add_2 = input_x * exp_x
    add_2 = add_2 * const_2

    # e^(1.702(x-|x|))
    pn_x = input_x - abs_x
    mul_pn_x = pn_x * const_2
    exp_pn_x = torch.exp(mul_pn_x)

    #  e^(-1.702x) + 1.702xe^(-1.702x) + e^(1.702(x-|x|))
    div_up = exp_x + add_2
    div_up = div_up + exp_pn_x

    # (e^(-1.702x)+1)^2
    div_down_i = exp_x + const_3
    div_down = div_down_i * div_down_i
    div_down_rec = torch.reciprocal(div_down)
    result_temp = div_up * div_down_rec
    result = grad * result_temp

    return result.cpu()
