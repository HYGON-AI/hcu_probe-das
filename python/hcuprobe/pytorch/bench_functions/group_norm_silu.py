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


def npu_group_norm_silu(x, gama, beta, group, eps):
    if len(x.shape) != 4:
        raise ValueError("x shape should be (N, C, H, W)")
    res = torch.ops.aten.native_group_norm(x, gama, beta, x.shape[0], x.shape[1], x.shape[2] * x.shape[3], group, eps)
    res = list(res)
    if not res:
        raise ValueError("run native_group_norm failed")
    res[0] = torch.nn.functional.silu(res[0])
    return res
