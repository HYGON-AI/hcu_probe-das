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


def matmul_backward(grad, self, other, mask):
    if len(mask) < 2:
        raise RuntimeError("Mask size at least 2")

    grad_self, grad_other = None, None
    dim_self = self.dim()
    dim_other = other.dim()

    size_grad = list(grad.size())
    size_self = list(self.size())
    size_other = list(other.size())

    if dim_self == 1 and dim_other == 1:
        grad_self = other.mul(grad) if mask[0] else grad_self
        grad_other = self.mul(grad) if mask[1] else grad_other
    elif dim_self == 2 and dim_other == 1:
        grad_self = grad.unsqueeze(1).mm(other.unsqueeze(0)) if mask[0] else grad_self
        grad_other = self.transpose(-1, -2).mm(grad.unsqueeze(1)).squeeze_(1) if mask[1] else grad_other
    elif dim_self == 1 and dim_other == 2:
        grad_self = grad.unsqueeze(0).mm(other.transpose(-1, -2)).squeeze_(0) if mask[0] else grad_self
        grad_other = self.unsqueeze(1).mm(grad.unsqueeze(0)) if mask[1] else grad_other
    elif dim_self >= 3 and (dim_other == 1 or dim_other == 2):
        if len(size_grad) < 1:
            raise RuntimeError("size_grad's length at least 1")
        view_size = 1 if dim_other == 1 else size_grad[-1]
        unfolded_grad = (grad.unsqueeze(-1) if dim_other == 1 else grad).contiguous().view(-1, view_size)
        if mask[0]:
            grad_self = unfolded_grad.mm(other.unsqueeze(0) if dim_other == 1 else other.transpose(-1, -2)) \
                .view(size_self)
        if mask[1]:
            if len(size_self) < 1:
                raise RuntimeError("size_self's length at least 1")
            unfolded_self = self.contiguous().view([-1, size_self[-1]])
            grad_other = unfolded_self.transpose(-1, -2).mm(unfolded_grad).view(size_other)
    elif (dim_self == 1 or dim_self == 2) and dim_other >= 3:
        if len(size_grad) < 2:
            raise RuntimeError("size_grad's length at least 2")
        view_size = 1 if dim_self == 1 else size_grad[-2]
        unfolded_grad_t = grad.view([-1, view_size]) \
            if dim_self == 1 else grad.transpose(-1, -2).contiguous().view([-1, view_size])
        if mask[0]:
            if len(size_other) < 2:
                raise RuntimeError("size_other's length at least 2")
            # create a 2D-matrix from other
            unfolded_other_t = \
                other.transpose(-1, -2).contiguous().view([-1, size_other[-2]]).transpose(-1, -2)
            grad_self = unfolded_other_t.mm(unfolded_grad_t).transpose(-1, -2).view(size_self)
        if mask[1]:
            size_other_t = size_other[:-2]
            size_other_t.extend(size_other[::-1][:2])
            grad_other = \
                unfolded_grad_t.mm(self.unsqueeze(0) if dim_self == 1 else self).view(size_other_t).transpose(-1, -2)
    else:
        grad_self = torch.matmul(grad, other.transpose(-1, -2)) if mask[0] else grad_self
        grad_other = torch.matmul(self.transpose(-1, -2), grad) if mask[1] else grad_other

    return grad_self.cpu(), grad_other.cpu()
