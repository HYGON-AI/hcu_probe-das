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

def npu_confusion_transpose(data, perm, shape, transpose_first):
    if transpose_first:
        output = data.permute(*perm).contiguous().view(shape)
    else:
        output = data.view(shape).permute(*perm)
    return output


def npu_confusion_transpose_backward(grad, perm, shape, transpose_first):
    try:
        shape_cal = shape if transpose_first else [shape[perm_dim] for perm_dim in perm]
    except IndexError as e:
        raise IndexError("npu_confusion_transpose_backward: Invalid perm index for shape") from e

    perm_cal = [0] * len(perm)
    for i, perm_dim in enumerate(perm):
        perm_cal[perm_dim] = i

    if transpose_first:
        result = grad.permute(*perm_cal).reshape(shape_cal)
    else:
        result = grad.reshape(shape_cal).permute(*perm_cal)
    return result.cpu()
