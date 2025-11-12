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
