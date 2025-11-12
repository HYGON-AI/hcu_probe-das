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

import mindspore
import numpy as np
from mindspore._c_expression import typing
from mindspore.common import dtype as mstype

from msprobe.mindspore.api_accuracy_checker import torch_mindtorch_importer

if torch_mindtorch_importer.is_valid_pt_mt_env:
    from msprobe.mindspore.api_accuracy_checker.torch_mindtorch_importer import mindtorch
    from msprobe.mindspore.api_accuracy_checker.torch_mindtorch_importer import torch
else:
    from msprobe.mindspore.api_accuracy_checker.torch_mindtorch_importer import mindtorch
    import torch

INT8 = "Int8"
UINT8 = "UInt8"
INT16 = "Int16"
UINT16 = "UInt16"
INT32 = "Int32"
UINT32 = "UInt32"
INT64 = "Int64"
UINT64 = "UInt64"
FLOAT16 = "Float16"
FLOAT32 = "Float32"
FLOAT64 = "Float64"
BOOL = "Bool"
BFLOAT16 = "BFloat16"
INT4 = "Int4"

dtype_str_to_ms_dtype = {
    INT8: mstype.int8,
    UINT8: mstype.uint8,
    INT16: mstype.int16,
    UINT16: mstype.uint16,
    INT32: mstype.int32,
    UINT32: mstype.uint32,
    INT64: mstype.int64,
    UINT64: mstype.uint64,
    FLOAT16: mstype.float16,
    FLOAT32: mstype.float32,
    FLOAT64: mstype.float64,
    BOOL: mstype.bool_,
    BFLOAT16: mstype.bfloat16,
    INT4: mstype.qint4x2
}
ms_dtype_to_dtype_str = {value: key for key, value in dtype_str_to_ms_dtype.items()}

dtype_str_to_np_dtype = {
    INT8: np.int8,
    UINT8: np.uint8,
    INT16: np.int16,
    UINT16: np.uint16,
    INT32: np.int32,
    UINT32: np.uint32,
    INT64: np.int64,
    UINT64: np.uint64,
    FLOAT16: np.float16,
    FLOAT32: np.float32,
    FLOAT64: np.float64,
    BOOL: np.bool_
}
np_dtype_to_dtype_str = {value: key for key, value in dtype_str_to_np_dtype.items()}

dtype_str_to_torch_dtype = {
    INT8: torch.int8,
    UINT8: torch.uint8,
    INT16: torch.int16,
    INT32: torch.int32,
    INT64: torch.int64,
    FLOAT16: torch.float16,
    FLOAT32: torch.float32,
    FLOAT64: torch.float64,
    BOOL: torch.bool,
    BFLOAT16: torch.bfloat16,
}
torch_dtype_to_dtype_str = {value: key for key, value in dtype_str_to_torch_dtype.items()}


dtype_str_to_mindtorch_dtype = {
    INT8: mindtorch.int8,
    UINT8: mindtorch.uint8,
    INT16: mindtorch.int16,
    INT32: mindtorch.int32,
    INT64: mindtorch.int64,
    FLOAT16: mindtorch.float16,
    FLOAT32: mindtorch.float32,
    FLOAT64: mindtorch.float64,
    BOOL: mindtorch.bool,
    BFLOAT16: mindtorch.bfloat16,
}
mindtorch_dtype_to_dtype_str = {value: key for key, value in dtype_str_to_mindtorch_dtype.items()}

MINDSPORE_TENSOR_TYPE_STR = "mindspore.Tensor"
BOOL_TYPE_STR = "bool"
INT_TYPE_STR = "int"
FLOAT_TYPE_STR = "float"
SLICE_TYPE_STR = "slice"
TUPLE_TYPE_STR = "tuple"
STR_TYPE_STR = "str"
MINDSPORE_DTYPE_TYPE_STR = "mindspore.dtype"
TORCH_DTYPE_TYPE_STR = "torch.dtype"

api_info_type_str_to_type = {
    MINDSPORE_TENSOR_TYPE_STR: mindspore.Tensor,
    BOOL_TYPE_STR: bool,
    INT_TYPE_STR: int,
    FLOAT_TYPE_STR: float,
    SLICE_TYPE_STR: slice,
    STR_TYPE_STR: str,
    MINDSPORE_DTYPE_TYPE_STR: typing.Type,
}
type_to_api_info_type_str = {value: key for key, value in api_info_type_str_to_type.items()}

DEFAULT_CONSTRUCT_NP_FLOAT_DTYPE = np.float64
DEFAULT_CONSTRUCT_NP_INT_DTYPE = np.float64
DEFAULT_CONSTRUCT_NP_UINT_DTYPE = np.float64

float_dtype_str_list = [
    FLOAT16,
    FLOAT32,
    FLOAT64,
    BFLOAT16,
]

int_dtype_str_list = [
    INT8,
    INT16,
    INT32,
    INT64,
    BOOL,
    INT4,
]

uint_dtype_str_list = [
    UINT8,
    UINT16,
    UINT32,
    UINT64,
]
