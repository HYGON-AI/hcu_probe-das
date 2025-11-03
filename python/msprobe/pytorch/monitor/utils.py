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

from msprobe.pytorch.common.log import logger


device = "cpu"
try:
    import torch_npu
    device = "npu"
except ImportError:
    if torch.cuda.is_available():
        device = "cuda"

NAN_TENSOR_ON_DEVICE = None


def get_nan_tensor():
    global NAN_TENSOR_ON_DEVICE
    if not NAN_TENSOR_ON_DEVICE:
        NAN_TENSOR_ON_DEVICE = torch.tensor(torch.nan, device=device)
    return NAN_TENSOR_ON_DEVICE


def get_param_struct(param):
    res = {}
    if isinstance(param, (tuple, list)):
        res['config'] = f'{type(param).__name__}[{len(param)}]'
        for i, x in enumerate(param):
            res[i] = f'size={tuple(x.shape)}, dtype={x.dtype}' if torch.is_tensor(x) else f'{type(x)}'
    elif torch.is_tensor(param):
        res['config'] = 'tensor'
        res['tensor'] = f'size={tuple(param.shape)}, dtype={param.dtype}'
    else:
        res['config'] = f'{type(param)}'
        logger.warning(f'Not support type({type(param)}) now, please check the type of param {param}')
    return res