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