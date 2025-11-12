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

from msprobe.core.common.const import Const
from msprobe.pytorch.common.log import logger

torch_version_above_or_equal_2 = torch.__version__.split('+')[0] >= '2.0'
if torch_version_above_or_equal_2:
    from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook


def register_optimizer_hook(data_collector):
    def optimizer_pre_step_hook(optimizer, args, kwargs):
        data_collector.optimizer_status = Const.OPTIMIZER

    def optimizer_post_step_hook(optimizer, args, kwargs):
        data_collector.optimizer_status = Const.END_PREFIX + Const.OPTIMIZER

    def patch_clip_grad(func):
        def wrapper(*args, **kwargs):
            data_collector.optimizer_status = Const.CLIP_GRAD
            result = func(*args, **kwargs)
            data_collector.optimizer_status = Const.END_PREFIX + Const.CLIP_GRAD
            return result

        return wrapper

    if torch_version_above_or_equal_2:
        register_optimizer_step_pre_hook(optimizer_pre_step_hook)
        register_optimizer_step_post_hook(optimizer_post_step_hook)
    else:
        logger.info_on_rank_0("Pytorch version is below 2.0, cannot register optimizer hook.")

    try:
        torch.nn.utils.clip_grad_norm_ = patch_clip_grad(torch.nn.utils.clip_grad_norm_)
        torch.nn.utils.clip_grad_norm = patch_clip_grad(torch.nn.utils.clip_grad_norm)
        torch.nn.utils.clip_grad_value_ = patch_clip_grad(torch.nn.utils.clip_grad_value_)
    except Exception as e:
        logger.info_on_rank_0("Cannot patch clip grad function. detail:%s" % str(e))

    try:
        from megatron.core.optimizer import MegatronOptimizer
        MegatronOptimizer.clip_grad_norm = patch_clip_grad(MegatronOptimizer.clip_grad_norm)
    except ImportError:
        pass
    except Exception as e:
        logger.info_on_rank_0("Cannot patch megatron clip grad function. detail:%s" % str(e))