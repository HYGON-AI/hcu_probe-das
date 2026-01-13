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