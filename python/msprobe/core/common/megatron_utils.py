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

from functools import wraps


class MegatronStepInfo:
    is_megatron = False
    is_forward = False
    is_backward = False
    forward_micro_step = -1
    backward_micro_step = -1

    @classmethod
    def reset(cls):
        """重置所有类属性到初始状态"""
        cls.is_megatron = False
        cls.is_forward = False
        cls.is_backward = False
        cls.forward_micro_step = -1
        cls.backward_micro_step = -1


def wrap_megatron_step(func, is_forward=True):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if not MegatronStepInfo.is_megatron:
            MegatronStepInfo.is_megatron = True
        if is_forward:
            MegatronStepInfo.is_forward = True
            MegatronStepInfo.is_backward = False
            MegatronStepInfo.forward_micro_step += 1
        else:
            MegatronStepInfo.is_forward = False
            MegatronStepInfo.is_backward = True
            MegatronStepInfo.backward_micro_step += 1
        return func(*args, **kwargs)

    return wrapped_func


def get_micro_step():
    return MegatronStepInfo.forward_micro_step if MegatronStepInfo.is_forward else MegatronStepInfo.backward_micro_step


def is_megatron():
    return MegatronStepInfo.is_megatron
