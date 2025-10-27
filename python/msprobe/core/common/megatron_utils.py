# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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
