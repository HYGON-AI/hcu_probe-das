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


from mindspore import nn
from mindspore import communication
from msprobe.core.common.log import logger
from msprobe.mindspore.common.utils import is_mindtorch
if is_mindtorch():
    import torch


def is_valid_instance(model):
    return isinstance(model, torch.nn.Module) if is_mindtorch() else isinstance(model, nn.Cell)


def get_submodules(model):
    if not is_valid_instance(model):
        logger.info("Counter invalid model, nothing to hook")
        return {}
    return model.named_modules() if is_mindtorch() else model.cells_and_names()


def get_parameters(model):
    if not is_valid_instance(model):
        return {}
    if is_mindtorch():
        return model.named_parameters()
    else:
        return model.parameters_and_names()


def get_rank():
    if comm_is_initialized():
        return communication.get_rank()
    return 0


def comm_is_initialized():
    return communication.GlobalComm.INITED
