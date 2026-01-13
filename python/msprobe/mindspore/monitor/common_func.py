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
