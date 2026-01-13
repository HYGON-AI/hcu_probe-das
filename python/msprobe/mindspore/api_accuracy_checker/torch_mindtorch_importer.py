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

import os
import gc
import sys
from pathlib import Path
import mindspore
from msprobe.mindspore.common.log import logger
from msprobe.core.common.const import Const, CompareConst
from msprobe.mindspore.common.const import MsCompareConst
import torch as mindtorch
from torch import Tensor as mindtorch_tensor
import torch.nn.functional as mindtorch_func
import torch.distributed as mindtorch_dist


is_valid_pt_mt_env = True


def is_mindtorch():
    mindtorch_check_result = False
    try:
        import torch as test_torch
        from mindspore import Tensor as MindsporeTensor
    except ImportError:
        return mindtorch_check_result
    tensor = test_torch.tensor(0.0)
    if isinstance(tensor, MindsporeTensor):
        mindtorch_check_result = True

    return mindtorch_check_result


def remove_torch_related_paths():
    removed_paths = []
    if not is_mindtorch():
        return
    try:
        import torch as remove_torch
        torch_file = remove_torch.__file__
    except ImportError:
        return

    torch_dir = os.path.dirname(torch_file)

    torch_dir_path = Path(torch_dir).resolve()
    parent_dir = torch_dir_path.parent

    paths_to_remove = [str(parent_dir)]

    for path in paths_to_remove:
        try:
            path_resolved = str(Path(path).resolve())
        except Exception as error:
            logger.debug(f"Failed to resolve path {path}: {error}")
            continue

        if path_resolved in sys.path:
            index = sys.path.index(path_resolved)
            removed_paths.append((path_resolved, index))
            sys.path.pop(index)

    return


def clear_torch_from_sys_modules():
    modules_to_remove = []
    for module in sys.modules:
        if module == "torch" or module.startswith("torch."):
            modules_to_remove.append(module)

    for module in modules_to_remove:
        del sys.modules[module]


def set_pt_mt_env_invalid():
    global is_valid_pt_mt_env
    is_valid_pt_mt_env = False


def delete_torch_paths():

    if not is_mindtorch():
        set_pt_mt_env_invalid()

    clear_torch_from_sys_modules()

    for count_delete_env_path in range(MsCompareConst.MAX_RECURSION_DEPTH):
        if not is_mindtorch():
            break

        remove_torch_related_paths()

        clear_torch_from_sys_modules()

        if count_delete_env_path >= MsCompareConst.MAX_RECURSION_DEPTH - 1:
            raise Exception(f"Please check if you have a valid PyTorch and MindTorch environment, and ensure "
                            f"the PYTHONPATH environment variable depth does not "
                            f"exceed {MsCompareConst.MAX_RECURSION_DEPTH}.")


if not is_mindtorch():
    set_pt_mt_env_invalid()

else:
    initial_sys_path = sys.path.copy()
    delete_torch_paths()

    gc.collect()

    import torch

    if is_mindtorch():
        set_pt_mt_env_invalid()

    sys.path = initial_sys_path


