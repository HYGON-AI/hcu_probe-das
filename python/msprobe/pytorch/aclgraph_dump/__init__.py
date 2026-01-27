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
import torch_npu

# Import the C++ extension to register TORCH_LIBRARY implementations.
try:
    from msprobe.lib import aclgraph_dump_ext  # noqa: F401
except Exception as exc:
    raise RuntimeError(f"Failed to import msprobe.lib.aclgraph_dump_ext: {exc}")

# Register Python fake implementation for meta tensors.
from ._meta import _register_meta  # noqa: E402
_register_meta()

from torch.fx.node import has_side_effect
has_side_effect(torch.ops.my_ns.acl_save.default)

def acl_save(x: torch.Tensor, path: str) -> torch.Tensor:
    """
    acl_save(tensor, path) -> tensor

    Copy tensor to CPU and save to a .pt file.
    The file name is generated as {base}_{seq}.pt in the same directory.
    For NPU input, the save runs on the current NPU stream; synchronize if needed.
    """
    return torch.ops.my_ns.acl_save(x, path)


__all__ = ["acl_save"]
