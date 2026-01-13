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
import inspect

from mindspore import Tensor, ops, mint
from mindspore.mint import distributed
from mindspore.mint.nn import functional
from mindspore.communication import comm_func

from msprobe.core.common.file_utils import load_yaml
from msprobe.core.common.utils import Const
from msprobe.core.dump.api_dump.api_registry import ApiRegistry
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.common.utils import is_mindtorch
from msprobe.mindspore.dump.dump_processor.hook_cell.hook_cell import HOOKCell


stub_tensor_existed = True
try:
    from mindspore.common._stub_tensor import StubTensor
except ImportError:
    stub_tensor_existed = False

cur_path = os.path.dirname(os.path.realpath(__file__))
if not is_mindtorch():
    _api_types = {
        Const.MS_FRAMEWORK: {
            Const.MS_API_TYPE_OPS: ((ops,), (ops,)),
            Const.MS_API_TYPE_TENSOR: ((Tensor,), (Tensor,)),
            Const.MS_API_TYPE_MINT: ((mint,), (mint,)),
            Const.MS_API_TYPE_MINT_FUNC: ((functional,), (functional,)),
            Const.MS_API_TYPE_COM: ((comm_func,), (comm_func,)),
            Const.MS_API_TYPE_MINT_DIST: ((distributed,), (distributed,))
        }
    }
    if stub_tensor_existed:
        _api_types.get(Const.MS_FRAMEWORK).update(
            {Const.MS_API_TYPE_STUB_TENSOR: ((StubTensor,), (StubTensor,))}
        )

    _supported_api_list_path = (os.path.join(cur_path, MsConst.SUPPORTED_API_LIST_FILE),)
    _blacklist = []
else:
    import torch
    import torch_npu
    _api_types = {
        Const.MT_FRAMEWORK: {
            Const.PT_API_TYPE_FUNCTIONAL: ((torch.nn.functional,), (torch.nn.functional,)),
            Const.PT_API_TYPE_TENSOR: ((torch.Tensor,), (torch.Tensor,)),
            Const.PT_API_TYPE_TORCH: ((torch,), (torch,)),
            Const.PT_API_TYPE_NPU: ((torch_npu,), (torch_npu,)),
            Const.PT_API_TYPE_DIST: ((torch.distributed,), (torch.distributed, torch.distributed.distributed_c10d))
        }
    }
    _supported_api_list_path = (os.path.join(cur_path, '../../../../pytorch/dump/api_dump',
                                             MsConst.SUPPORTED_API_LIST_FILE),)
    _blacklist = []

_inner_used_api = {
    Const.MS_FRAMEWORK + Const.SEP + Const.MS_API_TYPE_OPS: (
        ops, "norm", "square", "sqrt", "is_complex", "stack", "is_floating_point"
    ),
    Const.MS_FRAMEWORK + Const.SEP + Const.MS_API_TYPE_TENSOR: (
        Tensor, "to", "numel", 'sum'
    ),
    Const.MS_FRAMEWORK + Const.SEP + Const.MS_API_TYPE_MINT: (
        mint, "max", "min", "mean", "norm"
    )
}


class ApiTemplate(HOOKCell):
    def __init__(self, api_name, api_func, prefix, hook_build_func):
        self.api_name = api_name
        self.prefix_api_name = prefix + Const.SEP + str(api_name.split(Const.SEP)[-1]) + Const.SEP
        distributed_prefix = Const.DIST_API_TYPE_PREFIX if is_mindtorch() else Const.MINT_DIST_API_TYPE_PREFIX
        self.op_is_distributed = prefix == distributed_prefix
        super().__init__(hook_build_func)
        self.api_func = api_func

    @staticmethod
    def async_to_sync(output):
        # Fake handle, used to return after the CommHandle executes the wait method
        fake_handle = type("FakeHandle", (), {"wait": lambda self: None})()
        if isinstance(output, tuple) and len(output) == 2 and hasattr(output[1], "wait"):
            output[1].wait()
            output = (output[0], fake_handle)
        elif hasattr(output, "wait"):
            output.wait()
            output = fake_handle
        return output

    def construct(self, *args, **kwargs):
        if self.api_name.startswith(MsConst.DROPOUT_API_NAME_PREFIX):
            return args[0] if args else kwargs.get(Const.INPUT)

        output = self.api_func(*args, **kwargs)

        if self.prefix_api_name.startswith(
            (MsConst.DISTRIBUTED_DATA_PREFIX, Const.MINT_DIST_API_TYPE_PREFIX)
        ):
            try:
                bound = inspect.signature(self.api_func).bind(*args, **kwargs)
                bound.apply_defaults()
                use_async_op_flag = bound.arguments.get("async_op", False)
            except Exception as e:
                use_async_op_flag = False
                logger.warning(f"fail to get dist api's func signature because {e}, no wait")

            if use_async_op_flag or self.api_name in ["isend", "irecv"]:
                output = self.async_to_sync(output)
            if self.api_name == "batch_isend_irecv" and isinstance(output, list):
                output = [self.async_to_sync(handle) for handle in output]

        return output

    def forward(self, *args, **kwargs):
        if self.api_name.startswith(MsConst.DROPOUT_API_NAME_PREFIX):
            return args[0] if args else kwargs.get(Const.INPUT)
        return self.api_func(*args, **kwargs)


api_register = None
stub_tensor_set = False


def get_api_register(return_new=False):
    global stub_tensor_set

    def stub_method(method):
        def wrapped_method(*args, **kwargs):
            return method(*args, **kwargs)
        return wrapped_method
    if not is_mindtorch() and stub_tensor_existed and not stub_tensor_set:
        api_names = load_yaml(_supported_api_list_path[0]).get(Const.MS_API_TYPE_TENSOR, [])
        for attr_name in dir(StubTensor):
            attr = getattr(StubTensor, attr_name)
            if attr_name in api_names and callable(attr):
                setattr(StubTensor, attr_name, stub_method(attr))
        stub_tensor_set = True

    if return_new:
        return ApiRegistry(
            _api_types,
            _inner_used_api,
            _supported_api_list_path,
            ApiTemplate,
            _blacklist
        )

    global api_register
    if api_register is None:
        api_register = ApiRegistry(
            _api_types,
            _inner_used_api,
            _supported_api_list_path,
            ApiTemplate,
            _blacklist
        )
    return api_register
