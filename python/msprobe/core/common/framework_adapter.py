# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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
# limitations under the License.import functools
import functools
from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import check_file_or_directory_path
from msprobe.core.common.file_utils import save_npy


class FrameworkDescriptor:
    def __get__(self, instance, owner):
        if owner._framework is None:
            owner.import_framework()
        return owner._framework


class FmkAdp:
    fmk = Const.PT_FRAMEWORK
    supported_fmk = [Const.PT_FRAMEWORK, Const.MS_FRAMEWORK]
    supported_dtype_list = ["bfloat16", "float16", "float32", "float64"]
    _framework = None
    framework = FrameworkDescriptor()

    @classmethod
    def import_framework(cls):
        if cls.fmk == Const.PT_FRAMEWORK:
            import torch
            cls._framework = torch
        elif cls.fmk == Const.MS_FRAMEWORK:
            import mindspore
            cls._framework = mindspore
        else:
            raise Exception(f"init framework adapter error, not in {cls.supported_fmk}")

    @classmethod
    def set_fmk(cls, fmk=Const.PT_FRAMEWORK):
        if fmk not in cls.supported_fmk:
            raise Exception(f"init framework adapter error, not in {cls.supported_fmk}")
        cls.fmk = fmk
        cls._framework = None  # 重置框架，以便下次访问时重新导入

    @classmethod
    def get_rank(cls):
        if cls.fmk == Const.PT_FRAMEWORK:
            return cls.framework.distributed.get_rank()
        return cls.framework.communication.get_rank()

    @classmethod
    def get_rank_id(cls):
        if cls.is_initialized():
            return cls.get_rank()
        return 0

    @classmethod
    def is_initialized(cls):
        if cls.fmk == Const.PT_FRAMEWORK:
            return cls.framework.distributed.is_initialized()
        return cls.framework.communication.GlobalComm.INITED

    @classmethod
    def is_nn_module(cls, module):
        if cls.fmk == Const.PT_FRAMEWORK:
            return isinstance(module, cls.framework.nn.Module)
        return isinstance(module, cls.framework.nn.Cell)

    @classmethod
    def is_tensor(cls, tensor):
        if cls.fmk == Const.PT_FRAMEWORK:
            return isinstance(tensor, cls.framework.Tensor)
        return isinstance(tensor, cls.framework.Tensor)

    @classmethod
    def process_tensor(cls, tensor, func):
        if cls.fmk == Const.PT_FRAMEWORK:
            if not tensor.is_floating_point() or tensor.dtype == cls.framework.float64:
                tensor = tensor.float()
            return float(func(tensor))
        return float(func(tensor).asnumpy())

    @classmethod
    def tensor_max(cls, tensor):
        return cls.process_tensor(tensor, lambda x: x.max())

    @classmethod
    def tensor_min(cls, tensor):
        return cls.process_tensor(tensor, lambda x: x.min())

    @classmethod
    def tensor_mean(cls, tensor):
        return cls.process_tensor(tensor, lambda x: x.mean())

    @classmethod
    def tensor_norm(cls, tensor):
        return cls.process_tensor(tensor, lambda x: x.norm())
    
    @classmethod
    def save_tensor(cls, tensor, filepath):
        if cls.fmk == Const.PT_FRAMEWORK:
            tensor_npy = tensor.cpu().detach().float().numpy()
        else:
            tensor_npy = tensor.asnumpy()
        save_npy(tensor_npy, filepath)

    @classmethod
    def dtype(cls, dtype_str):
        if dtype_str not in cls.supported_dtype_list:
            raise Exception(f"{dtype_str} is not supported by adapter, not in {cls.supported_dtype_list}")
        return getattr(cls.framework, dtype_str)

    @classmethod
    def named_parameters(cls, module):
        if cls.fmk == Const.PT_FRAMEWORK:
            if not isinstance(module, cls.framework.nn.Module):
                raise Exception(f"{module} is not a torch.nn.Module")
            return module.named_parameters()
        if not isinstance(module, cls.framework.nn.Cell):
            raise Exception(f"{module} is not a mindspore.nn.Cell")
        return module.parameters_and_names()

    @classmethod
    def register_forward_pre_hook(cls, module, hook, with_kwargs=False):
        if cls.fmk == Const.PT_FRAMEWORK:
            if not isinstance(module, cls.framework.nn.Module):
                raise Exception(f"{module} is not a torch.nn.Module")
            module.register_forward_pre_hook(hook, with_kwargs=with_kwargs)
        else:
            if not isinstance(module, cls.framework.nn.Cell):
                raise Exception(f"{module} is not a mindspore.nn.Cell")
            original_construct = module.construct

            @functools.wraps(original_construct)
            def new_construct(*args, **kwargs):
                if with_kwargs:
                    hook(module, args, kwargs)
                else:
                    hook(module, args)
                return original_construct(*args, **kwargs)

            module.construct = new_construct

    @classmethod
    def load_checkpoint(cls, path, to_cpu=True, weights_only=True):
        check_file_or_directory_path(path)
        if cls.fmk == Const.PT_FRAMEWORK:
            try:
                if to_cpu:
                    return cls.framework.load(path, map_location=cls.framework.device("cpu"), weights_only=weights_only)
                else:
                    return cls.framework.load(path, weights_only=weights_only)
            except Exception as e:
                raise RuntimeError(f"load pt file {path} failed: {e}") from e
        return mindspore.load_checkpoint(path)
    
    @classmethod
    def asnumpy(cls, tensor):
        if cls.fmk == Const.PT_FRAMEWORK:
            return tensor.float().numpy()
        return tensor.float().asnumpy()