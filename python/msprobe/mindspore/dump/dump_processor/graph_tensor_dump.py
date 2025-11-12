# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import os
from collections import OrderedDict
import mindspore as ms
from mindspore import hal, ops, Tensor
from mindspore.ops.primitive import _run_op


def _iterate_items(data):
    if isinstance(data, (dict, OrderedDict)):
        return data.items()
    elif isinstance(data, (list, tuple)):
        return enumerate(data)
    else:
        raise TypeError("Unsupported data type")


class _SaveBase:
    def __init__(self, save_dir):
        super(_SaveBase, self).__init__()
        self.path = save_dir
        self.save_func = _npy_save

    def get_save_func(self):
        return self.save_func


@ms.jit_class
class _SaveCell(_SaveBase):
    def __call__(self, name, data):
        return self.get_save_func()(self.path, name, data)


class _SaveGradBase:
    def __init__(self, save_dir, name):
        super(_SaveGradBase, self).__init__()
        self.file = save_dir + name


@ms.jit_class
class _SaveGradCell(_SaveGradBase):
    def __init__(self, save_dir, name):
        super(_SaveGradCell, self).__init__(save_dir, name)
        self.ms_save_grad = ms.ops.InsertGradientOf(
            _wrapper_save_grad_func(self.file))

    def __call__(self, x):
        if isinstance(x, ms.Tensor):
            return self.ms_save_grad(x)
        else:
            raise TypeError(f"For 'save_grad', the type of argument 'data' must be mindspore.Tensor or torch.tensor, "
                            f"but got {type(x)}")


def _npy_save_ops(file, data):
    if isinstance(data, ms.Tensor):
        if data.dtype == ms.bfloat16:
            data = data.float()
        ms.ops.TensorDump()(file, data)
    else:
        raise TypeError(f"For 'save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, "
                        f"but got {type(data)}")


def _wrapper_save_grad_func(file):
    def _save_grad_func(grad):
        data = grad
        if data.dtype == ms.bfloat16:
            data = data.float()
        ms.ops.TensorDump()(file, data)
        return grad
    return _save_grad_func


def _npy_save(save_dir, item_name, data):
    if isinstance(data, (list, tuple, dict, OrderedDict)):
        for key, val in _iterate_items(data):
            _npy_save(save_dir, f"{item_name}.{key}", val)
    else:
        if data is None:
            return
        _npy_save_ops(f"{save_dir}{item_name}", data)


def generate_dump_dir(save_dir, sep=os.sep):
    """
    usage: generate dump directory path str in mindspore graph mode
    """
    full_suffix = '{step}' + sep + '{rank}' + sep
    if save_dir and save_dir[-1] != sep:
        result_dir = save_dir + sep + full_suffix
    else:
        result_dir = save_dir + full_suffix
    return result_dir


def save(save_dir, name, data):
    """
    save tensor.
    """
    dump_dir = generate_dump_dir(save_dir)
    _SaveCell(dump_dir)(name, data)


def save_grad(save_dir, name, data):
    """
    save grad.
    """
    dump_dir = generate_dump_dir(save_dir)
    suffix_name = name + '_grad'
    return _SaveGradCell(dump_dir, suffix_name)(data)


def step():
    hal.synchronize()
    temp_tensor = Tensor([1], dtype=ms.float32)
    step_flag = "<tensordump-update-step>"
    _run_op(ops.TensorDump(), "TensorDump", (step_flag, temp_tensor))
    ops.tensordump(step_flag, temp_tensor)
    hal.synchronize()