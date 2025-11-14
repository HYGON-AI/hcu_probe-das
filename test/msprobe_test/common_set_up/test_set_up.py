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
# limitations under the License.

import importlib
from unittest import TestCase
from unittest.mock import MagicMock

import mindspore as ms
from mindspore import mint

try:
    from mint import distributed
except ImportError:
    distributed = MagicMock()
    setattr(mint, 'distributed', distributed)

from mindspore.common.api import _pynative_executor
mock_requires_grad = MagicMock(return_value=True)
setattr(_pynative_executor, "requires_grad", mock_requires_grad)

from mindspore import ops
if not hasattr(ops, 'DumpGradient'):
    DumpGradient = MagicMock()
    setattr(ops, 'DumpGradient', DumpGradient)

# ensure not to import torch_npu
from msprobe.mindspore.dump import mindspore_service
from msprobe.mindspore.monitor import common_func

from .mindtorch import reset_torch_tensor
from msprobe.mindspore.common import utils
from msprobe.mindspore.common.utils import is_mindtorch, register_backward_hook_functions

utils.mindtorch_check_result = None
importlib.reload(mindspore_service)
importlib.reload(common_func)
reset_torch_tensor()


def register_backward_pre_hook(*args, **kwargs):
    pass


register_backward_hook_functions['full'] = ms.nn.Cell.register_backward_hook
register_backward_hook_functions["pre"] = register_backward_pre_hook


class SetUp(TestCase):
    def test_case(self):
        self.assertTrue(hasattr(mint, 'distributed'))
        self.assertTrue(hasattr(_pynative_executor, 'requires_grad'))
        self.assertTrue(is_mindtorch())
        utils.mindtorch_check_result = None
