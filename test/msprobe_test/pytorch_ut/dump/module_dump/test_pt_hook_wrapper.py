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

import unittest
from unittest.mock import MagicMock

import torch

from msprobe.pytorch.dump.module_dump.hook_wrapper import wrap_setup_backward_hook


class TestWrapSetupBackwardHook(unittest.TestCase):
    def setUp(self):
        self.mock_func = MagicMock()
        self.mock_func.return_value = ["clone_tensor1", "clone_tensor2"]

        self.decorated_func = wrap_setup_backward_hook(self.mock_func)

        self.tensor = torch.randn(3, requires_grad=True)
        torch.set_grad_enabled(True)

    def test_insufficient_args(self):
        result = self.decorated_func("test_case1")
        self.mock_func.assert_called_once_with("test_case1")
        self.assertListEqual(result, ["clone_tensor1", "clone_tensor2"])

    def test_normal_processing_flow(self):
        test_tensor = torch.randn(2, requires_grad=False)
        test_data = {
            "tensors": [self.tensor, torch.randn(2, requires_grad=True)],
            "nested": {
                "tuple": (self.tensor, test_tensor)
            }
        }

        mock_self = MagicMock()
        mock_self.module.inplace = False
        test_tensor1 = torch.randn(4, requires_grad=True)
        test_tensor2 = torch.randn(4, requires_grad=True)
        test_tensor3 = torch.randn(4, requires_grad=True)
        self.mock_func.return_value = [test_tensor1, test_tensor2, test_tensor3]
        result = self.decorated_func(mock_self, test_data)

        self.assertIsInstance(result, dict)
        self.assertTrue(torch.equal(result["tensors"][0], self.tensor))
        self.assertFalse(torch.equal(result["tensors"][1], test_tensor2))
        self.assertIsInstance(result["nested"]["tuple"][0], torch.Tensor)
        self.assertTrue(torch.equal(result["nested"]["tuple"][1], test_tensor))

    def test_complex_data_structures(self):
        test_case = [
            self.tensor,
            {"dict": torch.randn(4, requires_grad=True)},
            (torch.randn(5, requires_grad=True),),
            [torch.randn(6, requires_grad=True)]
        ]

        mock_self = MagicMock()
        mock_self.module.inplace = False
        test_tensor1 = torch.randn(4, requires_grad=True)
        test_tensor2 = torch.randn(5, requires_grad=True)
        test_tensor3 = torch.randn(6, requires_grad=True)
        self.mock_func.return_value = [self.tensor, test_tensor1, test_tensor2, test_tensor3]
        result = self.decorated_func(mock_self, test_case)

        self.assertIsInstance(result, list)
        self.assertFalse(torch.equal(result[1]["dict"], test_tensor1))
        self.assertFalse(torch.equal(result[2][0], test_tensor2))
        self.assertFalse(torch.equal(result[3][0], test_tensor3))
