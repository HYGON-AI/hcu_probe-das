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
