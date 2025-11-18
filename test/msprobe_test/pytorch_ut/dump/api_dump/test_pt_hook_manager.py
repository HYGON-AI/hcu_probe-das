# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

import threading

import unittest
from unittest.mock import MagicMock, patch
from contextlib import nullcontext
from msprobe.pytorch.dump.api_dump.pt_hook_manager import PytorchHookManager
from msprobe.core.common.const import Const
from msprobe.core.dump.hook_manager import HookSet, BaseHookManager


class TestPytorchHookManager(unittest.TestCase):
    def setUp(self):
        self.mock_data_collector = MagicMock()
        self.mock_config = MagicMock()
        self.mock_config.data_mode = ["all"]
        self.mock_config.task = "statistics"
        self.manager = PytorchHookManager(
            self.mock_data_collector,
            self.mock_config
        )
        BaseHookManager.inner_switch[threading.get_ident()] = False

    def test_no_grad_context(self):
        self.assertIsInstance(self.manager._no_grad_context(), nullcontext)

    def test_add_count(self):
        with patch('msprobe.pytorch.dump.api_dump.pt_hook_manager.HOOKModule.add_module_count') as mock_add:
            self.manager._add_count("test_layer")
            mock_add.assert_called_once_with("test_layer")

    def test_process_kwargs_and_output(self):
        kwargs, output = self.manager._process_kwargs_and_output(
            None,
            None,
            "API",
            "kwargs_value",
            "output_value"
        )
        self.assertEqual(kwargs, "kwargs_value")
        self.assertEqual(output, "output_value")

        with patch('msprobe.pytorch.dump.api_dump.pt_hook_manager.torch_version_above_or_equal_2', new=True):
            kwargs, output = self.manager._process_kwargs_and_output(
                None,
                None,
                None,
                "kwargs_value",
                "output_value"
            )
            self.assertEqual(kwargs, "kwargs_value")
            self.assertEqual(output, "output_value")

        with patch('msprobe.pytorch.dump.api_dump.pt_hook_manager.torch_version_above_or_equal_2', new=False):
            kwargs, output = self.manager._process_kwargs_and_output(
                None,
                None,
                None,
                "kwargs_value",
                "output_value"
            )
            self.assertEqual(kwargs, {})
            self.assertEqual(output, "kwargs_value")

    def test_build_hook(self):
        hook_set = self.manager.build_hook(Const.API, "test_api")
        self.assertIsInstance(hook_set, HookSet)
        self.assertTrue(callable(hook_set.forward_pre_hook))
        self.assertTrue(callable(hook_set.distributed_forward_hook))
        self.assertIsNone(hook_set.forward_hook)
        self.assertIsNone(hook_set.backward_pre_hook)
        self.assertIsNone(hook_set.backward_hook)

        hook_set = self.manager.build_hook(Const.MODULE, "test_module")
        self.assertEqual(hook_set.forward_hook.__name__, "forward_hook")
        self.assertEqual(hook_set.backward_hook.__name__, "backward_hook")

    def test_need_exchange(self):
        self.assertTrue(self.manager._need_exchange(None))
        self.assertTrue(self.manager._need_exchange(MagicMock()))

    def test_get_params_dict(self):
        mock_module = MagicMock()

        self.mock_config.task = Const.STRUCTURE
        params_dict = self.manager._get_params_dict(mock_module)
        self.assertEqual(params_dict, {})

        self.mock_config.task = "statistics"

        mock_named_params = [
            ("conv.weight", MagicMock()),
            ("bn.bias", MagicMock())
        ]
        mock_module.named_parameters.return_value = mock_named_params
        params_dict = self.manager._get_params_dict(mock_module)
        mock_module.named_parameters.assert_called_once_with(recurse=False)

        self.assertEqual(set(params_dict.keys()), {"weight", "bias"})
        self.assertEqual(params_dict["weight"], mock_named_params[0][1])
        self.assertEqual(params_dict["bias"], mock_named_params[1][1])
