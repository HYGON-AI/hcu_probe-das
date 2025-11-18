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
from unittest.mock import MagicMock, patch

import torch
from msprobe.pytorch.dump.api_dump.script_wrapper import wrap_jit_script_func


class TestWrapJitScriptFunc(unittest.TestCase):
    def setUp(self):
        self.original_script = torch.jit.script

        self.mock_api_register = MagicMock()
        self.mock_api_register.all_api_registered = True
        self.mock_api_register.register_all_api = MagicMock()
        self.mock_api_register.restore_all_api = MagicMock()

    def tearDown(self):
        torch.jit.script = self.original_script

    @patch('torch.jit.script', new_callable=MagicMock)
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.get_api_register', return_value=MagicMock())
    def test_patched_script(self, mock_get_api, mock_original_script):
        mock_original_script.return_value = "mocked_result"
        mock_get_api.return_value = self.mock_api_register

        wrap_jit_script_func()

        self.assertNotEqual(torch.jit.script, self.original_script)

        result = torch.jit.script("test_input")

        mock_original_script.assert_called_once_with("test_input")
        self.assertEqual(result, "mocked_result")

        self.mock_api_register.restore_all_api.assert_called_once()
        self.mock_api_register.register_all_api.assert_called_once()
