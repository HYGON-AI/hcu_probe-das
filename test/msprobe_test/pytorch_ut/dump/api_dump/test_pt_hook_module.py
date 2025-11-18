# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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
from collections import defaultdict
from unittest.mock import MagicMock, patch

from msprobe.core.dump.hook_manager import HookSet
from msprobe.pytorch.dump.api_dump.hook_module import HOOKModule


class TestHOOKModule(unittest.TestCase):
    def setUp(self):
        self.mock_build_hook = MagicMock(return_value=HookSet(MagicMock(), MagicMock(), MagicMock()))
        HOOKModule.module_count = defaultdict(int)
        HOOKModule.inner_stop_hook = defaultdict(bool)

    @patch.object(HOOKModule, '_call_func')
    def test_call_with_stop_hooks(self, mock_call_func):
        mock_call_func.return_value = "test_result"
        module1 = HOOKModule(self.mock_build_hook)

        result = module1("arg1", "arg2", key="value")
        mock_call_func.assert_called_once_with("arg1", "arg2", key="value")
        self.assertEqual(result, "test_result")

    @patch.object(HOOKModule, '_call_func')
    def test_call_with_start_hooks(self, mock_call_func):
        mock_call_func.return_value = "test_result"
        module1 = HOOKModule(self.mock_build_hook)

        result = module1("arg1", "arg2", key="value")
        mock_call_func.assert_called_once_with("arg1", "arg2", key="value")
        self.assertEqual(result, "test_result")

    def test_reset_module_stats(self):
        HOOKModule.module_count = {"Tensor.add.0.forward": 0}
        HOOKModule.reset_module_stats()
        self.assertDictEqual(HOOKModule.module_count, defaultdict(int))

    def test_add_module_count(self):
        HOOKModule.add_module_count("Tensor.add.0.forward")
        self.assertEqual(HOOKModule.module_count["Tensor.add.0.forward"], 1)

    def test_get_module_count(self):
        HOOKModule.module_count = {"Tensor.add.0.forward": 0}
        result = HOOKModule.get_module_count("Tensor.add.0.forward")
        self.assertEqual(result, 0)
