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
