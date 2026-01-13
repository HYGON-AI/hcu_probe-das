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
