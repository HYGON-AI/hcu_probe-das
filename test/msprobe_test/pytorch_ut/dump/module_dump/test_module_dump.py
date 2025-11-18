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
from unittest.mock import patch, MagicMock

from torch import nn

from msprobe.pytorch.common.log import logger
from msprobe.pytorch.dump.module_dump.module_dump import ModuleDumper
from msprobe.pytorch.dump.module_dump.module_processor import ModuleProcessor


class TestModuleDumper(unittest.TestCase):
    def setUp(self):
        self.service = MagicMock()
        with patch('msprobe.pytorch.dump.module_dump.module_dump.get_api_register'):
            self.module_dumper = ModuleDumper(self.service)

    def test__init__(self):
        self.service = MagicMock()
        with patch('msprobe.pytorch.dump.module_dump.module_dump.get_api_register') as mock_get_api_register:
            self.module_dumper = ModuleDumper(self.service)
            self.assertEqual(self.module_dumper.service, self.service)
            mock_get_api_register.assert_called_once()

    def test_start_module_dump(self):
        module = nn.Module()
        with patch.object(logger, 'info_on_rank_0') as mock_info:
            module.msprobe_hook = True
            ModuleProcessor.enable_module_dump = False
            self.module_dumper.api_register.restore_all_api.reset_mock()
            self.module_dumper.start_module_dump(module, 'dump_name')
            mock_info.assert_called_with(
                'The init dump is enabled, and the module dump function will not be available.'
            )
            self.assertFalse(ModuleProcessor.enable_module_dump)
            self.module_dumper.api_register.restore_all_api.assert_not_called()
            self.assertFalse(hasattr(module, 'msprobe_module_dump'))

            del module.msprobe_hook
            mock_info.reset_mock()
            self.module_dumper.start_module_dump(module, 'dump_name')
            mock_info.assert_not_called()
            self.assertTrue(ModuleProcessor.enable_module_dump)
            self.module_dumper.api_register.restore_all_api.assert_called_once()
            self.module_dumper.service.module_processor.register_module_hook.assert_called_with(
                module,
                self.module_dumper.service.build_hook,
                recursive=False,
                module_names=['dump_name']
            )
            self.assertTrue(module.msprobe_module_dump)
            ModuleProcessor.enable_module_dump = False

            self.module_dumper.api_register.restore_all_api.reset_mock()
            self.module_dumper.service.module_processor.register_module_hook.reset_mock()
            self.module_dumper.start_module_dump(module, 'dump_name')
            mock_info.assert_not_called()
            self.assertTrue(ModuleProcessor.enable_module_dump)
            self.module_dumper.api_register.restore_all_api.assert_called_once()
            self.module_dumper.service.module_processor.register_module_hook.assert_not_called()

            ModuleProcessor.enable_module_dump = False

    def test_stop_module_dump(self):
        ModuleProcessor.enable_module_dump = True
        self.module_dumper.api_register.register_all_api.reset_mock()
        self.module_dumper.stop_module_dump()
        self.assertFalse(ModuleProcessor.enable_module_dump)
        self.module_dumper.api_register.register_all_api.assert_called_once()

        self.module_dumper.api_register.register_all_api.reset_mock()
