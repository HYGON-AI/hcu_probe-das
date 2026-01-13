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
