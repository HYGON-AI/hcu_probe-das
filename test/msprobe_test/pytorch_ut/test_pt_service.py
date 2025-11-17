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

import unittest
from unittest.mock import MagicMock, patch
from msprobe.pytorch.dump.pytorch_service import PytorchService
from msprobe.core.common.utils import Const
from msprobe.pytorch.dump.module_dump.module_processor import ModuleProcessor
from msprobe.pytorch.dump.api_dump.hook_module import HOOKModule


class TestPytorchService(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.step = []
        self.config.rank = []
        self.config.level = Const.LEVEL_MIX
        self.config.task = Const.STATISTICS
        
        with patch('msprobe.core.dump.service.build_data_collector'):
            self.service = PytorchService(self.config)  

        self.service.logger = MagicMock()
        self.service.data_collector = MagicMock()
        self.service.module_processor = MagicMock()
        self.service.api_register = MagicMock()
    
    def test_framework_type(self):
        self.assertEqual(self.service._get_framework_type, Const.PT_FRAMEWORK)
    
    @patch('msprobe.pytorch.dump.pytorch_service.get_rank_if_initialized')
    def test_get_current_rank(self, mock_get_rank):
        mock_get_rank.return_value = 5
        self.assertEqual(self.service._get_current_rank(), 5)
    
    def test_init_specific_components(self):
        with patch('msprobe.core.dump.service.build_data_collector'):
            service = PytorchService(self.config)

        self.assertIsNotNone(service.logger)
        self.assertIsNotNone(service.api_register)
        self.assertIsNotNone(service.module_processor)
        self.assertIsNotNone(service.hook_manager)
    
    def test_register_hook(self):
        self.service._register_hook()
    
    @patch('msprobe.pytorch.dump.pytorch_service.register_optimizer_hook')
    def test_register_hook_mix_level(self, mock_register_opt):
        self.service.config.level = Const.LEVEL_MIX
        self.service._register_hook()
        mock_register_opt.assert_called_once_with(self.service.data_collector)
    
    @patch('msprobe.pytorch.dump.pytorch_service.register_optimizer_hook')
    def test_register_hook_not_mix_level(self, mock_register_opt):
        self.service.config.level = Const.LEVEL_L1
        self.service._register_hook()
        mock_register_opt.assert_not_called()
    
    @patch('msprobe.pytorch.dump.pytorch_service.wrap_script_func')
    def test_register_api_hook(self, mock_wrap_jit):
        self.service.config.level = Const.LEVEL_L1
        self.service._register_api_hook()
        mock_wrap_jit.assert_called_once()
        self.service.api_register.initialize_hook.assert_called_once()
    
    def test_register_module_hook(self):
        model_mock = MagicMock()
        self.service.model = model_mock
        self.service._register_module_hook()
  
        self.service.module_processor.register_module_hook.assert_called_once_with(
            model_mock, self.service.build_hook
        )

        self.assertTrue(self.service.module_processor.enable_module_dump)
    
   
    @patch.object(HOOKModule, 'reset_module_stats')
    @patch.object(ModuleProcessor, 'reset_module_stats')
    def test_reset_status(self, mock_reset_module_processor, mock_reset_hook_module):
        self.service._reset_status()
        mock_reset_hook_module.assert_called_once()
        mock_reset_module_processor.assert_called_once()
        self.service.data_collector.reset_status.assert_called_once()
    

    def test_register_module_hook(self):
        self.service.model = MagicMock()
        self.service._register_module_hook()
        self.service.module_processor.register_module_hook.assert_called_once()
