# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import mindspore as ms
from mindspore import Tensor
from mindspore.ops.operations import _inner_ops

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.runtime import Runtime
from msprobe.core.dump.data_dump.scope import ModuleRangeScope
from msprobe.core.dump.hook_manager import HookSet
from msprobe.mindspore.dump.cell_processor import CellProcessor, get_cell_construct
from msprobe.mindspore.common.log import logger


class TestCellProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CellProcessor.reset_cell_stats()
        cls.scope = MagicMock(spec=ModuleRangeScope)
        cls.processor = CellProcessor(cls.scope)

    @classmethod
    def tearDownClass(cls):
        CellProcessor.reset_cell_stats()

    def test_class_attribute(self):
        self.assertTrue(hasattr(CellProcessor, 'cell_count'))
        self.assertTrue(hasattr(CellProcessor, 'cell_stack'))
        self.assertTrue(hasattr(CellProcessor, 'api_parent_node'))
        self.assertTrue(hasattr(CellProcessor, 'module_node'))
        self.assertTrue(hasattr(CellProcessor, 'cell_bw_hook_kernels'))
        self.assertTrue(hasattr(CellProcessor, 'cell_backward_pre_hook'))
        self.assertTrue(hasattr(CellProcessor, 'cell_backward_hook'))

    def test__init(self):
        self.assertIsInstance(self.processor.scope, ModuleRangeScope)
        processor = CellProcessor(None)
        self.assertIsNone(processor.scope)

    def test_get_cell_construct(self):
        def construct(self, *args, **kwargs):
            return len(args)

        _constrct = get_cell_construct(construct)
        ret = _constrct(self, 'argument')
        self.assertFalse(hasattr(self, 'msprobe_input_kwargs'))
        self.assertEqual(ret, 1)

        setattr(self, 'msprobe_hook', True)
        _constrct = get_cell_construct(construct)
        ret = _constrct(self, 'argument')
        self.assertEqual(self.msprobe_input_kwargs, {})
        self.assertEqual(ret, 1)

        del self.msprobe_hook
        del self.msprobe_input_kwargs

    def test_set_and_get_calls_number(self):
        CellProcessor.cell_count = {}
        count = self.processor.set_and_get_calls_number("cell")
        self.assertEqual(count, 0)
        self.assertEqual(CellProcessor.cell_count["cell"], 0)

        count = self.processor.set_and_get_calls_number("cell")
        self.assertEqual(count, 1)
        self.assertEqual(CellProcessor.cell_count["cell"], 1)

        CellProcessor.cell_count = {}

    def test_reset_cell_stats(self):
        CellProcessor.cell_count['cell'] = 0
        CellProcessor.cell_stack['tid'] = 'cell'
        CellProcessor.api_parent_node['tid'] = 'cell'
        CellProcessor.module_node['cell'] = 'null'
        CellProcessor.cell_bw_hook_kernels['cell'] = 'bw'
        CellProcessor.cell_backward_pre_hook.append('backward_pre_hook')
        CellProcessor.cell_backward_hook.append('backward_hook')

        CellProcessor.reset_cell_stats()
        self.assertEqual(CellProcessor.cell_count, {})
        self.assertEqual(CellProcessor.cell_stack, {})
        self.assertEqual(CellProcessor.api_parent_node, {})
        self.assertEqual(CellProcessor.module_node, {})
        self.assertEqual(CellProcessor.cell_bw_hook_kernels, {})
        self.assertEqual(CellProcessor.cell_backward_pre_hook, [])
        self.assertEqual(CellProcessor.cell_backward_hook, [])

    def test_set_construct_info_in_pre_hook(self):
        CellProcessor.reset_cell_stats()
        self.processor.set_construct_info_in_pre_hook('full_name')
        self.assertEqual(CellProcessor.module_node['full_name'], None)
        self.assertEqual(CellProcessor.cell_stack[threading.get_ident()], ['full_name'])
        self.assertEqual(CellProcessor.api_parent_node[threading.get_ident()], 'full_name')
        self.scope.begin_module.assert_called_with('full_name')

        self.scope.begin_module.reset_mock()
        self.processor.set_construct_info_in_pre_hook('sub_cell_name')
        self.assertEqual(CellProcessor.module_node, {'full_name': None, 'sub_cell_name': 'full_name'})
        self.assertEqual(CellProcessor.cell_stack[threading.get_ident()], ['full_name', 'sub_cell_name'])
        self.assertEqual(CellProcessor.api_parent_node[threading.get_ident()], 'sub_cell_name')
        self.scope.begin_module.assert_called_with('sub_cell_name')

        CellProcessor.reset_cell_stats()

    def test_set_construct_info_in_hook(self):
        CellProcessor.reset_cell_stats()
        self.processor.set_construct_info_in_hook('full_name')
        self.assertIsNone(CellProcessor.api_parent_node[threading.get_ident()])
        self.scope.end_module.assert_called_with('full_name')

        self.scope.end_module.reset_mock()
        CellProcessor.cell_stack[threading.get_ident()] = ['full_name']
        self.processor.set_construct_info_in_hook('full_name')
        self.assertEqual(CellProcessor.cell_stack, {threading.get_ident(): []})
        self.assertIsNone(CellProcessor.api_parent_node[threading.get_ident()])
        self.scope.end_module.assert_called_with('full_name')

        self.scope.end_module.reset_mock()
        CellProcessor.cell_stack[threading.get_ident()] = ['Cell.0', 'Cell.1']
        self.processor.set_construct_info_in_hook('full_name')
        self.assertEqual(CellProcessor.cell_stack, {threading.get_ident():['Cell.0']})
        self.assertEqual(CellProcessor.api_parent_node[threading.get_ident()], 'Cell.0')
        self.scope.end_module.assert_called_with('full_name')

        CellProcessor.reset_cell_stats()
