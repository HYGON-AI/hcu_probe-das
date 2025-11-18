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

import sys
import threading
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock

import torch
from torch.utils.checkpoint import _StopRecomputationError

import msprobe.pytorch.dump.module_dump.module_processor as mp
from msprobe.core.dump.data_dump.scope import ModuleRangeScope
from msprobe.pytorch.dump.module_dump.module_processor import (
    ModuleProcessor,
    wrap_megatron_deallocate,
    wrap_forward_with_hook_safety
)

ori_checkpoint = torch.utils.checkpoint.checkpoint


class TestModule(torch.nn.Module):
    """测试用的模块类，可控制是否抛出异常"""

    def __init__(self, raise_exception=False):
        super().__init__()
        self.raise_exception = raise_exception

    def forward(self, x, *args, **kwargs):
        if self.raise_exception:
            raise _StopRecomputationError()
        return x * 2


def ModuleProcessor_forward_hook_fn(module, args, kwargs_or_output, output_or_kwargs=None):
    print(f"The forward_hook executed normally.")


class TestWrapper(unittest.TestCase):
    def setUp(self):
        torch.utils.checkpoint.checkpoint = ori_checkpoint
        self.held_output = StringIO()
        self.original_stdout = sys.stdout
        sys.stdout = self.held_output

    def tearDown(self):
        """恢复标准输出"""
        sys.stdout = self.original_stdout

    def get_output(self):
        """获取捕获的输出内容"""
        return self.held_output.getvalue().strip()

    def test_wrap_megatron_deallocate(self):
        mock_func = MagicMock(return_value="output_test")
        wrapped = wrap_megatron_deallocate(mock_func)

        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor._base = True
        mock_tensor.device = "cpu"
        mock_tensor.dtype = torch.float32
        mock_tensor.clone.return_value = "cloned"

        result = wrapped(mock_tensor, deallocate_pipeline_outputs=True)
        mock_tensor.clone.assert_called_once()
        self.assertEqual(mock_tensor.data.shape, (1,))
        self.assertEqual(result, "output_test")
        mock_func.assert_called_once_with("cloned", True)

        result = wrapped("normal_input", False)
        self.assertEqual(result, "output_test")
        mock_func.assert_called_with("normal_input", False)

    def test_normal_forward_execution(self):
        """测试正常执行forward时的情况"""
        # 准备测试模块和hook
        module = TestModule(raise_exception=False)
        module.register_forward_hook(ModuleProcessor_forward_hook_fn)

        # 应用包装函数
        wrap_forward_with_hook_safety(module)

        # 执行forward
        input_tensor = torch.tensor(3.0)
        output = module(input_tensor)

        # 验证结果和hook调用
        self.assertEqual(output.item(), 6.0)
        self.assertIn("The forward_hook executed normally.", self.get_output())

    def test_stop_recomputation_exception_triggers_hook(self):
        """测试抛出_StopRecomputationError时hook被调用"""
        # 准备测试模块和hook
        module = TestModule(raise_exception=True)
        module.register_forward_hook(ModuleProcessor_forward_hook_fn)

        # 应用包装函数
        wrap_forward_with_hook_safety(module)

        # 执行forward并验证异常
        input_tensor = torch.tensor(3.0)
        with self.assertRaises(_StopRecomputationError):
            module(input_tensor)

        self.assertIn("The forward_hook executed normally.", self.get_output())


class TestModuleProcessor(unittest.TestCase):
    def setUp(self):
        ModuleProcessor.module_count = {}
        ModuleProcessor.module_stack = {}
        ModuleProcessor.module_node = {}
        ModuleProcessor.api_parent_node = {}

        self.scope = ModuleRangeScope([], [])
        self.mock_scope = MagicMock()

    @patch('msprobe.pytorch.dump.module_dump.module_processor.wrap_setup_input_output_hook')
    def test_init_with_valid_scope(self, mock_wrap):
        processor = ModuleProcessor(self.scope)
        self.assertEqual(processor.scope, self.scope)
        mock_wrap.assert_called_once()

    @patch('msprobe.pytorch.dump.module_dump.module_processor.logger.info_on_rank_0')
    def test_init_without_megatron(self, mock_log):
        ModuleProcessor(self.scope)
        mock_log.assert_called_with("No megatron find.")

    def test_set_and_get_calls_number(self):
        count = ModuleProcessor.set_and_get_calls_number("test_module")
        self.assertEqual(count, 0)

        count = ModuleProcessor.set_and_get_calls_number("test_module")
        self.assertEqual(count, 1)

    def test_has_register_backward_hook(self):
        module1 = torch.nn.Linear(10, 10)
        self.assertFalse(ModuleProcessor.has_register_backward_hook(module1))

        module2 = MagicMock()
        module2._backward_hooks = [1, 2, 3]
        module2._is_full_backward_hook = False
        self.assertTrue(ModuleProcessor.has_register_backward_hook(module2))

        module2._is_full_backward_hook = True
        self.assertFalse(ModuleProcessor.has_register_backward_hook(module2))

    def test_get_modules_and_names_with_model_list(self):
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_model1.named_modules.return_value = [("layer1", "obj1"), ("layer2", "obj2")]
        mock_model2.named_modules.return_value = [("layer3", "obj3")]

        result = ModuleProcessor.get_modules_and_names(
            [mock_model1, mock_model2],
            recursive=True,
            module_names=["model1", "model2"]
        )
        self.assertEqual(result, {
            "0": [("layer1", "obj1"), ("layer2", "obj2")],
            "1": [("layer3", "obj3")]
        })

    def test_get_modules_and_names_with_model_tuple(self):
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_model1.named_modules.return_value = [("layer1", "obj1")]
        mock_model2.named_modules.return_value = [("layer2", "obj2")]

        result = ModuleProcessor.get_modules_and_names(
            (mock_model1, mock_model2),
            recursive=True,
            module_names=["model1", "model2"]
        )
        self.assertEqual(result, {
            "0": [("layer1", "obj1")],
            "1": [("layer2", "obj2")]
        })

    def test_get_modules_and_names_with_single_recursive(self):
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [("layer1", "obj1")]

        result = ModuleProcessor.get_modules_and_names(
            mock_model,
            recursive=True,
            module_names=["single_model"]
        )
        self.assertEqual(result, {
            "-1": [("layer1", "obj1")]
        })

    def test_get_modules_and_names_with_single_non_recursive(self):
        mock_model = MagicMock()
        result = ModuleProcessor.get_modules_and_names(
            mock_model,
            recursive=False,
            module_names=["single_model"]
        )
        self.assertEqual(result, {
            "-1": [("single_model", mock_model)]
        })

    def test_get_modules_and_names_invalid_case(self):
        result = ModuleProcessor.get_modules_and_names(
            [MagicMock(), MagicMock()],
            recursive=False,
            module_names=["only_one_name"]
        )
        self.assertEqual(result, {})

        result = ModuleProcessor.get_modules_and_names(
            MagicMock(),
            recursive=False,
            module_names=["name1", "name2"]
        )
        self.assertEqual(result, {})

    def test_reset_module_stats(self):
        ModuleProcessor.module_count = {"test": 1}
        ModuleProcessor.module_stack = ["layer1"]
        ModuleProcessor.api_parent_node = "parent"
        ModuleProcessor.module_node = {"key": "value"}
        ModuleProcessor.module_bw_hook_kernels = {"hook": "data"}
        ModuleProcessor.enable_module_dump = True

        ModuleProcessor.reset_module_stats()

        self.assertEqual(ModuleProcessor.module_count, {})
        self.assertEqual(ModuleProcessor.module_stack, {})
        self.assertEqual(ModuleProcessor.api_parent_node, {})
        self.assertEqual(ModuleProcessor.module_node, {})
        self.assertEqual(ModuleProcessor.module_bw_hook_kernels, {})
        self.assertFalse(ModuleProcessor.enable_module_dump)

    def test_set_construct_info_in_pre_hook_with_stack(self):
        processor = ModuleProcessor(self.mock_scope)
        ModuleProcessor.module_stack[threading.get_ident()] = ["parent_module"]
        processor.scope = self.mock_scope

        processor.set_construct_info_in_pre_hook("current_module")

        self.assertEqual(ModuleProcessor.module_node["current_module"], "parent_module")
        self.assertEqual(
            ModuleProcessor.module_stack[threading.get_ident()],
            ["parent_module", "current_module"]
        )
        self.assertEqual(ModuleProcessor.api_parent_node[threading.get_ident()], "current_module")
        self.mock_scope.begin_module.assert_called_once_with("current_module")

    def test_set_construct_info_in_pre_hook_empty_stack(self):
        processor = ModuleProcessor(self.mock_scope)
        processor.scope = self.mock_scope
        processor.set_construct_info_in_pre_hook("root_module")

        self.assertIsNone(ModuleProcessor.module_node["root_module"])
        self.assertEqual(ModuleProcessor.module_stack[threading.get_ident()], ["root_module"])
        self.assertEqual(ModuleProcessor.api_parent_node[threading.get_ident()], "root_module")

    def test_set_construct_info_in_hook_with_forward(self):
        mp.torch_version_above_or_equal_2 = True
        processor = ModuleProcessor(self.mock_scope)
        ModuleProcessor.module_stack = {threading.get_ident(): ["parent", "current"]}
        processor.scope = self.mock_scope

        processor.set_construct_info_in_hook("current")

        self.assertEqual(ModuleProcessor.module_stack[threading.get_ident()], ["parent"])
        self.assertEqual(ModuleProcessor.api_parent_node[threading.get_ident()], "parent")
        self.mock_scope.end_module.assert_called_once_with("current")

    def test_set_construct_info_in_hook_with_backward(self):
        mp.torch_version_above_or_equal_2 = False
        processor = ModuleProcessor(self.mock_scope)
        processor.scope = self.mock_scope

        processor.set_construct_info_in_hook("backward_module", is_forward=False)

        self.assertEqual(ModuleProcessor.api_parent_node[threading.get_ident()], "backward_module")
        self.mock_scope.begin_module.assert_called_once_with("backward_module")

    def test_set_construct_info_in_hook_empty_stack(self):
        mp.torch_version_above_or_equal_2 = True
        processor = ModuleProcessor(self.mock_scope)

        processor.set_construct_info_in_hook("module")

        self.assertEqual(ModuleProcessor.api_parent_node, {threading.get_ident(): None})
