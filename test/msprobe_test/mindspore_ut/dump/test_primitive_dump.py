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

from collections import defaultdict
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops

from msprobe.core.common.utils import Const
from msprobe.mindspore.dump.mindspore_service import MindsporeService
from msprobe.core.common.runtime import Runtime
from msprobe.core.dump.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_processor.hook_cell.hook_cell import HOOKCell
from msprobe.mindspore.dump.dump_processor.hook_cell.primitive_hooks import PrimitiveHookService
from msprobe.mindspore.dump.ms_config import StatisticsConfig


class TestPrimitiveHookService(unittest.TestCase):
    def setUp(self):
        # 创建一个临时目录作为 dump_path
        self.temp_dir = tempfile.TemporaryDirectory()
        dump_path = self.temp_dir.name
        json_config = {
            "task": "statistics",
            "dump_path": dump_path,
            "rank": [],
            "step": [0, 2],
            "level": "L1"
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        config = DebuggerConfig(common_config, task_config)

        with patch('msprobe.core.dump.service.build_data_collector'), \
             patch('msprobe.mindspore.dump.mindspore_service.CellProcessor'), \
             patch('msprobe.mindspore.dump.mindspore_service.PrimitiveHookService'), \
             patch('msprobe.mindspore.dump.mindspore_service.get_api_register'):
            self.mock_service_instance = MindsporeService(config)
            Runtime.is_running = True
            self.primitive_hook_service = PrimitiveHookService(self.mock_service_instance)

    def tearDown(self):
        # 测试结束时删除临时目录
        self.temp_dir.cleanup()

    def test_two_input_backward_hook(self):
        # 模拟梯度输入
        captured_grads = []
        num_tensors = 2
        updated_primitive_name = "test_primitive_output"
        hook_type = Const.INPUT

        # 调用 wrap_primitive 获取包装函数通过闭包显式调用backward_hook
        hook_primitive_inputs = self.primitive_hook_service.wrap_primitive(None, "example").__closure__[0].cell_contents

        create_backward_hook = hook_primitive_inputs.__closure__[0].cell_contents

        backward_hook = create_backward_hook(captured_grads, num_tensors, updated_primitive_name, hook_type)
        # 模拟 ndarray 梯度
        grad_1 = np.array([1.0, 2.0, 3.0])  # 模拟第一个梯度
        grad_2 = np.array([4.0, 5.0, 6.0])  # 模拟第二个梯度

        # 模拟反向梯度
        backward_hook(grad_1)
        self.assertEqual(len(captured_grads), 3)  # 只捕获了一个梯度

        backward_hook(grad_2)
        self.assertEqual(len(captured_grads), 6)  # 捕获到两个梯度

        # 调用到达阈值，验证数据收集
        self.assertTrue(self.mock_service_instance.data_collector.backward_output_data_collect.called)

    def test_four_input_backward_hook(self):
        # 模拟梯度输入
        captured_grads = []
        num_tensors = 4
        updated_primitive_name = "test_primitive_output"
        hook_type = Const.INPUT

        # 调用 wrap_primitive 获取包装函数通过闭包显式调用backward_hook
        hook_primitive_inputs = self.primitive_hook_service.wrap_primitive(None, "example").__closure__[0].cell_contents

        create_backward_hook = hook_primitive_inputs.__closure__[0].cell_contents

        backward_hook = create_backward_hook(captured_grads, num_tensors, updated_primitive_name, hook_type)

        # 模拟 ndarray 梯度
        grad_1 = np.array([1.0, 2.0, 3.0])  # 模拟第一个梯度
        grad_2 = np.array([4.0, 5.0, 6.0])  # 模拟第二个梯度
        grad_3 = np.array([7.0, 8.0, 9.0])  # 模拟第三个梯度
        grad_4 = np.array([10.0, 11.0, 12.0])  # 模拟第四个梯度

        # 模拟反向梯度
        backward_hook(grad_1)
        self.assertEqual(len(captured_grads), 3)  # 只捕获了一个梯度

        backward_hook(grad_2)
        self.assertEqual(len(captured_grads), 6)  # 捕获到两个梯度

        backward_hook(grad_3)
        self.assertEqual(len(captured_grads), 9)  # 捕获到三个梯度

        backward_hook(grad_4)
        self.assertEqual(len(captured_grads), 12)  # 捕获到四个梯度

        # 调用到达阈值，验证数据收集
        self.assertTrue(self.mock_service_instance.data_collector.backward_output_data_collect.called)

    def test_two_output_backward_hook(self):
        # 模拟梯度输入
        captured_grads = []
        num_tensors = 2
        updated_primitive_name = "test_primitive_output"
        hook_type = Const.OUTPUT

        # 调用 wrap_primitive 获取包装函数通过闭包显式调用backward_hook
        hook_primitive_inputs = self.primitive_hook_service.wrap_primitive(None, "example").__closure__[0].cell_contents

        create_backward_hook = hook_primitive_inputs.__closure__[0].cell_contents

        backward_hook = create_backward_hook(captured_grads, num_tensors, updated_primitive_name, hook_type)
        # 模拟 ndarray 梯度
        grad_1 = np.array([1.0, 2.0, 3.0])  # 模拟第一个梯度
        grad_2 = np.array([4.0, 5.0, 6.0])  # 模拟第二个梯度

        # 模拟反向梯度
        backward_hook(grad_1)
        self.assertEqual(len(captured_grads), 3)  # 只捕获了一个梯度

        backward_hook(grad_2)
        self.assertEqual(len(captured_grads), 6)  # 捕获到两个梯度

        # 调用到达阈值，验证数据收集
        self.assertTrue(self.mock_service_instance.data_collector.backward_input_data_collect.called)

    def test_four_output_backward_hook(self):
        # 模拟梯度输入
        captured_grads = []
        num_tensors = 4
        updated_primitive_name = "test_primitive_output"
        hook_type = Const.OUTPUT

        # 调用 wrap_primitive 获取包装函数通过闭包显式调用backward_hook
        hook_primitive_inputs = self.primitive_hook_service.wrap_primitive(None, "example").__closure__[0].cell_contents

        create_backward_hook = hook_primitive_inputs.__closure__[0].cell_contents

        backward_hook = create_backward_hook(captured_grads, num_tensors, updated_primitive_name, hook_type)
        # 模拟 ndarray 梯度
        grad_1 = np.array([1.0, 2.0, 3.0])  # 模拟第一个梯度
        grad_2 = np.array([4.0, 5.0, 6.0])  # 模拟第二个梯度
        grad_3 = np.array([7.0, 8.0, 9.0])  # 模拟第三个梯度
        grad_4 = np.array([10.0, 11.0, 12.0])  # 模拟第四个梯度

        # 模拟反向梯度
        backward_hook(grad_1)
        self.assertEqual(len(captured_grads), 3)  # 只捕获了一个梯度

        backward_hook(grad_2)
        self.assertEqual(len(captured_grads), 6)  # 捕获到两个梯度

        backward_hook(grad_3)
        self.assertEqual(len(captured_grads), 9)  # 捕获到三个梯度

        backward_hook(grad_4)
        self.assertEqual(len(captured_grads), 12)  # 捕获到四个梯度

        # 调用到达阈值，验证数据收集
        self.assertTrue(self.mock_service_instance.data_collector.backward_input_data_collect.called)

    def test_hook_primitive_inputs(self):
        # 模拟前向输入
        args = (Tensor(np.array([1.0, 2.0]), ms.float32), Tensor(np.array([3.0, 4.0]), ms.float32))
        captured_grads_input = []
        updated_primitive_name = "test_primitive_input"

        # 调用 hook_primitive_inputs
        hook_primitive_inputs = self.primitive_hook_service.wrap_primitive(None, "example").__closure__[0].cell_contents
        with patch.object(ops, 'HookBackward') as mock_HookBackward:
            target_value = Tensor([1.0])
            mock_hbw = mock_HookBackward.return_value
            mock_hbw.return_value = target_value
            hooked_inputs = hook_primitive_inputs(args, captured_grads_input, updated_primitive_name)
            self.assertEqual(mock_HookBackward.call_count, len(args))
            for hooked_input in hooked_inputs:
                self.assertTrue((hooked_input == target_value).all())

    def test_hook_primitive_outputs(self):
        # 模拟前向输出
        out = (Tensor(np.array([1.0, 2.0]), ms.float32), Tensor(np.array([3.0, 4.0]), ms.float32))
        captured_grads_output = []
        updated_primitive_name = "test_primitive_output"

        # 调用 hook_primitive_outputs
        hook_primitive_outputs = self.primitive_hook_service.wrap_primitive(None,
                                                                            "example").__closure__[1].cell_contents
        with patch.object(ops, 'HookBackward') as mock_HookBackward:
            target_value = Tensor([1.0])
            mock_hbw = mock_HookBackward.return_value
            mock_hbw.return_value = target_value
            hooked_outputs = hook_primitive_outputs(out, captured_grads_output, updated_primitive_name)
            self.assertEqual(mock_HookBackward.call_count, len(out))
            for hooked_output in hooked_outputs:
                self.assertTrue((hooked_output == target_value).all())

    def test_wrapped_primitive_call_args(self):
        # 模拟前向输入
        args = (Tensor(np.array([1.0, 2.0]), ms.float32), Tensor(np.array([3.0, 4.0]), ms.float32))
        captured_grads_input = []
        updated_primitive_name = "test_primitive_args"

        # 获取 wrapped_primitive_call 函数
        wrapped_primitive_call = self.primitive_hook_service.wrap_primitive(lambda x, y: x + y, "add")

        # 调用 wrapped_primitive_call 并检查 hooked_inputs 是否与原始 args 相同
        try:
            with patch.object(ops, 'HookBackward') as mock_HookBackward:
                target_value = Tensor([1.0])
                mock_hbw = mock_HookBackward.return_value
                mock_hbw.return_value = target_value
                hooked_inputs = wrapped_primitive_call.__closure__[0].cell_contents(args, captured_grads_input,
                                                                                    updated_primitive_name)
                self.assertEqual(mock_HookBackward.call_count, len(args))
                for hooked_input in hooked_inputs:
                    self.assertTrue((hooked_input == target_value).all())
        except Exception as e:
            self.fail(f"wrapped_primitive_call raised an exception: {e}")

    def test_update_primitive_counters_multiple(self):
        # 测试更新 primitive 计数器的功能，增加多个不同名称的测试
        primitive_names = ["MatMul", "Conv2D", "ReLU", "Softmax"]

        for name in primitive_names:
            for i in range(3):
                self.primitive_hook_service.update_primitive_counters(name)
                self.assertEqual(self.primitive_hook_service.primitive_counters[name], i)

    @patch('msprobe.mindspore.dump.dump_processor.hook_cell.primitive_hooks.ops.HookBackward')
    def test_wrap_primitive_forward_hook_various_inputs(self, mock_hook_backward):
        # 测试不同形状和大小的 Tensor 输入
        input_tensors = [
            Tensor(np.random.randn(2, 2).astype(np.float32)),
            Tensor(np.random.randn(4, 4).astype(np.float32)),
            Tensor(np.random.randn(10, 10).astype(np.float32)),
        ]

        for input_tensor in input_tensors:
            mock_origin_func = Mock(return_value=input_tensor)
            wrapped_func = self.primitive_hook_service.wrap_primitive(mock_origin_func, "MatMul")

            result = wrapped_func(Mock(), input_tensor)

            mock_origin_func.assert_called_once()
            self.assertIsInstance(result, Tensor)

    def test_wrap_primitive_no_hook_with_invalid_input(self):
        # 测试在 switch 关闭时传入无效输入时的行为
        Runtime.is_running = False

        invalid_inputs = [None, "invalid_tensor", 123]

        for invalid_input in invalid_inputs:
            mock_origin_func = Mock(return_value=invalid_input)
            wrapped_func = self.primitive_hook_service.wrap_primitive(mock_origin_func, "MatMul")

            result = wrapped_func(Mock(), invalid_input)
            mock_origin_func.assert_called_once()
            self.assertEqual(result, invalid_input)

    @patch('msprobe.mindspore.dump.dump_processor.hook_cell.primitive_hooks.ops.HookBackward')
    def test_wrap_primitive_with_multiple_hooks(self, mock_hook_backward):
        # 测试多个钩子函数同时应用的行为
        input_tensor = Tensor(np.random.randn(2, 2).astype(np.float32))

        # 模拟多个 primitive
        primitive_names = ["MatMul", "Add", "Sub"]

        for name in primitive_names:
            mock_origin_func = Mock(return_value=input_tensor)
            wrapped_func = self.primitive_hook_service.wrap_primitive(mock_origin_func, name)
            result = wrapped_func(Mock(), input_tensor)

            mock_origin_func.assert_called_once()
            self.assertIsInstance(result, Tensor)

    @patch('msprobe.mindspore.dump.dump_processor.hook_cell.primitive_hooks.ops.HookBackward')
    def test_wrap_primitive_with_exception_handling_multiple(self, mock_hook_backward):
        # 模拟多个异常情况并确保它们被正确捕获
        input_tensor = Tensor(np.random.randn(2, 2).astype(np.float32))

        exception_messages = ["Invalid operation", "Null reference", "Type error"]

        for exception_message in exception_messages:
            mock_origin_func = Mock(side_effect=Exception(exception_message))
            wrapped_func = self.primitive_hook_service.wrap_primitive(mock_origin_func, "MatMul")

            with self.assertRaises(Exception) as context:
                wrapped_func(Mock(), input_tensor)
            self.assertIn(exception_message, str(context.exception))

    def test_create_backward_hook_multiple(self):
        # 测试创建多个 backward 钩子并模拟不同数量的梯度捕获
        captured_grads_sets = [[Mock()], [Mock(), Mock()], [Mock(), Mock(), Mock()]]

        for captured_grads in captured_grads_sets:
            updated_primitive_name = "MatMul.Backward"
            hook = self.primitive_hook_service.wrap_primitive(Mock(), "MatMul")

            backward_hook = hook(Mock(), captured_grads, updated_primitive_name, Const.INPUT)
            self.assertIsNotNone(backward_hook)

    @patch('msprobe.mindspore.dump.dump_processor.hook_cell.primitive_hooks.ops.HookBackward')
    def test_wrap_primitive_forward_and_backward_hooks(self, mock_hook_backward):
        # 模拟前向和后向钩子在同一个 primitive 中的行为
        input_tensor = Tensor(np.random.randn(2, 2).astype(np.float32))

        mock_origin_func = Mock(return_value=input_tensor)
        wrapped_func = self.primitive_hook_service.wrap_primitive(mock_origin_func, "Conv2D")

        result = wrapped_func(Mock(), input_tensor)

        # 确保前向和后向 hook 均被调用
        mock_origin_func.assert_called_once()

        self.assertIsInstance(result, Tensor)

    def test_update_primitive_counters_different_names(self):
        # 测试不同 primitive 名称的计数器更新
        primitive_names = ["MatMul", "Add", "Sub", "Mul", "Conv2D"]

        for name in primitive_names:
            for i in range(5):
                self.primitive_hook_service.update_primitive_counters(name)
                self.assertEqual(self.primitive_hook_service.primitive_counters[name], i)

    def test_update_primitive_counters(self):
        primitive_name = "MatMul"
        self.primitive_hook_service.update_primitive_counters(primitive_name)
        self.assertEqual(self.primitive_hook_service.primitive_counters[primitive_name], 0)
        self.primitive_hook_service.update_primitive_counters(primitive_name)
        self.assertEqual(self.primitive_hook_service.primitive_counters[primitive_name], 1)

    @patch('msprobe.mindspore.dump.dump_processor.hook_cell.primitive_hooks.ops.HookBackward')
    def test_wrap_primitive_forward_hook(self, mock_hook_backward):
        # 模拟一个 Tensor 输入
        input_tensor = Tensor(np.random.randn(2, 2).astype(np.float32))

        # 模拟原始函数
        mock_origin_func = Mock(return_value=input_tensor)

        # 包装原始 primitive
        wrapped_func = self.primitive_hook_service.wrap_primitive(mock_origin_func, "MatMul")

        # 调用包装后的 primitive
        result = wrapped_func(Mock(), input_tensor)

        # 确保原始函数被调用
        mock_origin_func.assert_called_once()

        # 检查返回值是否是 Mock 实例
        self.assertIsInstance(result, Tensor)
        #
        # # 确保 HookBackward 被应用
        # mock_hook_backward.assert_called()

    @patch('msprobe.mindspore.dump.dump_processor.hook_cell.primitive_hooks.ops.HookBackward')
    def test_wrap_primitive_backward_hook(self, mock_hook_backward):
        # 模拟 Tensor 输入和输出
        input_tensor = Tensor(np.random.randn(2, 2).astype(np.float32))
        grad_tensor = Tensor(np.random.randn(2, 2).astype(np.float32))

        # 确保 HookBackward 返回一个可调用对象，该对象返回 Tensor
        mock_hook_backward.return_value = lambda x: grad_tensor

        # 模拟原始函数
        mock_origin_func = Mock(return_value=input_tensor)

        # 包装 primitive
        wrapped_func = self.primitive_hook_service.wrap_primitive(mock_origin_func, "MatMul")

        # 模拟反向传播过程，调用包装的 primitive
        with patch.object(self.mock_service_instance.data_collector, 'backward_data_collect'):
            result = wrapped_func(Mock(), input_tensor)

            # 验证结果是 Tensor 实例
            self.assertIsInstance(result, Tensor)

    def test_wrap_primitive_no_hook_when_switch_off(self):
        # 模拟 switch 关闭的情况
        Runtime.is_running = False

        # 模拟 Tensor 输入
        input_tensor = Tensor(np.random.randn(2, 2).astype(np.float32))

        # 模拟原始函数
        mock_origin_func = Mock(return_value=input_tensor)

        # 包装 primitive
        wrapped_func = self.primitive_hook_service.wrap_primitive(mock_origin_func, "MatMul")

        # 调用包装后的 primitive
        result = wrapped_func(Mock(), input_tensor)

        # 确保在 switch 关闭时不应用 hook
        mock_origin_func.assert_called_once()
        HOOKCell.cell_count = defaultdict(int)
        self.assertTrue((result == input_tensor).all())  # 使用 .all() 来比较 Tensor

    @patch('msprobe.mindspore.dump.dump_processor.hook_cell.primitive_hooks.ops.HookBackward')
    def test_wrap_primitive_error_handling(self, mock_hook_backward):
        # 模拟 Tensor 输入
        input_tensor = Tensor(np.random.randn(2, 2).astype(np.float32))

        # 模拟抛出异常的原始函数
        mock_origin_func = Mock(side_effect=Exception("Mocked exception"))

        # 包装 primitive
        wrapped_func = self.primitive_hook_service.wrap_primitive(mock_origin_func, "MatMul")

        # 验证是否正确捕获异常
        with self.assertRaises(Exception) as context:
            wrapped_func(Mock(), input_tensor)
        self.assertIn("Mocked exception", str(context.exception))

    @patch('msprobe.mindspore.dump.dump_processor.hook_cell.primitive_hooks.ops.HookBackward')
    def test_create_backward_hook(self, mock_hook_backward):
        # 测试 create_backward_hook 的功能
        captured_grads = []
        updated_primitive_name = "MatMul.Backward"

        # 创建 backward hook
        backward_hook = self.primitive_hook_service.wrap_primitive(Mock(), "MatMul")
        hook = backward_hook(Mock(), captured_grads, updated_primitive_name, Const.INPUT)

        # 确保 hook 被创建并可调用
        self.assertIsNotNone(hook)
