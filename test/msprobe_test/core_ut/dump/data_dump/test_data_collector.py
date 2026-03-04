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

import os

import unittest
from unittest.mock import patch, MagicMock

from msprobe.core.common.utils import Const
from msprobe.core.dump.data_dump.data_collector import DataCollector
from msprobe.pytorch.dump.debugger.debugger_config import DebuggerConfig
from msprobe.pytorch.dump.pt_config import parse_json_config


class TestDataCollector(unittest.TestCase):
    def setUp(self):
        mock_json_data = {
            "dump_path": "./ut_dump",
        }
        with patch("msprobe.pytorch.dump.pt_config.load_json", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json", Const.STATISTICS)
        config = DebuggerConfig(common_config, task_config, Const.STATISTICS, "./ut_dump", "L1")
        self.data_collector = DataCollector(config)

    def test_dump_data_dir(self):
        self.assertEqual(self.data_collector.dump_data_dir, None)

        self.data_collector.data_writer.dump_tensor_data_dir = "./test_dump"
        self.assertEqual(self.data_collector.dump_data_dir, "./test_dump")

    def test_dump_file_path(self):
        self.assertEqual(self.data_collector.dump_file_path, None)

        self.data_collector.data_writer.dump_file_path = "./test_dump/dump.json"
        self.assertEqual(self.data_collector.dump_file_path, "./test_dump/dump.json")

    def test_scope_none_and_pid_match(self):
        mock_name = "test_module"
        current_pid = os.getpid()
        result = self.data_collector.check_scope_and_pid(None, mock_name, current_pid)
        self.assertTrue(result)

    def test_scope_valid_and_pid_match(self):
        mock_scope = MagicMock()
        mock_scope.check.return_value = True
        mock_name = "valid_module"
        current_pid = os.getpid()
        result = self.data_collector.check_scope_and_pid(mock_scope, mock_name, current_pid)
        self.assertTrue(result)
        mock_scope.check.assert_called_once_with(mock_name)

    def test_scope_invalid_and_pid_match(self):
        mock_scope = MagicMock()
        mock_scope.check.return_value = False
        mock_name = "invalid_module"
        current_pid = os.getpid()
        result = self.data_collector.check_scope_and_pid(mock_scope, mock_name, current_pid)
        self.assertFalse(result)

    def test_scope_valid_but_pid_mismatch(self):
        mock_scope = MagicMock()
        mock_scope.check.return_value = True
        mock_name = "valid_module"
        fake_pid = os.getpid() + 1
        result = self.data_collector.check_scope_and_pid(mock_scope, mock_name, fake_pid)
        self.assertFalse(result)

    def test_scope_none_but_pid_mismatch(self):
        mock_name = "test_module"
        fake_pid = os.getpid() + 1
        result = self.data_collector.check_scope_and_pid(None, mock_name, fake_pid)
        self.assertFalse(result)

    def test_normal_case(self):
        data_info = {"key1": {"other_field": "value"}}
        self.data_collector.set_is_recomputable(data_info, True)
        self.assertTrue(data_info["key1"]["is_recompute"])

        self.data_collector.set_is_recomputable(data_info, False)
        self.assertFalse(data_info["key1"]["is_recompute"])

    def test_empty_data_info(self):
        data_info = {}
        original_data = data_info.copy()
        self.data_collector.set_is_recomputable(data_info, True)
        self.assertEqual(data_info, original_data)

    def test_data_info_length_not_one(self):
        data_info = {"key1": {}, "key2": {}}
        original_data = data_info.copy()
        self.data_collector.set_is_recomputable(data_info, True)
        self.assertEqual(data_info, original_data)

    def test_is_recompute_none(self):
        data_info = {"key1": {}}
        original_data = data_info.copy()
        self.data_collector.set_is_recomputable(data_info, None)
        self.assertEqual(data_info, original_data)

    def test_nested_structure(self):
        data_info = {"layer1": {"sub_layer": {"value": 1}}}
        self.data_collector.set_is_recomputable(data_info, True)
        self.assertTrue(data_info["layer1"]["is_recompute"])
        self.assertEqual(data_info["layer1"]["sub_layer"]["value"], 1)

    def test_reset_status(self):
        self.data_collector.optimizer_status = "test_optimizer_status"
        self.data_collector.reset_status()

        self.assertEqual(self.data_collector.optimizer_status, "")
        self.assertEqual(
            self.data_collector.optimizer_status_first_start,
            {Const.OPTIMIZER: True, Const.CLIP_GRAD: True}
        )
        self.assertEqual(self.data_collector.backward_module_names, {})

    def test_update_api_or_module_name(self):
        self.assertEqual(self.data_collector.data_processor.current_api_or_module_name, None)

        self.data_collector.update_api_or_module_name("test_api_name")
        self.assertEqual(self.data_collector.data_processor.current_api_or_module_name, "test_api_name")

    def test_write_json(self):
        self.data_collector.data_writer = MagicMock()

        self.data_collector.write_json()
        self.data_collector.data_writer.write_json.assert_called_once()

    def test_write_json_at_exit_with_async_dump_tensor(self):
        self.data_collector.data_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()
        self.data_collector.config.async_dump = True
        self.data_collector.config.task = "tensor"

        self.data_collector.write_json_at_exit()

        self.data_collector.data_processor.dump_async_data.assert_called_once()
        self.data_collector.data_writer.write_json.assert_called_once()

    def test_write_json_at_exit_with_no_async_dump(self):
        self.data_collector.data_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()
        self.data_collector.config.async_dump = False
        self.data_collector.config.task = "tensor"

        self.data_collector.write_json_at_exit()

        self.data_collector.data_processor.dump_async_data.assert_not_called()
        self.data_collector.data_writer.write_json.assert_called_once()

    def test_write_json_at_exit_with_statistics(self):
        self.data_collector.data_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()
        self.data_collector.config.async_dump = True
        self.data_collector.config.task = "statistics"

        self.data_collector.write_json_at_exit()

        self.data_collector.data_processor.dump_async_data.assert_not_called()
        self.data_collector.data_writer.write_json.assert_called_once()

    def test_call_stack_collect(self):
        self.data_collector.data_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()

        test_name = "test_api"
        mock_stack = "stack_info", False
        data_info = {}
        self.data_collector.data_processor.analyze_api_call_stack.return_value = mock_stack

        self.data_collector.call_stack_collect(data_info, test_name)

        self.data_collector.data_processor.analyze_api_call_stack.assert_called_once()
        self.data_collector.data_writer.update_stack.assert_called_once_with(test_name, "stack_info")

    def test_update_construct_without_construct(self):
        self.data_collector.data_writer = MagicMock()

        self.data_collector.config.level = "L1"
        self.data_collector.update_construct("test")
        self.data_collector.data_writer.update_construct.assert_not_called()

    def test_update_construct_with_first_start(self):
        self.data_collector.module_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()
        self.data_collector.config.level = "L0"
        self.data_collector.optimizer_status = "optimizer"
        self.data_collector.optimizer_status_first_start = {"optimizer": True}

        self.data_collector.update_construct("test_name")
        calls = [
            unittest.mock.call({"optimizer": None}),
            unittest.mock.call({"test_name": "optimizer"}),
            unittest.mock.call(self.data_collector.module_processor.module_node)
        ]
        self.data_collector.data_writer.update_construct.assert_has_calls(calls)

    def test_update_construct_with_not_first_start(self):
        self.data_collector.module_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()
        self.data_collector.config.level = "L0"
        self.data_collector.optimizer_status = "clip_grad"
        self.data_collector.optimizer_status_first_start = {"clip_grad": False}

        self.data_collector.update_construct("test_name")
        calls = [
            unittest.mock.call({"test_name": "clip_grad"}),
            unittest.mock.call(self.data_collector.module_processor.module_node)
        ]
        self.data_collector.data_writer.update_construct.assert_has_calls(calls)

    def test_update_construct_with_module_prefix(self):
        self.data_collector.module_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()
        self.data_collector.config.level = "mix"
        self.data_collector.optimizer_status = "other_status"
        test_name = "Module_test_name"

        self.data_collector.update_construct(test_name)
        self.data_collector.data_writer.update_construct.assert_called_with(
            self.data_collector.module_processor.module_node
        )

    def test_handle_data(self):
        with patch.object(DataCollector, "update_data") as mock_update_data, \
                patch.object(DataCollector, "write_json") as mock_write_json, \
                patch("msprobe.core.dump.data_dump.json_writer.DataWriter.flush_data_periodically") as mock_flush:
            self.data_collector.handle_data("Tensor.add", {"min": 0})
            mock_update_data.assert_called_with("Tensor.add", {"min": 0})

            mock_flush.assert_called()
            mock_write_json.assert_not_called()

            mock_update_data.reset_mock()
            mock_flush.reset_mock()
            self.data_collector.handle_data("Tensor.add", {}, flush=True)
            mock_update_data.assert_not_called()
            mock_flush.assert_not_called()
            mock_write_json.assert_called()


class TestForwardDataCollect(unittest.TestCase):
    def setUp(self):
        mock_json_data = {
            "dump_path": "./test_fwd_dump",
        }
        with patch("msprobe.pytorch.dump.pt_config.load_json", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json", Const.STATISTICS)
        config = DebuggerConfig(common_config, task_config, Const.STATISTICS, "./test_fwd_dump", "L1")
        self.data_collector = DataCollector(config)

        self.data_collector.update_construct = MagicMock()
        self.data_collector.config = MagicMock()
        self.data_collector.data_processor = MagicMock()
        self.data_collector.scope = "test_scope"
        self.data_collector.check_scope_and_pid = MagicMock()
        self.data_collector._should_collect_by_risk_level = MagicMock()
        self.data_collector.set_is_recomputable = MagicMock()
        self.data_collector.handle_data = MagicMock()
        self.data_collector.call_stack_collect = MagicMock()

        self.Const = MagicMock()
        self.Const.FREE_BENCHMARK = "free_benchmark"
        self.Const.TENSOR = "tensor"
        self.Const.FORWARD = "forward"
        self.Const.BACKWARD = "backward"
        self.Const.STRUCTURE = "structure"
        self.Const.LEVEL_L2 = "L2"

    def test_forward_input_with_scope_pid_check_fail(self):
        self.data_collector.config.task = self.Const.TENSOR
        self.data_collector.check_scope_and_pid.return_value = False
        self.data_collector._should_collect_by_risk_level.return_value = False
        self.data_collector.forward_input_data_collect(
            "test", "module1", 123, "input_output"
        )

        self.data_collector.data_processor.analyze_forward_input.assert_not_called()

    def test_forward_input_with_structure_task(self):
        self.data_collector.config.task = self.Const.STRUCTURE
        self.data_collector.check_scope_and_pid.return_value = True

        self.data_collector.forward_input_data_collect(
            "test", "module1", 123, "input_output"
        )

        self.data_collector.data_processor.analyze_forward_input.assert_not_called()
        self.data_collector.call_stack_collect.assert_called_once_with({}, "test")

    def test_forward_input_with_level_l2(self):
        self.data_collector.config.task = self.Const.TENSOR
        self.data_collector.config.level = self.Const.LEVEL_L2
        self.data_collector.check_scope_and_pid.return_value = True

        self.data_collector.forward_input_data_collect(
            "test", "module1", 123, "input_output"
        )

        self.data_collector.handle_data.assert_not_called()

    def test_forward_input_with_recompute(self):
        self.data_collector.config.task = self.Const.TENSOR
        self.data_collector.config.level = "L1"
        self.data_collector.check_scope_and_pid.return_value = True
        mock_data = {"key": "value"}
        self.data_collector.data_processor.analyze_forward_input.return_value = mock_data
        self.data_collector.data_processor.is_recompute.return_value = True
        self.data_collector.forward_input_data_collect(
            "test", "module1", 123, "input_output"
        )

        self.data_collector.call_stack_collect.assert_called_once_with(mock_data, "test")
        self.data_collector.handle_data.assert_called_once_with(
            "test", mock_data, flush=self.data_collector.data_processor.is_terminated
        )

    def test_forward_output_with_scope_check_fail(self):
        self.data_collector.check_scope_and_pid.return_value = False
        self.data_collector._should_collect_by_risk_level.return_value = False
        self.data_collector.forward_output_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_forward_output.assert_not_called()

    def test_forward_output_with_structure_task(self):
        self.data_collector.config.task = self.Const.STRUCTURE
        self.data_collector.forward_output_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_forward_output.assert_not_called()

    def test_forward_output_with_level_l2(self):
        self.data_collector.config.level = self.Const.LEVEL_L2
        self.data_collector.forward_output_data_collect("test", "module", 123, "data")
        self.data_collector.handle_data.assert_not_called()

    def test_forward_output_normal(self):
        mock_data = {"key": "value"}
        self.data_collector.data_processor.analyze_forward_output.return_value = mock_data
        self.data_collector.forward_output_data_collect("test", "module", 123, "data")
        self.data_collector.handle_data.assert_called_once_with(
            "test",
            mock_data,
            flush=self.data_collector.data_processor.is_terminated
        )

    def test_forward_with_scope_check_fail(self):
        self.data_collector.check_scope_and_pid.return_value = False
        self.data_collector._should_collect_by_risk_level.return_value = False
        self.data_collector.forward_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_forward.assert_not_called()

    def test_forward_with_structure_task(self):
        self.data_collector.config.task = self.Const.STRUCTURE
        self.data_collector.forward_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_forward.assert_not_called()

    def test_forward_normal(self):
        mock_data = {"key": "value"}
        self.data_collector.data_processor.analyze_forward.return_value = mock_data
        self.data_collector.forward_data_collect("test", "module", 123, "data")
        self.data_collector.call_stack_collect.assert_called_once_with(mock_data, "test")
        self.data_collector.handle_data.assert_called_once_with(
            "test",
            mock_data,
            flush=self.data_collector.data_processor.is_terminated
        )


class TestBackwardDataCollector(unittest.TestCase):
    def setUp(self):
        mock_json_data = {
            "dump_path": "./test_bwd_dump",
        }
        with patch("msprobe.pytorch.dump.pt_config.load_json", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json", Const.STATISTICS)
        config = DebuggerConfig(common_config, task_config, Const.STATISTICS, "./test_bwd_dump", "L1")
        self.data_collector = DataCollector(config)

        self.data_collector.config = MagicMock()
        self.data_collector.data_processor = MagicMock()
        self.data_collector.scope = "test_scope"
        self.data_collector.check_scope_and_pid = MagicMock(return_value=True)
        self.data_collector._should_collect_by_risk_level = MagicMock(return_value=True)
        self.data_collector.set_is_recomputable = MagicMock()
        self.data_collector.handle_data = MagicMock()
        self.data_collector.update_construct = MagicMock()
        self.data_collector.backward_module_names = {}

        self.Const = MagicMock()
        self.Const.STRUCTURE = "structure"
        self.Const.TENSOR = "tensor"
        self.Const.LEVEL_L2 = "L2"
        self.Const.SEP = "."
        self.Const.MODULE_PREFIX = ["module"]

    def test_backward_with_scope_check_fail(self):
        self.data_collector.check_scope_and_pid.return_value = False
        self.data_collector._should_collect_by_risk_level.return_value = False
        self.data_collector.backward_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_backward.assert_not_called()

    def test_backward_with_level_l2(self):
        self.data_collector.config.level = self.Const.LEVEL_L2
        self.data_collector.backward_data_collect("test", "module", 123, "data")
        self.data_collector.handle_data.assert_not_called()

    def test_backward_data_module_prefix_match(self):
        self.data_collector.check_scope_and_pid.return_value = True
        self.data_collector.config.task = self.Const.TENSOR
        self.data_collector.config.level = "L1"
        mock_data = {"key": "value"}
        self.data_collector.data_processor.analyze_backward.return_value = mock_data
        test_name = "Module.layer1.backward"
        self.data_collector.backward_data_collect(test_name, "module", 123, "data")
        self.assertEqual(self.data_collector.backward_module_names, {"Module": True})

    def test_backward_input_with_structure_task(self):
        self.data_collector.config.task = self.Const.STRUCTURE
        self.data_collector.backward_input_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_backward_input.assert_not_called()

    def test_backward_input_with_normal(self):
        mock_data = {"key": "value"}
        self.data_collector.data_processor.analyze_backward_input.return_value = mock_data
        self.data_collector.backward_input_data_collect("test", "module", 123, "data")

    def test_backward_output_with_scope_check_fail(self):
        self.data_collector.check_scope_and_pid.return_value = False
        self.data_collector._should_collect_by_risk_level.return_value = False
        self.data_collector.backward_output_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_backward_output.assert_not_called()

    def test_backward_output_with_recompute(self):
        mock_data = {"key": "value"}
        self.data_collector.data_processor.analyze_backward_output.return_value = mock_data
        self.data_collector.backward_output_data_collect("test", "module", 123, "data")
