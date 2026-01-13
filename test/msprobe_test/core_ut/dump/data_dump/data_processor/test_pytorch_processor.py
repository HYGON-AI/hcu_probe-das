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
import zlib
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from torch import distributed as dist
from torch._subclasses import FakeTensorMode

from msprobe.core.common.log import logger
from msprobe.core.dump.data_dump.data_processor.pytorch_processor import (
    PytorchDataProcessor,
    TensorDataProcessor,
    TensorStatInfo,
    KernelDumpDataProcessor
)
from msprobe.core.common.log import logger
from msprobe.core.dump.data_dump.data_processor.pytorch_processor import (
    PytorchDataProcessor,
    TensorDataProcessor,
    TensorStatInfo,
    KernelDumpDataProcessor,
    DiffCheckDataProcessor,  TensorHandler # 新增
)



class TestPytorchDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = PytorchDataProcessor(self.config, self.data_writer)

    def test_tensor_bytes_view_cpu_and_crc32(self):
        # 构造一个 CPU contiguous 的 tensor
        t = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        t = t.contiguous()

        mv = PytorchDataProcessor.tensor_bytes_view_cpu(t)
        # 可能返回 memoryview / bytes / numpy.ndarray 都要兼容处理
        if isinstance(mv, np.ndarray):
            expected_crc = zlib.crc32(mv.tobytes())
        else:
            expected_crc = zlib.crc32(mv)

        crc_hex = PytorchDataProcessor.compute_crc32_from_tensor(t)
        self.assertEqual(crc_hex, f"{expected_crc:08x}")

    def test_tensor_bytes_view_cpu_empty_tensor(self):
        t = torch.tensor([], dtype=torch.float32)
        mv = PytorchDataProcessor.tensor_bytes_view_cpu(t)
        # 空 tensor 应该返回空的内存视图/bytes
        if isinstance(mv, np.ndarray):
            self.assertEqual(mv.size, 0)
        else:
            self.assertEqual(len(mv), 0)

    def test_dump_async_data(self):
        # 准备缓存两条 tensor
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        self.processor._async_dump_cache = {
            "path1": t1,
            "path2": t2,
        }

        with patch.object(self.processor.tensor_handler, "save_tensor") as mock_save:
            self.processor.dump_async_data()

        self.assertEqual(mock_save.call_count, 2)
        self.assertEqual(self.processor._async_dump_cache, {})

    def test_get_md5_for_tensor(self):
        tensor = torch.tensor([1, 2, 3])
        expected_hash = zlib.crc32(tensor.numpy().tobytes())
        self.assertEqual(self.processor.get_md5_for_tensor(tensor), f"{expected_hash:08x}")

    def test_get_md5_for_tensor_bfloat16(self):
        tensor_bfloat16 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        expected_hash = zlib.crc32(tensor_bfloat16.float().cpu().detach().numpy().tobytes())
        result_hash = self.processor.get_md5_for_tensor(tensor_bfloat16)
        self.assertEqual(result_hash, f"{expected_hash:08x}")

    def test_analyze_device_in_kwargs(self):
        device = torch.device('cuda:0')
        result = self.processor.analyze_device_in_kwargs(device)
        expected = {'type': 'torch.device', 'value': 'cuda:0'}
        self.assertEqual(result, expected)

    def test_analyze_dtype_in_kwargs(self):
        dtype = torch.float32
        result = self.processor.analyze_dtype_in_kwargs(dtype)
        expected = {'type': 'torch.dtype', 'value': 'torch.float32'}
        self.assertEqual(result, expected)

    @staticmethod
    def mock_tensor(is_meta):
        tensor = MagicMock()
        tensor.is_meta = is_meta
        return tensor

    def test_get_stat_info_with_meta_tensor(self):
        mock_data = self.mock_tensor(is_meta=True)
        result = self.processor.get_stat_info(mock_data)
        self.assertIsInstance(result, TensorStatInfo)

    def test_get_stat_info_with_fake_tensor(self):
        with FakeTensorMode() as fake_tensor_mode:
            fake_tensor = fake_tensor_mode.from_tensor(torch.randn(1, 2, 3))
        result = self.processor.get_stat_info(fake_tensor)
        self.assertIsNone(result.max)
        self.assertIsNone(result.min)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    def test_get_stat_info_float(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, 3.0)
        self.assertEqual(result.min, 1.0)
        self.assertEqual(result.mean, 2.0)
        self.assertEqual(result.norm, torch.norm(tensor).item())

    def test_get_stat_info_int(self):
        tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = self.processor.get_stat_info(tensor)

        self.assertEqual(result.max, 3)
        self.assertEqual(result.min, 1)
        self.assertEqual(result.mean, 2)
        self.assertEqual(result.norm, torch.norm(tensor.float()).item())

    def test_get_stat_info_empty(self):
        tensor = torch.tensor([])
        result = self.processor.get_stat_info(tensor)
        self.assertIsNone(result.max)
        self.assertIsNone(result.min)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    def test_get_stat_info_bool(self):
        tensor = torch.tensor([True, False, True])
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, True)
        self.assertEqual(result.min, False)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    def test_get_stat_info_with_scalar_tensor(self):
        scalar_tensor = torch.tensor(42.0)
        result = self.processor.get_stat_info(scalar_tensor)
        self.assertIsInstance(result, TensorStatInfo)
        self.assertEqual(result.max, 42.0)
        self.assertEqual(result.min, 42.0)
        self.assertEqual(result.mean, 42.0)
        self.assertEqual(result.norm, 42.0)

    def test_get_stat_info_with_complex_tensor(self):
        complex_tensor = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
        result = self.processor.get_stat_info(complex_tensor)
        expected_max = np.abs(np.array([1 + 2j, 3 + 4j])).max().item()
        expected_min = np.abs(np.array([1 + 2j, 3 + 4j])).min().item()
        expected_mean = np.abs(np.array([1 + 2j, 3 + 4j])).mean().item()
        self.assertIsInstance(result, TensorStatInfo)
        self.assertAlmostEqual(result.max, expected_max, places=6)
        self.assertAlmostEqual(result.min, expected_min, places=6)
        self.assertAlmostEqual(result.mean, expected_mean, places=6)

    def test_analyze_builtin(self):
        result = self.processor._analyze_builtin(slice(1, torch.tensor(10, dtype=torch.int32), np.int64(2)))
        expected = {'type': 'slice', 'value': [1, 10, 2]}
        self.assertEqual(result, expected)

        result = self.processor._analyze_builtin(slice(torch.tensor([1, 2], dtype=torch.int32), None, None))
        expected = {'type': 'slice', 'value': [None, None, None]}
        self.assertEqual(result, expected)

    def test_process_group_hash(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend='gloo', world_size=1, rank=0)
        process_group_element = dist.group.WORLD
        result = self.processor.process_group_hash(process_group_element)
        expected = f"{zlib.crc32(str([0]).encode('utf-8')):08x}"
        self.assertEqual(result, expected)
        dist.destroy_process_group()

    def test_analyze_torch_size(self):
        size = torch.Size([3, 4, 5])
        result = self.processor._analyze_torch_size(size)
        expected = {'type': 'torch.Size', 'value': [3, 4, 5]}
        self.assertEqual(result, expected)

    def test_analyze_memory_format(self):
        memory_format_element = torch.contiguous_format
        result = self.processor._analyze_memory_format(memory_format_element)
        expected = {'type': 'torch.memory_format', 'format': 'contiguous_format'}
        self.assertEqual(result, expected)

    def test_analyze_process_group(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend='gloo', world_size=1, rank=0)
        process_group_element = dist.group.WORLD
        result = self.processor._analyze_process_group(process_group_element)
        expected = {
            'type': 'torch.ProcessGroup',
            'group_ranks': [0],
            'group_id': f"{zlib.crc32(str([0]).encode('utf-8')):08x}"
        }
        self.assertEqual(result, expected)
        dist.destroy_process_group()

    def test_analyze_reduce_op_successful(self):
        arg = dist.ReduceOp.SUM
        result = self.processor._analyze_reduce_op(arg)
        expected = {'type': 'torch.distributed.ReduceOp', 'value': 'RedOpType.SUM'}
        self.assertEqual(result, expected)

    @patch.object(logger, 'warning')
    def test_analyze_reduce_op_failed(self, mock_logger_warning):
        class TestReduceOp:
            def __str__(self):
                raise Exception("failed to convert str type")

        arg = TestReduceOp()
        self.processor._analyze_reduce_op(arg)
        mock_logger_warning.assert_called_with(
            "Failed to get value of torch.distributed.ReduceOp with error info: failed to convert str type."
        )

    def test_get_special_types(self):
        special_types = self.processor.get_special_types()
        self.assertIn(torch.Tensor, special_types)

    def test_analyze_single_element_torch_size(self):
        size_element = torch.Size([2, 3])
        result = self.processor.analyze_single_element(size_element, [])
        self.assertEqual(result, self.processor._analyze_torch_size(size_element))

    def test_analyze_single_element_memory_size(self):
        memory_format_element = torch.contiguous_format
        result = self.processor.analyze_single_element(memory_format_element, [])
        self.assertEqual(result, self.processor._analyze_memory_format(memory_format_element))

    def test_analyze_single_element_process_group(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend='gloo', world_size=1, rank=0)
        process_group_element = dist.group.WORLD
        result = self.processor.analyze_single_element(process_group_element, [])
        self.assertEqual(result, self.processor._analyze_process_group(process_group_element))
        dist.destroy_process_group()

    def test_analyze_single_element_numpy_conversion(self):
        numpy_element = np.int32(5)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": 'int32', "value": 5}
        self.assertEqual(result, expected)

        numpy_element = np.float32(3.14)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": 'float32', "value": 3.140000104904175}
        self.assertEqual(result, expected)

        numpy_element = np.bool_(True)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": 'bool_', "value": True}
        self.assertEqual(result, expected)

        numpy_element = np.str_("abc")
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": 'str_', "value": "abc"}
        self.assertEqual(result, expected)

        numpy_element = np.byte(1)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": 'int8', "value": 1}
        self.assertEqual(result, expected)

        numpy_element = np.complex128(1 + 2j)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": 'complex128', "value": (1 + 2j)}
        self.assertEqual(result, expected)

    def test_analyze_single_element_tensor(self):
        tensor_element = torch.tensor([1, 2, 3])
        result = self.processor.analyze_single_element(tensor_element, ['tensor'])
        expected_result = self.processor._analyze_tensor(tensor_element, "tensor")
        self.assertEqual(result, expected_result, f"{result} {expected_result}")

    def test_analyze_single_element_bool(self):
        bool_element = True
        result = self.processor.analyze_single_element(bool_element, [])
        expected_result = self.processor._analyze_builtin(bool_element)
        self.assertEqual(result, expected_result)

    def test_analyze_single_element_builtin_ellipsis(self):
        result = self.processor.analyze_single_element(Ellipsis, [])
        expected_result = self.processor._analyze_builtin(Ellipsis)
        self.assertEqual(result, expected_result)

    @patch.object(PytorchDataProcessor, 'get_md5_for_tensor')
    def test_analyze_tensor(self, get_md5_for_tensor):
        get_md5_for_tensor.return_value = 'mocked_md5'
        tensor = torch.tensor([1.0, 2.0, 3.0])
        self.config.summary_mode = 'md5'
        self.config.async_dump = False
        result = self.processor._analyze_tensor(tensor, 'suffix')
        expected = {
            'type': 'torch.Tensor',
            'dtype': str(tensor.dtype),
            'shape': tensor.shape,
            'requires_grad': tensor.requires_grad
        }
        result.pop('tensor_stat_index', None)
        result.pop('md5_index', None)
        self.assertDictEqual(expected, result)

    def test_analyze_tensor_with_empty_tensor(self):
        tensor = torch.tensor([])
        result = self.processor._analyze_tensor(tensor, 'suffix')

        self.assertEqual(result['type'], "torch.Tensor")
        self.assertEqual(result['dtype'], 'torch.float32')
        self.assertEqual(result['shape'], torch.Size([0]))
        self.assertEqual(result['requires_grad'], False)


class TestTensorDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = TensorDataProcessor(self.config, self.data_writer)
        self.data_writer.dump_tensor_data_dir = "./dump_data"
        self.processor.current_api_or_module_name = "test_api"
        self.processor.api_data_category = "input"

    @patch('torch.save')
    def test_analyze_tensor(self, mock_save):
        self.config.framework = "pytorch"
        self.config.async_dump = False
        tensor = torch.tensor([1.0, 2.0, 3.0])
        suffix = 'suffix'
        result = self.processor._analyze_tensor(tensor, suffix)
        mock_save.assert_called_once()
        expected = {
            'type': 'torch.Tensor',
            'dtype': 'torch.float32',
            'shape': tensor.shape,
            'requires_grad': False,
            'data_name': 'test_api.input.suffix.pt'
        }
        result.pop('tensor_stat_index', None)
        self.assertEqual(expected, result)


class TestKernelDumpDataProcessor(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = KernelDumpDataProcessor(self.config, self.data_writer)

    @patch.object(logger, 'warning')
    def test_print_unsupported_log(self, mock_logger_warning):
        self.processor._print_unsupported_log("test_api_name")
        mock_logger_warning.assert_called_with("The kernel dump does not support the test_api_name API.")

    @patch('msprobe.core.dump.data_dump.data_processor.pytorch_processor.is_gpu')
    @patch.object(logger, 'warning')
    def test_analyze_pre_forward_with_gpu(self, mock_logger_warning, mock_is_gpu):
        mock_is_gpu = True
        self.processor.analyze_forward_input("test_api_name", None, None)
        mock_logger_warning.assert_called_with(
            "The current environment is not a complete NPU environment, and kernel dump cannot be used.")
        self.assertFalse(self.processor.enable_kernel_dump)

    @patch('msprobe.core.dump.data_dump.data_processor.pytorch_processor.is_gpu', new=False)
    @patch('msprobe.core.dump.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.analyze_element')
    @patch.object(logger, 'warning')
    def test_analyze_pre_forward_with_not_gpu(self, mock_logger_warning, mock_analyze_element):
        self.config.is_backward_kernel_dump = True
        mock_module = MagicMock()
        mock_module_input_output = MagicMock()
        self.processor.analyze_forward_input("test_api_name", mock_module, mock_module_input_output)
        mock_module.forward.assert_called_once()
        mock_analyze_element.assert_called()
        mock_logger_warning.assert_called_with("The kernel dump does not support the test_api_name API.")
        self.assertFalse(self.processor.enable_kernel_dump)

    @patch('msprobe.core.dump.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.stop_kernel_dump')
    @patch.object(logger, 'info')
    def test_analyze_forward_successfully(self, mock_logger_info, mock_stop_kernel_dump):
        self.processor.enable_kernel_dump = True
        self.processor.config.is_backward_kernel_dump = False
        self.processor.analyze_forward_output('test_api_name', None, None)
        self.assertFalse(self.processor.enable_kernel_dump)
        mock_stop_kernel_dump.assert_called_once()
        mock_logger_info.assert_called_with("The kernel data of test_api_name is dumped successfully.")

    @patch('msprobe.core.dump.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.analyze_element')
    @patch.object(logger, 'warning')
    def test_analyze_backward_unsuccessfully(self, mock_logger_warning, mock_analyze_element):
        self.processor.enable_kernel_dump = True
        self.processor.is_found_grad_input_tensor = False
        mock_module_input_output = MagicMock()
        self.processor.analyze_backward("test_api_name", None, mock_module_input_output)
        mock_analyze_element.assert_called_once()
        mock_logger_warning.assert_called_with("The kernel dump does not support the test_api_name API.")
        self.assertFalse(self.processor.enable_kernel_dump)

    @patch('msprobe.core.dump.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.stop_kernel_dump')
    @patch('msprobe.core.dump.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.start_kernel_dump')
    @patch('msprobe.core.dump.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.analyze_element')
    @patch.object(logger, 'info')
    def test_analyze_backward_successfully(self, mock_logger_info, mock_analyze_element, mock_start, mock_stop):
        self.processor.enable_kernel_dump = True
        self.processor.is_found_grad_input_tensor = True
        self.processor.forward_output_tensor = MagicMock()
        mock_module_input_output = MagicMock()
        self.processor.analyze_backward("test_api_name", None, mock_module_input_output)
        mock_analyze_element.assert_called_once()
        self.assertFalse(self.processor.enable_kernel_dump)
        self.processor.forward_output_tensor.backward.assert_called_once()
        mock_start.assert_called_once()
        mock_stop.assert_called_once()
        mock_logger_info.assert_called_with("The kernel data of test_api_name is dumped successfully.")

    def test_clone_tensor(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        clone_tensor = self.processor.clone_and_detach_tensor(tensor)
        self.assertTrue(torch.equal(tensor, clone_tensor))
        self.assertFalse(clone_tensor.requires_grad)

        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        clone_tensor = self.processor.clone_and_detach_tensor(tensor)
        self.assertTrue(torch.equal(tensor, clone_tensor))
        self.assertTrue(clone_tensor.requires_grad)

        tensor1 = torch.tensor([1.0], requires_grad=True)
        tensor2 = torch.tensor([1.0])
        input_tuple = (tensor1, tensor2)
        clone_tuple = self.processor.clone_and_detach_tensor(input_tuple)
        self.assertEqual(len(input_tuple), len(clone_tuple))
        self.assertTrue(clone_tuple[0].requires_grad)
        self.assertFalse(clone_tuple[1].requires_grad)

        input_list = [tensor1, tensor2]
        clone_list = self.processor.clone_and_detach_tensor(input_list)
        self.assertEqual(len(input_list), len(clone_list))
        self.assertTrue(clone_tuple[0].requires_grad)
        self.assertFalse(clone_tuple[1].requires_grad)

        input_dict = {'tensor1': tensor1, 'tensor2': tensor2}
        clone_dict = self.processor.clone_and_detach_tensor(input_dict)
        self.assertEqual(len(clone_dict), len(input_dict))
        self.assertTrue(clone_dict["tensor1"].requires_grad)
        self.assertFalse(clone_dict["tensor2"].requires_grad)

        non_tensor_input = 1
        result = self.processor.clone_and_detach_tensor(non_tensor_input)
        self.assertEqual(result, non_tensor_input)

    def test_analyze_single_element_with_output_grad(self):
        self.processor.is_found_output_tensor = False
        tensor = torch.tensor([1.0], requires_grad=True)
        self.processor.analyze_single_element(tensor, None)
        self.assertTrue(self.processor.is_found_output_tensor)

    def test_analyze_single_element_without_output_grad(self):
        self.processor.is_found_output_tensor = False
        tensor = torch.tensor([1.0])
        self.processor.analyze_single_element(tensor, None)
        self.assertFalse(self.processor.is_found_output_tensor)

    def test_analyze_single_element_with_grad_input(self):
        self.processor.is_found_output_tensor = True
        self.processor.is_found_grad_input_tensor = False
        tensor = torch.tensor([1.0])
        self.processor.analyze_single_element(tensor, None)
        self.assertTrue(self.processor.is_found_grad_input_tensor)

    def test_analyze_single_element_without_grad_input(self):
        self.processor.is_found_output_tensor = True
        self.processor.is_found_grad_input_tensor = True
        tensor = torch.tensor([1.0])
        self.processor.analyze_single_element(tensor, None)
        self.assertTrue(self.processor.is_found_grad_input_tensor)

    def test_reset_status(self):
        self.processor.enable_kernel_dump = False
        self.processor.is_found_output_tensor = True
        self.processor.is_found_grad_input_tensor = True
        self.processor.forward_args = 0
        self.processor.forward_kwargs = 1
        self.processor.forward_output_tensor = 2
        self.processor.grad_input_tensor = 3

        self.processor.reset_status()

        self.assertTrue(self.processor.enable_kernel_dump)
        self.assertFalse(self.processor.is_found_output_tensor)
        self.assertFalse(self.processor.is_found_grad_input_tensor)
        self.assertIsNone(self.processor.forward_args)
        self.assertIsNone(self.processor.forward_kwargs)
        self.assertIsNone(self.processor.forward_output_tensor)
        self.assertIsNone(self.processor.grad_input_tensor)


class TestDiffCheckDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        # diff_nums 用于 is_terminated / handle_diff 分支
        self.config.diff_nums = 2
        self.config.precision = MagicMock()
        self.config.async_dump = False
        self.config.summary_mode = MagicMock()
        self.config.task = MagicMock()
        self.data_writer = MagicMock()
        self.processor = DiffCheckDataProcessor(self.config, self.data_writer)

    def test_is_terminated_property(self):
        # diff_nums = -1 时永不终止
        self.processor.diff_nums = -1
        self.processor.real_diff_nums = 100
        self.assertFalse(self.processor.is_terminated)

        # 正常计数
        self.processor.diff_nums = 2
        self.processor.real_diff_nums = 1
        self.assertFalse(self.processor.is_terminated)
        self.processor.real_diff_nums = 2
        self.assertTrue(self.processor.is_terminated)

    def test_parse_data_name(self):
        # 带 name: 前缀
        name = "name:Functional.relu.2.forward.input.0.pt"
        parsed = DiffCheckDataProcessor._parse_data_name(name)
        self.assertEqual(parsed, ("Functional.relu.2.forward", "input", 0))

        # 不带前缀
        name2 = "MyApi.output.3.pt"
        parsed2 = DiffCheckDataProcessor._parse_data_name(name2)
        self.assertEqual(parsed2, ("MyApi", "output", 3))

        # 非法格式
        self.assertIsNone(DiffCheckDataProcessor._parse_data_name("invalid_name"))

    def test_build_bench_map_from_json_and_expected_counts(self):
        data = {
            "MyApi.forward": {
                "input_args": [
                    {"type": "torch.Tensor", "md5": "aaa", "shape": [1, 2]},
                    {"type": "int", "value": 1},  # 非 tensor，应该被忽略
                ],
                "output": [
                    {"type": "torch.Tensor", "md5": "bbb", "shape": [3, 4]},
                    {"type": "str", "value": "xxx"},  # 非 tensor
                ],
            }
        }

        mp = self.processor._build_bench_map_from_json(data)
        self.processor._bench_map = mp

        # input / output 键存在
        self.assertIn(("MyApi.forward", "input", 0), mp)
        self.assertIn(("MyApi.forward", "output", 0), mp)
        # 非 tensor 的条目不会出现在 map 里
        self.assertNotIn(("MyApi.forward", "input", 1), mp)
        self.assertNotIn(("MyApi.forward", "output", 1), mp)

        n_in, n_out = self.processor._bench_expected_counts_for_api("MyApi.forward")
        self.assertEqual(n_in, 1)
        self.assertEqual(n_out, 1)

    def test_ensure_bench_map_loaded_success(self):
        # 模拟 dump.json 存在且能够被加载
        with patch.object(self.processor, "_resolve_bench_json_path", return_value="/fake/dump.json"), \
             patch("os.path.getmtime", return_value=123456), \
             patch("msprobe.core.dump.data_dump.data_processor.pytorch_processor.load_json",
                   return_value={"data": {"Api": {"input_args": [], "output": []}}}) as mock_load, \
             patch.object(self.processor, "_build_bench_map_from_json",
                          return_value={"dummy": {"md5": "xx", "shape": [1]}}) as mock_build:
            ok = self.processor._ensure_bench_map_loaded()

        self.assertTrue(ok)
        mock_load.assert_called_once()
        mock_build.assert_called_once()
        self.assertEqual(self.processor._bench_ref_path, "/fake/dump.json")
        self.assertEqual(self.processor._bench_ref_mtime, 123456)
        self.assertIn("dummy", self.processor._bench_map)

    def test_ensure_bench_map_loaded_path_none(self):
        # 无 bench 路径时直接 False
        with patch.object(self.processor, "_resolve_bench_json_path", return_value=None):
            ok = self.processor._ensure_bench_map_loaded()
        self.assertFalse(ok)

    def test_analyze_maybe_diff_tensor_input_not_in_ref(self):
        # bench map 已加载，但没有对应的 (api, io, idx)
        with patch.object(self.processor, "_ensure_bench_map_loaded", return_value=True):
            self.processor._bench_map = {}
            tensor_json = {
                "data_name": "MyApi.forward.input.0.pt",
                "shape": [1, 2],
                "md5": "0011",
            }
            self.processor._analyze_maybe_diff_tensor(tensor_json)

        st = self.processor._bench_state["MyApi.forward"]
        self.assertFalse(st["inputs_equal"])
        self.assertTrue(st["seen_input_not_in_ref"])
        self.assertFalse(self.processor.has_diff)

    def test_analyze_maybe_diff_tensor_input_in_ref_equal(self):
        # bench 有 input，shape/md5 一致
        self.processor._bench_map = {
            ("MyApi.forward", "input", 0): {"md5": "abcd", "shape": [1, 2]},
        }
        with patch.object(self.processor, "_ensure_bench_map_loaded", return_value=True):
            tensor_json = {
                "data_name": "MyApi.forward.input.0.pt",
                "shape": [1, 2],
                "md5": "abcd",
            }
            self.processor._analyze_maybe_diff_tensor(tensor_json)

        st = self.processor._bench_state["MyApi.forward"]
        self.assertTrue(st["inputs_equal"])
        self.assertEqual(st["checked_in"], 1)
        self.assertEqual(st["expected_in"], 1)
        self.assertFalse(st["seen_input_not_in_ref"])

    def test_analyze_maybe_diff_tensor_output_diff_when_inputs_ok(self):
        # 手动构造 bench_map 和 bench_state，模拟：输入已经全部一致，输出 md5 不一致 -> has_diff = True
        api = "MyApi.forward"
        self.processor._bench_map = {
            (api, "input", 0): {"md5": "in_md5", "shape": [2, 2]},
            (api, "output", 0): {"md5": "out_ref", "shape": [2, 2]},
        }
        self.processor._bench_state[api] = {
            "expected_in": 1,
            "checked_in": 1,
            "inputs_equal": True,
            "seen_input_not_in_ref": False,
            "any_output_neq": False,
        }

        with patch.object(self.processor, "_ensure_bench_map_loaded", return_value=True):
            tensor_json = {
                "data_name": f"{api}.output.0.pt",
                "shape": [2, 2],
                "md5": "out_cur_not_equal",
            }
            self.processor.has_diff = False
            self.processor._analyze_maybe_diff_tensor(tensor_json)

        st = self.processor._bench_state[api]
        self.assertTrue(st["any_output_neq"])
        self.assertTrue(self.processor.has_diff)

    def test_handle_diff_and_clear_bench_state(self):
        # 准备 has_diff = True，且有缓存的 tensor
        fake_tensor = torch.tensor([1.0])
        self.processor.cached_tensors_and_file_paths = {
            "/tmp/a.pt": fake_tensor,
        }
        self.processor.has_diff = True
        self.processor.real_diff_nums = 0
        self.processor.diff_nums = 2
        self.processor.current_api_or_module_name = "TestApi"
        self.processor._bench_state["TestApi"] = {"dummy": 1}

        with patch.object(self.processor.tensor_handler, "save_tensor") as mock_save:
            self.processor.handle_diff()

        # 有 diff 时会调用 save_tensor，并且 real_diff_nums +1，bench_state 对应 api 被清掉
        mock_save.assert_called_once()
        self.assertEqual(self.processor.real_diff_nums, 1)
        self.assertNotIn("TestApi", self.processor._bench_state)
        self.assertEqual(self.processor.cached_tensors_and_file_paths, {})
