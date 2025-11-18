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
import zlib
from unittest.mock import patch, MagicMock

import mindspore as ms
import numpy as np
from mindspore import Tensor, ops, mint

from msprobe.core.dump.data_dump.data_processor.mindspore_processor import (
    MindsporeDataProcessor,
    TensorDataProcessor,
    KernelDumpDataProcessor,
)
from msprobe.mindspore.common.log import logger


def patch_norm(value):
    return ops.norm(value)


setattr(mint, "norm", patch_norm)


def get_tensor_layout(tensor):
    layout_dic = {}
    if hasattr(tensor.layout, "device_matrix") and tensor.layout.device_matrix is not None:
        layout_dic['Device Matrix'] = tensor.layout.device_matrix
    if hasattr(tensor.layout, "alias_name") and tensor.layout.alias_name is not None: 
        layout_dic['Alias Name'] = tensor.layout.alias_name
        interleaved = "Yes" if ("interleaved_parallel" in tensor.layout.alias_name) else "No"
        layout_dic['Interleaved'] = interleaved
    if hasattr(tensor.layout, "partial") and tensor.layout.partial is not None: 
        layout_dic['Partial'] = tensor.layout.partial
    if hasattr(tensor.layout, "tensor_map") and tensor.layout.tensor_map is not None: 
        layout_dic['Tensor Map'] = tensor.layout.tensor_map
    if hasattr(tensor.layout, "rank_list") and tensor.layout.rank_list is not None: 
        layout_dic['Rank List'] = tensor.layout.rank_list
    tensor_json['layout'] = layout_dic


class TestMindsporeDataProcessor(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = MindsporeDataProcessor(self.config, self.data_writer)
        self.tensor = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))

    def test_get_md5_for_tensor(self):
        tensor = ms.Tensor([1.0, 2.0, 3.0], dtype=ms.bfloat16)
        expected_crc32 = zlib.crc32(np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes())
        expected_md5 = f"{expected_crc32:08x}"
        result = self.processor.get_md5_for_tensor(tensor)
        self.assertEqual(result, expected_md5)

    def test_analyze_builtin(self):
        test_slice = slice(1, 3, None)
        expected_result = {"type": "slice", "value": [1, 3, None]}
        result = self.processor._analyze_builtin(test_slice)
        self.assertEqual(result, expected_result)

        test_int = 42
        expected_result = {"type": "int", "value": 42}
        result = self.processor._analyze_builtin(test_int)
        self.assertEqual(result, expected_result)

    def test_get_stat_info_float(self):
        self.config.async_dump = False
        tensor = ms.Tensor([1.0, 2.0, 3.0])
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, 3.0)
        self.assertEqual(result.min, 1.0)
        self.assertEqual(result.mean, 2.0)
        self.assertEqual(result.norm, ms.ops.norm(tensor).item())

    def test_get_stat_info_float_async(self):
        self.config.async_dump = True
        tensor = ms.tensor([1.0, 2.0, 3.0])
        result = self.processor.get_stat_info(tensor)
        result_max = result.max
        result_min = result.min
        result_mean = result.mean
        result_norm = result.norm

        self.assertEqual(result_max.item(), 3.0)
        self.assertEqual(result_min.item(), 1.0)
        self.assertEqual(result_mean.item(), 2.0)
        self.assertEqual(result_norm.item(), ms.ops.norm(tensor).item())

    def test_get_stat_info_int(self):
        self.config.async_dump = False
        tensor = ms.Tensor([1, 2, 3], dtype=ms.int32)
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, 3)
        self.assertEqual(result.min, 1)
        self.assertEqual(result.mean, 2)
        self.assertEqual(result.norm, ms.ops.norm(tensor).item())

    def test_get_stat_info_int_async(self):
        self.config.async_dump = True
        tensor = ms.tensor([1, 2, 3])
        result = self.processor.get_stat_info(tensor)

        result_max = result.max
        result_min = result.min

        self.assertEqual(result_max.item(), 3.0)
        self.assertEqual(result_min.item(), 1.0)

    def test_get_stat_info_bool(self):
        self.config.async_dump = False
        tensor = ms.Tensor([True, False, True])
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, True)
        self.assertEqual(result.min, False)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    def test_get_stat_info_bool_async(self):
        self.config.async_dump = True
        tensor = ms.Tensor([True, False, True])
        result = self.processor.get_stat_info(tensor)

        result_max = result.max
        result_min = result.min

        self.assertEqual(result_max.item(), True)
        self.assertEqual(result_min.item(), False)

    @patch.object(MindsporeDataProcessor, 'get_md5_for_tensor')
    def test__analyze_tensor(self, get_md5_for_tensor):
        get_md5_for_tensor.return_value = "test_md5"
        tensor = ms.Tensor(np.array([1, 2, 3], dtype=np.int32))
        self.config.summary_mode = 'md5'
        self.config.async_dump = False
        suffix = "test_tensor"
        expected_result = {
            'type': 'mindspore.Tensor',
            'dtype': 'Int32',
            'shape': (3,)
        }
        if hasattr(tensor, "layout") and tensor.layout is not None:
            expected_result['layout'] = get_tensor_layout(tensor)
        if hasattr(tensor, "hsdp_effective_shard_size") and tensor.hsdp_effective_shard_size is not None:
            expected_result['hsdp_shard_size'] = tensor.hsdp_effective_shard_size
        result = self.processor._analyze_tensor(tensor, suffix)
        # 删除不必要的字段
        result.pop('tensor_stat_index', None)
        result.pop('md5_index', None)

        self.assertEqual(result, expected_result)


class TestTensorDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = TensorDataProcessor(self.config, self.data_writer)
        self.data_writer.dump_tensor_data_dir = "./dump_data"
        self.processor.current_api_or_module_name = "test_api"
        self.processor.api_data_category = "input"

    @patch('msprobe.core.dump.data_dump.data_processor.mindspore_processor.save_tensor_as_npy')
    def test_analyze_tensor(self, mock_save):
        self.config.framework = "mindspore"
        self.config.async_dump = False
        tensor = ms.Tensor([1.0, 2.0, 3.0])
        suffix = 'suffix'
        result = self.processor._analyze_tensor(tensor, suffix)
        mock_save.assert_called_once()
        expected = {
            'type': 'mindspore.Tensor',
            'dtype': str(tensor.dtype),
            'shape': tensor.shape,
            'data_name': 'test_api.input.suffix.npy'
        }
        if hasattr(tensor, "layout") and tensor.layout is not None:
            expected['layout'] = get_tensor_layout(tensor)
        if hasattr(tensor, "hsdp_effective_shard_size") and tensor.hsdp_effective_shard_size is not None:
            expected['hsdp_shard_size'] = tensor.hsdp_effective_shard_size    
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

    @patch('msprobe.core.dump.data_dump.data_processor.mindspore_processor.KernelDumpDataProcessor.start_kernel_dump')
    @patch('msprobe.core.dump.data_dump.data_processor.mindspore_processor.has_adump', new=True)
    def test_analyze_pre_forward_with_adump(self, mock_start_kernel_dump):
        self.processor.analyze_forward_input("test_api_name", None, None)
        mock_start_kernel_dump.assert_called_once()
        self.assertTrue(self.processor.enable_kernel_dump)

    @patch('msprobe.core.dump.data_dump.data_processor.mindspore_processor.has_adump', new=False)
    @patch.object(logger, 'warning')
    def test_analyze_pre_forward_without_adump(self, mock_logger_warning):
        self.processor.enable_kernel_dump = True
        self.processor.analyze_forward_input("test_api_name", None, None)
        mock_logger_warning.assert_called_with(
            "The current msprobe package does not compile adump, and kernel dump cannot be used.")
        self.assertFalse(self.processor.enable_kernel_dump)

    @patch('msprobe.core.dump.data_dump.data_processor.mindspore_processor.KernelDumpDataProcessor.stop_kernel_dump')
    @patch.object(logger, 'info')
    def test_analyze_forward_successfully(self, mock_logger_info, mock_stop_kernel_dump):
        self.processor.enable_kernel_dump = True
        self.processor.analyze_forward_output('test_api_name', None, None)
        self.assertFalse(self.processor.enable_kernel_dump)
        mock_stop_kernel_dump.assert_called_once()
        mock_logger_info.assert_called_with("The kernel data of test_api_name is dumped successfully.")

    @patch('msprobe.core.dump.data_dump.data_processor.mindspore_processor.has_adump', new=True)
    @patch('msprobe.core.dump.data_dump.data_processor.mindspore_processor.KernelDumpDataProcessor.start_kernel_dump')
    def test_analyze_pre_backward_with_adump(self, mock_start_kernel_dump):
        self.processor.enable_kernel_dump = True
        self.processor.analyze_backward_input("test_api_name", None, None)
        self.assertTrue(self.processor.enable_kernel_dump)
        mock_start_kernel_dump.assert_called_once()

    @patch('msprobe.core.dump.data_dump.data_processor.mindspore_processor.has_adump', new=False)
    @patch.object(logger, 'warning')
    def test_analyze_pre_backward_without_adump(self, mock_logger_warning):
        self.processor.enable_kernel_dump = True
        self.processor.analyze_backward_input("test_api_name", None, None)
        self.assertFalse(self.processor.enable_kernel_dump)
        mock_logger_warning.assert_called_with(
            "The current msprobe package does not compile adump, and kernel dump cannot be used.")

    @patch('msprobe.core.dump.data_dump.data_processor.mindspore_processor.KernelDumpDataProcessor.stop_kernel_dump')
    @patch.object(logger, 'info')
    def test_analyze_backward_successfully(self, mock_logger_info, mock_stop_kernel_dump):
        self.processor.enable_kernel_dump = True
        self.processor.analyze_backward('test_api_name', None, None)
        self.assertFalse(self.processor.enable_kernel_dump)
        mock_stop_kernel_dump.assert_called_once()
        mock_logger_info.assert_called_with("The kernel data of test_api_name is dumped successfully.")

    def test_reset_status(self):
        self.processor.enable_kernel_dump = False
        self.processor.reset_status()
        self.assertTrue(self.processor.enable_kernel_dump)
