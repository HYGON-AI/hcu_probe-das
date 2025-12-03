# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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

import importlib
import os
import sys
import time
from unittest import TestCase
from unittest.mock import patch, MagicMock

import pandas as pd
import torch

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.log import logger
from msprobe.core.compare import atb_data_compare
from msprobe.core.compare.atb_data_compare import (
    TensorBinFile, get_comparison_mode,
    convert_str_to_float, convert_str_to_int,
    compare_single_tensor, COMMON_HEADER_STATS, COMMON_HEADER,
    cal_single_tensor_stats, cal_comparison_metrics_by_tensor,
    TENSOR_SPECIAL_HEADER, cal_comparison_metrics_by_stats,
    STATS_SPECIAL_HEADER, get_stats_map, compare_atb_mode
)


class TestTensorBinFile(TestCase):
    bin_file = None

    @classmethod
    def setUpClass(cls):
        with patch.object(TensorBinFile, '_parse_bin_file'):
            cls.bin_file = TensorBinFile('file_path')

    def test_bin_file_init(self):
        with patch.object(TensorBinFile, '_parse_bin_file') as mock__parse_bin_file:
            bin_file = TensorBinFile('file_path')
        self.assertEqual(bin_file.file_path, 'file_path')
        self.assertEqual(bin_file.real_data_path, '')
        self.assertEqual(bin_file.dtype, 0)
        self.assertEqual(bin_file.format, 0)
        self.assertEqual(bin_file.dims, [])
        self.assertEqual(bin_file.file_path, 'file_path')
        self.assertTrue(bin_file.is_valid)
        mock__parse_bin_file.assert_called_once_with()
        default_dtype_dict = {
            0: torch.float32,
            1: torch.float16,
            2: torch.int8,
            3: torch.int32,
            9: torch.int64,
            12: torch.bool,
            27: torch.bfloat16
        }
        self.assertEqual(bin_file.dtype_dict, default_dtype_dict)

    @patch.object(logger, 'warning')
    def test_get_data(self, mock_log_warning):
        tensor = torch.zeros(1)
        self.bin_file.is_valid = False
        self.bin_file.real_data_path = ''
        ret = self.bin_file.get_data()
        self.assertEqual(ret, tensor)

        self.bin_file.is_valid = False
        self.bin_file.real_data_path = 'data_path'
        ret = self.bin_file.get_data()
        self.assertEqual(ret, tensor)

        self.bin_file.is_valid = True
        self.bin_file.real_data_path = 'data_path'
        ret = self.bin_file.get_data()
        self.assertEqual(ret, tensor)

        self.bin_file.is_valid = True
        self.bin_file.real_data_path = ''
        self.bin_file.dtype = -1
        ret = self.bin_file.get_data()
        mock_log_warning.assert_called_with(f'Unsupported dtype: {self.bin_file.dtype}')
        self.assertFalse(self.bin_file.is_valid)
        self.assertEqual(ret, tensor)

        self.bin_file.is_valid = True
        self.bin_file.dtype = 3
        self.bin_file.obj_buffer = b'\x01\x00\x00\x00'
        self.bin_file.dims = [1]
        ret = self.bin_file.get_data()
        self.assertTrue((ret == torch.tensor([1], dtype=torch.int32)).all())

        self.bin_file.dims = [2]
        ret = self.bin_file.get_data()
        mock_log_warning.assert_called_with(f'Can not convert {self.bin_file.file_path} to PyTorch Tensor')
        self.assertFalse(self.bin_file.is_valid)
        self.assertEqual(ret, tensor)

    def test__parse_bin_file(self):
        self.bin_file.is_valid = True
        self.bin_file._parse_bin_file()
        self.assertFalse(self.bin_file.is_valid)

        self.bin_file.is_valid = True
        bin_content = bytearray(
            b'$Version=1.0\ndtype=27\nformat=1\ndims=2,2\n$End=1\n\x00\x00\x80\x3f')
        with patch('msprobe.core.compare.atb_data_compare.get_file_content_bytes', return_value=bin_content):
            self.bin_file._parse_bin_file()
            self.assertTrue(self.bin_file.is_valid)
            self.assertEqual(self.bin_file.dtype, 27)
            self.assertEqual(self.bin_file.format, 1)
            self.assertEqual(self.bin_file.dims, [2, 2])
            self.assertEqual(self.bin_file.obj_buffer,
                             bytearray(b'\x00\x00\x80\x3f'))

        self.bin_file.is_valid = True
        bin_content = bytearray(b'dtype:27\n')
        with patch('msprobe.core.compare.atb_data_compare.get_file_content_bytes', return_value=bin_content):
            self.bin_file._parse_bin_file()
            self.assertFalse(self.bin_file.is_valid)

        self.bin_file.is_valid = True
        bin_content = bytearray(b'dims=2,n\nop=Add\n')
        with patch('msprobe.core.compare.atb_data_compare.get_file_content_bytes', return_value=bin_content):
            self.bin_file._parse_bin_file()
            self.assertEqual(self.bin_file.dims, [2, -1])
            self.assertFalse(self.bin_file.is_valid)

    def test__parse_user_attr(self):
        self.bin_file.is_valid = True
        self.bin_file.real_data_path = ''
        self.bin_file._parse_user_attr('data', 'data')
        self.assertFalse(self.bin_file.is_valid)
        self.assertEqual(self.bin_file.real_data_path, 'data')

        self.bin_file.is_valid = True
        self.bin_file._parse_user_attr('dtype', '1')
        self.assertTrue(self.bin_file.is_valid)
        self.assertEqual(self.bin_file.dtype, 1)
        self.bin_file._parse_user_attr('dtype', 'a')
        self.assertFalse(self.bin_file.is_valid)
        self.assertEqual(self.bin_file.dtype, -1)

        self.bin_file.is_valid = True
        self.bin_file._parse_user_attr('format', '1')
        self.assertTrue(self.bin_file.is_valid)
        self.assertEqual(self.bin_file.format, 1)
        self.bin_file._parse_user_attr('format', 'a')
        self.assertFalse(self.bin_file.is_valid)
        self.assertEqual(self.bin_file.format, -1)

        self.bin_file.is_valid = True
        self.bin_file._parse_user_attr('dims', '2,1')
        self.assertTrue(self.bin_file.is_valid)
        self.assertEqual(self.bin_file.dims, [2, 1])
        self.bin_file._parse_user_attr('dims', '2,a')
        self.assertFalse(self.bin_file.is_valid)
        self.assertEqual(self.bin_file.dims, [2, -1])


class TestAtbDataCompareFunctions(TestCase):
    def test_torch_available(self):
        def get_env(*args, **kwargs):
            raise ImportError

        ori_torch = sys.modules['torch']
        sys.modules.pop('torch')
        with patch.object(os, 'getenv', new=get_env):
            importlib.reload(atb_data_compare)
            self.assertFalse(atb_data_compare.torch_available)

        sys.modules['torch'] = ori_torch
        importlib.reload(atb_data_compare)
        self.assertTrue(atb_data_compare.torch_available)

    def test_get_comparison_mode(self):
        mode = get_comparison_mode(['row0'])
        self.assertEqual(mode, CompareConst.UNKNOWN_COMPARISION_MODE)

        mode = get_comparison_mode(['row0', ''])
        self.assertEqual(mode, CompareConst.UNKNOWN_COMPARISION_MODE)

        mode = get_comparison_mode([['header'], ['N/A']])
        self.assertEqual(mode, CompareConst.STATS_COMPARISION_MODE)

        mode = get_comparison_mode([['header'], ['Tensor Path']])
        self.assertEqual(mode, CompareConst.TENSOR_COMPARISION_MODE)

    def test_convert_str_to_float(self):
        ret, value = convert_str_to_float('1.0')
        self.assertTrue(ret)
        self.assertEqual(value, float(1))

        ret, value = convert_str_to_float('1.0+')
        self.assertFalse(ret)
        self.assertEqual(value, float(0))

    def test_convert_str_to_int(self):
        ret, value = convert_str_to_int('1')
        self.assertTrue(ret)
        self.assertEqual(value, 1)

        ret, value = convert_str_to_int('1+')
        self.assertFalse(ret)
        self.assertEqual(value, 0)

        ret, value = convert_str_to_int('1+', -1)
        self.assertFalse(ret)
        self.assertEqual(value, -1)

    def test_compare_single_tensor(self):
        TENSOR_SPECIAL_HEADER_LENGTH = 6
        default_ret = [CompareConst.N_A] * TENSOR_SPECIAL_HEADER_LENGTH
        golden_tensor = torch.tensor([0.0])
        target_tensor = torch.tensor([0.0])

        with patch('msprobe.core.compare.atb_data_compare.get_error_flag_and_msg', return_value=(0, 0, True, '')):
            ret = compare_single_tensor(golden_tensor, target_tensor)
            self.assertEqual(ret, default_ret)

        with patch('msprobe.core.compare.atb_data_compare.get_error_flag_and_msg', return_value=(0, 0, False, '')), \
             patch('msprobe.core.compare.atb_data_compare.compare_ops_apply',
                   return_value=(['Cosine', CompareConst.NAN], False)):
            ret = compare_single_tensor(golden_tensor, target_tensor)
            self.assertEqual(ret, ['Cosine', CompareConst.N_A])

        golden_tensor = torch.tensor([0.0], dtype=torch.bfloat16)
        target_tensor = torch.tensor([0.0], dtype=torch.bfloat16)
        with patch('msprobe.core.compare.atb_data_compare.get_error_flag_and_msg', return_value=(0, 0, False, '')), \
             patch('msprobe.core.compare.atb_data_compare.compare_ops_apply',
                   return_value=(['Cosine', CompareConst.NAN], False)):
            ret = compare_single_tensor(golden_tensor, target_tensor)
            self.assertEqual(ret, ['Cosine', CompareConst.N_A])

    def test_cal_single_tensor_stats(self):
        defalut_stats = {}
        for stat in COMMON_HEADER_STATS:
            defalut_stats[stat] = CompareConst.N_A
        tensor = torch.tensor([0])

        ret = cal_single_tensor_stats(tensor, False)
        self.assertEqual(ret, defalut_stats)

        tensor = torch.empty(0)
        ret = cal_single_tensor_stats(tensor, True)
        self.assertEqual(ret, defalut_stats)

        tensor = torch.empty(1, dtype=torch.bool)
        ret = cal_single_tensor_stats(tensor, True)
        target_stats = defalut_stats.copy()
        target_stats[Const.MAX] = str(torch.any(tensor))
        target_stats[Const.MIN] = str(torch.all(tensor))
        self.assertEqual(ret, target_stats)

        tensor = torch.empty(1, dtype=torch.bool)
        ret = cal_single_tensor_stats(tensor, True)
        target_stats = defalut_stats.copy()
        target_stats[Const.MAX] = str(torch.any(tensor))
        target_stats[Const.MIN] = str(torch.all(tensor))
        self.assertEqual(ret, target_stats)

        real = torch.tensor([1, 2], dtype=torch.float32)
        imag = torch.tensor([3, 4], dtype=torch.float32)
        tensor = torch.complex(real, imag)
        ret = cal_single_tensor_stats(tensor, True)
        target_stats = defalut_stats.copy()
        target_stats[Const.MAX] = CompareConst.N_A
        target_stats[Const.MIN] = CompareConst.N_A
        target_stats[Const.MEAN] = str(tensor.mean().item())
        target_stats[Const.NORM] = str(tensor.norm().item())
        self.assertEqual(ret, target_stats)

        tensor = torch.tensor([1, 2])
        ret = cal_single_tensor_stats(tensor, True)
        tensor = tensor.float()
        target_stats = defalut_stats.copy()
        target_stats[Const.MAX] = str(tensor.max().item())
        target_stats[Const.MIN] = str(tensor.min().item())
        target_stats[Const.MEAN] = str(tensor.mean().item())
        target_stats[Const.NORM] = str(tensor.norm().item())
        self.assertEqual(ret, target_stats)

        tensor = torch.tensor([1, 2], dtype=torch.float64)
        ret = cal_single_tensor_stats(tensor, True)
        tensor = tensor.float()
        target_stats = defalut_stats.copy()
        target_stats[Const.MAX] = str(tensor.max().item())
        target_stats[Const.MIN] = str(tensor.min().item())
        target_stats[Const.MEAN] = str(tensor.mean().item())
        target_stats[Const.NORM] = str(tensor.norm().item())
        self.assertEqual(ret, target_stats)

        tensor = torch.tensor([1, 2], dtype=torch.float32)
        ret = cal_single_tensor_stats(tensor, True)
        target_stats = defalut_stats.copy()
        target_stats[Const.MAX] = str(tensor.max().item())
        target_stats[Const.MIN] = str(tensor.min().item())
        target_stats[Const.MEAN] = str(tensor.mean().item())
        target_stats[Const.NORM] = str(tensor.norm().item())
        self.assertEqual(ret, target_stats)

        tensor = torch.tensor([1, float('nan')], dtype=torch.float32)
        ret = cal_single_tensor_stats(tensor, True)
        self.assertEqual(ret, defalut_stats)

    def test_cal_comparison_metrics_by_tensor(self):
        target_stats_map = {
            'data0': {
                CompareConst.ATB_DATA_NAME: f'target {CompareConst.ATB_DATA_NAME}',
                CompareConst.DEVICE_AND_PID: f'target {CompareConst.DEVICE_AND_PID}',
                CompareConst.EXECUTION_COUNT: f'target {CompareConst.EXECUTION_COUNT}',
                CompareConst.DATA_TYPE: f'target {CompareConst.DATA_TYPE}',
                CompareConst.DATA_SHAPE: f'target {CompareConst.DATA_SHAPE}',
                CompareConst.TENSOR_PATH: f'target {CompareConst.TENSOR_PATH}'
            },
            'data1': {
                CompareConst.ATB_DATA_NAME: f'target {CompareConst.ATB_DATA_NAME}',
                CompareConst.DEVICE_AND_PID: f'target {CompareConst.DEVICE_AND_PID}',
                CompareConst.EXECUTION_COUNT: f'target {CompareConst.EXECUTION_COUNT}',
                CompareConst.DATA_TYPE: f'target {CompareConst.DATA_TYPE}',
                CompareConst.DATA_SHAPE: f'target {CompareConst.DATA_SHAPE}',
                CompareConst.TENSOR_PATH: f'target {CompareConst.TENSOR_PATH}'
            }
        }

        golden_stats_map = {
            'data0': {
                CompareConst.ATB_DATA_NAME: f'golden {CompareConst.ATB_DATA_NAME}',
                CompareConst.DEVICE_AND_PID: f'golden {CompareConst.DEVICE_AND_PID}',
                CompareConst.EXECUTION_COUNT: f'golden {CompareConst.EXECUTION_COUNT}',
                CompareConst.DATA_TYPE: f'golden {CompareConst.DATA_TYPE}',
                CompareConst.DATA_SHAPE: f'golden {CompareConst.DATA_SHAPE}',
                CompareConst.TENSOR_PATH: f'golden {CompareConst.TENSOR_PATH}'
            }
        }

        single_tensor_stats = {
            Const.MAX: Const.MAX,
            Const.MIN: Const.MIN,
            Const.MEAN: Const.MEAN,
            Const.NORM: Const.NORM
        }

        comparison_metrics = [
            'Cosine', 'Euc Distance', 'Max Absolute Err', 'Max Relative Err',
            'One Thousandth Err Ratio', 'Five Thousandth Err Ratio'
        ]

        expected_result_map = {}
        for col_name in COMMON_HEADER:
            expected_result_map[f'Target {col_name}'] = [target_stats_map["data0"][col_name]]
            expected_result_map[f'Golden {col_name}'] = [golden_stats_map["data0"][col_name]]
        for col_name in COMMON_HEADER_STATS:
            expected_result_map[f'Target {col_name}'] = [single_tensor_stats[col_name]]
            expected_result_map[f'Golden {col_name}'] = [single_tensor_stats[col_name]]
        for index, col_name in enumerate(TENSOR_SPECIAL_HEADER):
            expected_result_map[col_name] = [comparison_metrics[index]]

        with patch('msprobe.core.compare.atb_data_compare.TensorBinFile', autospec=True) as mock_bin_class, \
             patch('msprobe.core.compare.atb_data_compare.cal_single_tensor_stats',
                   return_value=single_tensor_stats), \
             patch('msprobe.core.compare.atb_data_compare.compare_single_tensor',
                   return_value=comparison_metrics) as mock_compare_single_tensor:
            mock_bin = mock_bin_class.return_value
            mock_bin.is_valid = True
            mock_bin.get_data.return_value = "tensor"
            comparison_result_map = dict()
            for col_name in COMMON_HEADER:
                comparison_result_map[f'Target {col_name}'] = []
                comparison_result_map[f'Golden {col_name}'] = []
            for col_name in COMMON_HEADER_STATS:
                comparison_result_map[f'Target {col_name}'] = []
                comparison_result_map[f'Golden {col_name}'] = []
            for index, col_name in enumerate(TENSOR_SPECIAL_HEADER):
                comparison_result_map[col_name] = []
            cal_comparison_metrics_by_tensor(golden_stats_map, target_stats_map, comparison_result_map)
            self.assertEqual(comparison_result_map, expected_result_map)

            mock_compare_single_tensor.return_value = comparison_metrics[:len(TENSOR_SPECIAL_HEADER) - 2]
            comparison_result_map = dict()
            for col_name in COMMON_HEADER:
                comparison_result_map[f'Target {col_name}'] = []
                comparison_result_map[f'Golden {col_name}'] = []
            for col_name in COMMON_HEADER_STATS:
                comparison_result_map[f'Target {col_name}'] = []
                comparison_result_map[f'Golden {col_name}'] = []
            for index, col_name in enumerate(TENSOR_SPECIAL_HEADER):
                comparison_result_map[col_name] = []
            expected_result_map[TENSOR_SPECIAL_HEADER[-1]] = [CompareConst.N_A]
            expected_result_map[TENSOR_SPECIAL_HEADER[-2]] = [CompareConst.N_A]
            cal_comparison_metrics_by_tensor(golden_stats_map, target_stats_map, comparison_result_map)
            self.assertEqual(comparison_result_map, expected_result_map)

            mock_bin.is_valid = False
            mock_bin.get_data.return_value = "tensor"
            comparison_result_map = dict()
            for col_name in COMMON_HEADER:
                comparison_result_map[f'Target {col_name}'] = []
                comparison_result_map[f'Golden {col_name}'] = []
            for col_name in COMMON_HEADER_STATS:
                comparison_result_map[f'Target {col_name}'] = []
                comparison_result_map[f'Golden {col_name}'] = []
            for index, col_name in enumerate(TENSOR_SPECIAL_HEADER):
                comparison_result_map[col_name] = []
            for index, col_name in enumerate(TENSOR_SPECIAL_HEADER):
                expected_result_map[col_name] = [CompareConst.N_A]
            cal_comparison_metrics_by_tensor(golden_stats_map, target_stats_map, comparison_result_map)
            self.assertEqual(comparison_result_map, expected_result_map)

    def test_cal_comparison_metrics_by_stats(self):
        target_stats_map = {
            'data0': {
                CompareConst.ATB_DATA_NAME: f'target {CompareConst.ATB_DATA_NAME}',
                CompareConst.DEVICE_AND_PID: f'target {CompareConst.DEVICE_AND_PID}',
                CompareConst.EXECUTION_COUNT: f'target {CompareConst.EXECUTION_COUNT}',
                CompareConst.DATA_TYPE: f'target {CompareConst.DATA_TYPE}',
                CompareConst.DATA_SHAPE: f'target {CompareConst.DATA_SHAPE}',
                CompareConst.TENSOR_PATH: f'target {CompareConst.TENSOR_PATH}',
                Const.MAX: '2',
                Const.MIN: '0',
                Const.MEAN: '0.5',
                Const.NORM: '0'
            },
            'data1': {
                CompareConst.ATB_DATA_NAME: f'target {CompareConst.ATB_DATA_NAME}',
                CompareConst.DEVICE_AND_PID: f'target {CompareConst.DEVICE_AND_PID}',
                CompareConst.EXECUTION_COUNT: f'target {CompareConst.EXECUTION_COUNT}',
                CompareConst.DATA_TYPE: f'target {CompareConst.DATA_TYPE}',
                CompareConst.DATA_SHAPE: f'target {CompareConst.DATA_SHAPE}',
                CompareConst.TENSOR_PATH: f'target {CompareConst.TENSOR_PATH}',
                Const.MAX: '2',
                Const.MIN: '0',
                Const.MEAN: '0.5',
                Const.NORM: '0'
            }
        }

        golden_stats_map = {
            'data0': {
                CompareConst.ATB_DATA_NAME: f'golden {CompareConst.ATB_DATA_NAME}',
                CompareConst.DEVICE_AND_PID: f'golden {CompareConst.DEVICE_AND_PID}',
                CompareConst.EXECUTION_COUNT: f'golden {CompareConst.EXECUTION_COUNT}',
                CompareConst.DATA_TYPE: f'golden {CompareConst.DATA_TYPE}',
                CompareConst.DATA_SHAPE: f'golden {CompareConst.DATA_SHAPE}',
                CompareConst.TENSOR_PATH: f'golden {CompareConst.TENSOR_PATH}',
                Const.MAX: '2',
                Const.MIN: '0',
                Const.MEAN: '0.5',
                Const.NORM: '0'
            }
        }

        expected_result_map = {}
        for col_name in COMMON_HEADER:
            expected_result_map[f'Target {col_name}'] = [target_stats_map["data0"][col_name]]
            expected_result_map[f'Golden {col_name}'] = [golden_stats_map["data0"][col_name]]
        for col_name in COMMON_HEADER_STATS:
            expected_result_map[f'Target {col_name}'] = [target_stats_map["data0"][col_name]]
            expected_result_map[f'Golden {col_name}'] = [golden_stats_map["data0"][col_name]]
        for col_name in STATS_SPECIAL_HEADER:
            expected_result_map[col_name] = ['0.0']

        comparison_result_map = dict()
        for col_name in COMMON_HEADER:
            comparison_result_map[f'Target {col_name}'] = []
            comparison_result_map[f'Golden {col_name}'] = []
        for col_name in COMMON_HEADER_STATS:
            comparison_result_map[f'Target {col_name}'] = []
            comparison_result_map[f'Golden {col_name}'] = []
        for col_name in STATS_SPECIAL_HEADER:
            comparison_result_map[col_name] = []
        cal_comparison_metrics_by_stats(golden_stats_map, target_stats_map, comparison_result_map)
        self.assertEqual(comparison_result_map, expected_result_map)

        comparison_result_map = dict()
        for col_name in COMMON_HEADER:
            comparison_result_map[f'Target {col_name}'] = []
            comparison_result_map[f'Golden {col_name}'] = []
        for col_name in COMMON_HEADER_STATS:
            comparison_result_map[f'Target {col_name}'] = []
            comparison_result_map[f'Golden {col_name}'] = []
        for col_name in STATS_SPECIAL_HEADER:
            comparison_result_map[col_name] = []
        golden_stats_map['data0'][Const.MAX] = '1'
        expected_result_map['Golden Max'] = ['1']
        expected_result_map['Max Diff'] = ['1.0']
        expected_result_map['Relative Err of Max(%)'] = ['100.0']
        cal_comparison_metrics_by_stats(golden_stats_map, target_stats_map, comparison_result_map)
        self.assertEqual(comparison_result_map, expected_result_map)

        comparison_result_map = dict()
        for col_name in COMMON_HEADER:
            comparison_result_map[f'Target {col_name}'] = []
            comparison_result_map[f'Golden {col_name}'] = []
        for col_name in COMMON_HEADER_STATS:
            comparison_result_map[f'Target {col_name}'] = []
            comparison_result_map[f'Golden {col_name}'] = []
        for col_name in STATS_SPECIAL_HEADER:
            comparison_result_map[col_name] = []
        golden_stats_map['data0'][Const.MAX] = '0'
        expected_result_map['Golden Max'] = ['0']
        expected_result_map['Max Diff'] = ['2.0']
        expected_result_map['Relative Err of Max(%)'] = [CompareConst.N_A]
        cal_comparison_metrics_by_stats(golden_stats_map, target_stats_map, comparison_result_map)
        self.assertEqual(comparison_result_map, expected_result_map)

        comparison_result_map = dict()
        for col_name in COMMON_HEADER:
            comparison_result_map[f'Target {col_name}'] = []
            comparison_result_map[f'Golden {col_name}'] = []
        for col_name in COMMON_HEADER_STATS:
            comparison_result_map[f'Target {col_name}'] = []
            comparison_result_map[f'Golden {col_name}'] = []
        for col_name in STATS_SPECIAL_HEADER:
            comparison_result_map[col_name] = []
        golden_stats_map['data0'][Const.MAX] = 'a'
        expected_result_map['Golden Max'] = ['a']
        expected_result_map['Max Diff'] = [CompareConst.N_A]
        expected_result_map['Relative Err of Max(%)'] = [CompareConst.N_A]
        cal_comparison_metrics_by_stats(golden_stats_map, target_stats_map, comparison_result_map)
        self.assertEqual(comparison_result_map, expected_result_map)

    def test_get_stats_map(self):
        root_path = '/root_path'
        csv_content = [['header']]

        expected_stats_map = dict()
        ret = get_stats_map(csv_content, root_path)
        self.assertEqual(ret, expected_stats_map)

        csv_content = [
            ['header'],
            [
                'Device and PID1', 'Execution Count1', 'name1/name1', 'Op Type1', '1_1', 'Input/Output1', 'Index1',
                'Dtype1', 'Format1', 'Shape1', 'Max1', 'Min1', 'Mean1', 'Norm1'
            ],
            [
                'Device and PID2', 'Execution Count2', 'name2/name2', 'Op Type2', '2_2', 'Input/Output2', 'Index2',
                'Dtype2', 'Format2', 'Shape2', 'Max2', 'Min2', 'Mean2', 'Norm2', '/root/atb_dump_data/Tensor Path'
            ],
            [
                'Device and PID3', 'Execution Count3', 'name3/name3', 'Op Type3', '3', 'Input/Output3', 'Index3',
                'Dtype3', 'Format3', 'Shape3', 'Max3', 'Min3', 'Mean3', 'Norm3', 'N/A'
            ],
            [
                'Device and PID4', 'Execution Count4', 'name4/name4', 'Op Type4', '4_4', 'Input/Output4', 'Index4',
                'Dtype4', 'Format4', 'Shape4', 'Max4', 'Min4', 'Mean4', 'Norm4', 'N/A'
            ],
            [
                'Device and PID5', 'Execution Count5', 'name5/name5', 'Op Type5', '5_5', 'Input/Output5', 'Index5',
                'Dtype5', 'Format5', 'Shape5', 'Max5', 'Min5', 'Mean5', 'Norm5',
                '/root/atb_dump_data/data/0_2525/5/5_Prefill_layer/before/intensor0.bin'
            ]
        ]

        single_stat_map = {
            CompareConst.ATB_DATA_NAME: '4_name4/4_name4/Input/Output4.Index4',
            CompareConst.DEVICE_AND_PID: 'Device and PID4',
            CompareConst.EXECUTION_COUNT: 'Execution Count4',
            CompareConst.DATA_TYPE: 'Dtype4',
            CompareConst.DATA_SHAPE: 'Shape4',
            Const.MAX: 'Max4',
            Const.MIN: 'Min4',
            Const.MEAN: 'Mean4',
            Const.NORM: 'Norm4',
            CompareConst.TENSOR_PATH: 'N/A'
        }
        expected_stats_map['4_name4/4_name4/Input/Output4.Index4'] = single_stat_map
        single_stat_map = {
            CompareConst.ATB_DATA_NAME: '5_name5/5_name5/Input/Output5.Index5',
            CompareConst.DEVICE_AND_PID: 'Device and PID5',
            CompareConst.EXECUTION_COUNT: 'Execution Count5',
            CompareConst.DATA_TYPE: 'Dtype5',
            CompareConst.DATA_SHAPE: 'Shape5',
            Const.MAX: 'Max5',
            Const.MIN: 'Min5',
            Const.MEAN: 'Mean5',
            Const.NORM: 'Norm5',
            CompareConst.TENSOR_PATH: '/root_path/5_Prefill_layer/before/intensor0.bin'
        }
        expected_stats_map['5_name5/5_name5/Input/Output5.Index5'] = single_stat_map
        ret = get_stats_map(csv_content, root_path)
        self.assertEqual(ret, expected_stats_map)

    def test_compare_atb_mode(self):
        args = MagicMock()
        args.golden_path = os.path.realpath(__file__)
        args.target_path = os.path.dirname(os.path.realpath(__file__))
        args.output_path = os.path.dirname(os.path.realpath(__file__))

        with patch.object(logger, 'error') as mock_logger_error:
            with self.assertRaises(FileCheckException) as context:
                compare_atb_mode(args)
                self.assertEqual(str(context.exception),
                                 FileCheckException.err_strs.get(FileCheckException.ILLEGAL_PATH_ERROR))
                mock_logger_error.assert_called_with(
                    'The golden_path and target_path should be both files or directories')

        args.target_path = os.path.realpath(__file__)
        with patch.object(logger, 'warning') as mock_logger_warning:
            compare_atb_mode(args)
            mock_logger_warning.assert_called_with('Comparison between ATB data files is not yet supported')

        args.target_path = os.path.dirname(os.path.realpath(__file__))
        with patch('msprobe.core.compare.atb_data_compare.check_and_get_real_path',
                   return_value=args.target_path) as mock_get_real_path, \
             patch('msprobe.core.compare.atb_data_compare.read_csv',
                   return_value=[]) as mock_read_csv, \
             patch.object(logger, 'error') as mock_logger_error, \
             patch('msprobe.core.compare.atb_data_compare.create_directory'), \
             patch('msprobe.core.compare.atb_data_compare.save_excel') as mock_save_excel, \
             patch.object(logger, 'info') as mock_logger_info, \
             patch.object(time, 'strftime', return_value='YYYYmmddHHMMSS'), \
             patch('msprobe.core.compare.atb_data_compare.cal_comparison_metrics_by_tensor') as mock_cal_by_tensor:
            compare_atb_mode(args)
            self.assertEqual(mock_get_real_path.call_count, 5)
            self.assertEqual(mock_read_csv.call_count, 2)
            mock_logger_error.assert_called_with('Unvalid ATB dump data')

            mock_read_csv.return_value = [['header'], ['tensor']]
            from msprobe.core.compare import atb_data_compare
            atb_data_compare.torch_available = False
            mock_logger_error.reset_mock()
            compare_atb_mode(args)
            mock_logger_error.assert_called_with(
                'Unable to compare ATB Tensor without torch. Please install with \"pip install torch\"')

            atb_data_compare.torch_available = True
            mock_read_csv.return_value = [
                ['header'],
                [
                    '0_252571', '3', 'Prefill_layer', 'Prefill_layer', '2', 'input', '0', 'bf16', 'nd',
                    '3584', '0.582031', '-0.00132', '0.28355', '17.01721', 'N/A'
                ],
                [
                    '0_252571', '3', 'Prefill_layer', 'Prefill_layer', '2', 'input', '0', 'bf16', 'nd',
                    '3584', '0.582031', '-0.00132', '0.28355', '17.01721', 'N/A'
                ]
            ]
            compare_atb_mode(args)
            expected_comparison_result_map = {
                f'Target {CompareConst.ATB_DATA_NAME}': ['2_Prefill_layer/input.0'],
                f'Golden {CompareConst.ATB_DATA_NAME}': ['2_Prefill_layer/input.0'],
                f'Target {CompareConst.DEVICE_AND_PID}': ['0_252571'],
                f'Golden {CompareConst.DEVICE_AND_PID}': ['0_252571'],
                f'Target {CompareConst.EXECUTION_COUNT}': ['3'],
                f'Golden {CompareConst.EXECUTION_COUNT}': ['3'],
                f'Target {CompareConst.DATA_TYPE}': ['bf16'],
                f'Golden {CompareConst.DATA_TYPE}': ['bf16'],
                f'Target {CompareConst.DATA_SHAPE}': ['3584'],
                f'Golden {CompareConst.DATA_SHAPE}': ['3584']
            }
            for col_name in STATS_SPECIAL_HEADER:
                expected_comparison_result_map[col_name] = ['0.0']

            expected_comparison_result_map['Target Max'] = ['0.582031']
            expected_comparison_result_map['Golden Max'] = ['0.582031']
            expected_comparison_result_map['Target Min'] = ['-0.00132']
            expected_comparison_result_map['Golden Min'] = ['-0.00132']
            expected_comparison_result_map['Target Mean'] = ['0.28355']
            expected_comparison_result_map['Golden Mean'] = ['0.28355']
            expected_comparison_result_map['Target Norm'] = ['17.01721']
            expected_comparison_result_map['Golden Norm'] = ['17.01721']
            expected_xlsx_file_path = os.path.join(args.target_path, 'atb_stats_compare_result_YYYYmmddHHMMSS.xlsx')
            self.assertEqual(mock_save_excel.call_args[0][0], expected_xlsx_file_path)
            self.assertTrue(pd.DataFrame(expected_comparison_result_map).equals(mock_save_excel.call_args[0][1]))
            mock_logger_info.assert_called_with(
                f'Complete ATB dump data comparison. Comparison result has been saved at {expected_xlsx_file_path}')

            mock_read_csv.return_value = [
                ['header'],
                [
                    '0_252571', '3', 'Prefill_layer', 'Prefill_layer', '2', 'input', '0', 'bf16', 'nd',
                    '3584', '0.582031', '-0.00132', '0.28355', '17.01721',
                    '/root/atb_dump_data/data/0_252571/3/2_Prefill_layer/before/intensor0.bin'
                ],
                [
                    '0_252571', '3', 'Prefill_layer', 'Prefill_layer', '2', 'input', '0', 'bf16', 'nd',
                    '3584', '0.582031', '-0.00132', '0.28355', '17.01721',
                    '/root/atb_dump_data/data/0_252571/3/2_Prefill_layer/before/intensor0.bin'
                ]
            ]
            stats_map = {
                '2_Prefill_layer/input.0': {
                    CompareConst.ATB_DATA_NAME: '2_Prefill_layer/input.0',
                    CompareConst.DEVICE_AND_PID: '0_252571',
                    CompareConst.EXECUTION_COUNT: '3',
                    CompareConst.DATA_TYPE: 'bf16',
                    CompareConst.DATA_SHAPE: '3584',
                    Const.MAX: '0.582031',
                    Const.MIN: '-0.00132',
                    Const.MEAN: '0.28355',
                    Const.NORM: '17.01721',
                    CompareConst.TENSOR_PATH: f'{args.target_path}/2_Prefill_layer/before/intensor0.bin'
                }
            }
            expected_comparison_result_map.clear()
            for col_name in COMMON_HEADER:
                expected_comparison_result_map[f'Target {col_name}'] = []
                expected_comparison_result_map[f'Golden {col_name}'] = []
            for col_name in TENSOR_SPECIAL_HEADER:
                expected_comparison_result_map[col_name] = []
            for col_name in COMMON_HEADER_STATS:
                expected_comparison_result_map[f'Target {col_name}'] = []
                expected_comparison_result_map[f'Golden {col_name}'] = []
            mock_save_excel.reset_mock()
            compare_atb_mode(args)
            mock_cal_by_tensor.assert_called_once_with(stats_map, stats_map, expected_comparison_result_map)
            mock_save_excel.assert_not_called()
