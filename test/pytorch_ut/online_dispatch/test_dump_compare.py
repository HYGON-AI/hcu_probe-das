# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

import unittest
import json
import os
import threading
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from unittest.mock import MagicMock

import torch
import numpy as np

from msprobe.core.common.const import CompareConst
from msprobe.pytorch.online_dispatch.dump_compare import support_basic_type, dump_data, save_temp_summary, \
    dispatch_workflow, get_torch_func, dispatch_multiprocess, error_call, DispatchRunParam, DisPatchDataInfo


class TestDumpCompare(unittest.TestCase):
    def setUp(self):
        self.summary_path = "summary.json"
        Path(self.summary_path).touch()
        self.data = {CompareConst.NPU_NAME: 1,
                    CompareConst.BENCH_NAME: 1,
                    CompareConst.NPU_DTYPE: 1,
                    CompareConst.BENCH_DTYPE: 11,
                    CompareConst.NPU_SHAPE: 1,
                    CompareConst.BENCH_SHAPE: 1,
                    CompareConst.NPU_MAX: 1,
                    CompareConst.NPU_MIN: 1,
                    CompareConst.NPU_MEAN: 1,
                    CompareConst.BENCH_MAX: 1,
                    CompareConst.BENCH_MIN: 1,
                    CompareConst.BENCH_MEAN: 1,
                    CompareConst.COSINE: 1,
                    CompareConst.MAX_ABS_ERR: 1,
                    CompareConst.MAX_RELATIVE_ERR: 1,
                    CompareConst.ACCURACY: 1,
                    CompareConst.ERROR_MESSAGE: 1}
        self.data_gt = {CompareConst.NPU_NAME: 1,
                    CompareConst.BENCH_NAME: 1,
                    CompareConst.NPU_DTYPE: 1,
                    CompareConst.BENCH_DTYPE: 11,
                    CompareConst.NPU_SHAPE: 1,
                    CompareConst.BENCH_SHAPE: 1,
                    CompareConst.NPU_MAX: 1,
                    CompareConst.NPU_MIN: 1,
                    CompareConst.NPU_MEAN: 1,
                    CompareConst.BENCH_MAX: 1,
                    CompareConst.BENCH_MIN: 1,
                    CompareConst.BENCH_MEAN: 1,
                    CompareConst.COSINE: 1,
                    CompareConst.MAX_ABS_ERR: 1,
                    CompareConst.MAX_RELATIVE_ERR: 1,
                    CompareConst.ACCURACY: 1,
                    CompareConst.STACK: 2,
                    CompareConst.ERROR_MESSAGE: 1}

        self.dump_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'dump_data')
        os.makedirs(self.dump_path, mode=0o750, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.summary_path):
            os.remove(self.summary_path)
        if os.path.exists(self.dump_path):
            shutil.rmtree(self.dump_path)

    def test_dispatch_run_param(self):
        debug_flag = True
        device_id = 0
        root_npu_path = '/path/to/npu'
        root_cpu_path = '/path/to/cpu'
        process_num = 4
        comparator = 'comparator_function'

        dispatch_run_param = DispatchRunParam(debug_flag, device_id, root_npu_path, root_cpu_path, process_num,
                                              comparator)

        # 验证静态参数是否正确初始化
        self.assertEqual(dispatch_run_param.debug_flag, debug_flag)
        self.assertEqual(dispatch_run_param.device_id, device_id)
        self.assertEqual(dispatch_run_param.root_npu_path, root_npu_path)
        self.assertEqual(dispatch_run_param.root_cpu_path, root_cpu_path)
        self.assertEqual(dispatch_run_param.process_num, process_num)
        self.assertEqual(dispatch_run_param.comparator, comparator)

        # 验证动态参数是否有默认值
        self.assertFalse(dispatch_run_param.process_flag)  # 默认值应为 False
        self.assertIsNone(dispatch_run_param.func_name)  # 默认值应为 None
        self.assertIsNone(dispatch_run_param.func_namespace)  # 默认值应为 None
        self.assertIsNone(dispatch_run_param.aten_api)  # 默认值应为 None
        self.assertIsNone(dispatch_run_param.aten_api_overload_name)  # 默认值应为 None
        self.assertIsNone(dispatch_run_param.single_api_index)  # 默认值应为 None
        self.assertIsNone(dispatch_run_param.api_index)  # 默认值应为 None
        self.assertIsNone(dispatch_run_param.dump_flag)  # 默认值应为 None
        self.assertIsNone(dispatch_run_param.auto_dump_flag)  # 默认值应为 None

    def test_dispatch_datainfo(self):
        mock_func = MagicMock()
        mock_lock = MagicMock()
        cpu_args = (1, 2, 3)
        cpu_kwargs = {'arg1': 1, 'arg2': 2}
        all_summary = ['summary1', 'summary2']
        npu_out_cpu = [10, 20]
        cpu_out = [30, 40]

        dispatch_data = DisPatchDataInfo(cpu_args, cpu_kwargs, all_summary, mock_func, npu_out_cpu, cpu_out, mock_lock)

        self.assertEqual(dispatch_data.cpu_args, cpu_args)
        self.assertEqual(dispatch_data.cpu_kwargs, cpu_kwargs)
        self.assertEqual(dispatch_data.all_summary, all_summary)
        self.assertEqual(dispatch_data.func, mock_func)
        self.assertEqual(dispatch_data.npu_out_cpu, npu_out_cpu)
        self.assertEqual(dispatch_data.cpu_out, cpu_out)
        self.assertEqual(dispatch_data.lock, mock_lock)

    def test_support_basic_type_should_return_true_when_is_instance(self):
        self.assertTrue(support_basic_type(2.3))

    def test_support_basic_type_should_return_false_when_isnot_instance(self):
        self.assertFalse(support_basic_type("abcde"))

    def test_data_dump_tensor(self):
        # 测试正常的 tensor 是否保存
        data = torch.tensor([1, 2, 3])
        prefix = 'test_tensor'
        dump_data(data, prefix, self.dump_path)
        expected_path = os.path.join(self.dump_path, f'{prefix}.npy')
        self.assertTrue(os.path.exists(expected_path))
        # 检查保存的内容
        saved_data = np.load(expected_path)
        self.assertTrue(np.array_equal(saved_data, data.numpy()))

    def test_data_dump_basic(self):
        # 测试基本类型数据的保存
        data = 42
        prefix = 'test_basic'
        dump_data(data, prefix, self.dump_path)
        expected_path = os.path.join(self.dump_path, f'{prefix}.npy')
        self.assertTrue(os.path.exists(expected_path))
        # 检查保存的内容是否正确
        saved_data = np.load(expected_path)
        self.assertEqual(saved_data, data)

    def test_data_dump_recursive(self):
        # 测试嵌套 list 的递归调用
        data = [42, [3.14, 2.71]]
        prefix = 'test_recursive'
        dump_data(data, prefix, self.dump_path)
        # 检查保存的文件
        self.assertTrue(os.path.exists(os.path.join(self.dump_path, 'test_recursive.0.npy')))
        self.assertTrue(os.path.exists(os.path.join(self.dump_path, 'test_recursive.1.0.npy')))
        self.assertTrue(os.path.exists(os.path.join(self.dump_path, 'test_recursive.1.1.npy')))

    def test_data_dump_meta(self):
        # 测试 meta tensor 是否跳过
        data = torch.empty(3, 3, device='meta')
        prefix = 'test_tensor_meta'
        dump_data(data, prefix, self.dump_path)
        expected_path = os.path.join(self.dump_path, f'{prefix}.npy')
        # Meta tensor 不应该生成文件
        self.assertFalse(os.path.exists(expected_path))

    def test_save_temp_summary(self):
        api_index = '1'
        single_api_summary = "conv2d"
        path = ''
        data = []
        lock = threading.Lock()

        save_temp_summary(api_index=api_index, single_api_summary=single_api_summary, path=path, lock=lock)

        with open(self.summary_path, 'r') as f:
            content = f.readlines()
            for line in content:
                data.append(json.loads(line))
        self.assertEqual([['1', 'conv2d']], data)

    @patch('msprobe.pytorch.online_dispatch.dump_compare.dump_data')
    @patch('msprobe.pytorch.online_dispatch.dump_compare.save_temp_summary')
    def test_dispatch_workflow_should_dump_when_flag_is_true(self, mock_save_temp_summary, mock_dump_data):
        mock_run_param = Mock()
        mock_run_param.aten_api = "aten_api"
        mock_run_param.single_api_index = "single_api_index"
        mock_run_param.root_npu_path = ""
        mock_data_info = Mock()
        mock_data_info.cpu_args = None
        mock_data_info.cpu_kwargs = []

        mock_run_param.dump_flag = True
        mock_run_param.process_num = 0  
        mock_run_param.api_index = 1
        mock_data_info.all_summary = [1]

        dispatch_workflow(mock_run_param, mock_data_info)
        mock_dump_data.assert_called()
        mock_save_temp_summary.assert_not_called()

    @patch('msprobe.pytorch.online_dispatch.dump_compare.dump_data')
    @patch('msprobe.pytorch.online_dispatch.dump_compare.save_temp_summary')
    def test_dispatch_workflow_should_not_dump_when_flag_is_false(self, mock_save_temp_summary,mock_dump_data):
        mock_run_param = Mock()
        mock_run_param.aten_api = "aten_api"
        mock_run_param.single_api_index = "single_api_index"
        mock_run_param.root_npu_path = ""
        mock_data_info = Mock()
        mock_data_info.cpu_args = None
        mock_data_info.cpu_kwargs = []

        mock_run_param.dump_flag = False
        mock_run_param.auto_dump_flag = False
        mock_run_param.process_num = 1
        mock_run_param.api_index = 1
        mock_data_info.all_summary = [1]

        dispatch_workflow(mock_run_param, mock_data_info)
        mock_dump_data.assert_not_called()
        mock_save_temp_summary.assert_called()   

    def test_get_torch_func_should_return_none_when_outside_input(self):
        mock_run_param = Mock()
        mock_run_param.func_namespace = "new_attr1"
        mock_run_param.aten_api = "new_attr2"
        mock_run_param.aten_api_overload_name = "new_attr3"
        self.assertIsNone(get_torch_func(mock_run_param))

    @patch('msprobe.core.common.log.BaseLogger.error')
    def test_dispatch_multiprocess_should_logger_error_when_wrong_api_input(self, mock_error):
        mock_run_param = Mock()
        mock_run_param.func_namespace = "new_attr1"
        mock_run_param.aten_api = "new_attr2"
        mock_run_param.aten_api_overload_name = "new_attr3"
        mock_dispatch_data_info = Mock()
        dispatch_multiprocess(mock_run_param, mock_dispatch_data_info)
        mock_error.assert_called_once_with(f'can not find suitable call api:{mock_run_param.aten_api}')
        
    @patch('msprobe.pytorch.online_dispatch.dump_compare.dispatch_workflow')
    def test_dispatch_multiprocess_should_workflow_when_right_api_input(self, mock_workflow):
        mock_run_param = Mock()
        mock_run_param.func_namespace = "aten"
        mock_run_param.aten_api = "add"
        mock_run_param.aten_api_overload_name = "Scalar"
        mock_dispatch_data_info = Mock()
        mock_workflow.return_value = 1
        dispatch_multiprocess(mock_run_param, mock_dispatch_data_info)
        mock_workflow.assert_called_once_with(mock_run_param, mock_dispatch_data_info)

    @patch('msprobe.core.common.log.BaseLogger.error')
    def test_error_call(self, mock_error):
        error_call("messages")
        mock_error.assert_called_once_with("multiprocess messages")
