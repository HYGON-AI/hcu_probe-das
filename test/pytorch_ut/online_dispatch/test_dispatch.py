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
from unittest.mock import Mock, patch
import os
import shutil

from msprobe.pytorch.online_dispatch.dispatch import PtdbgDispatch
from msprobe.pytorch.online_dispatch.utils import DispatchException


class RunParam:
    def __init__(self, aten_api, aten_api_overload_name):
        self.aten_api = aten_api
        self.aten_api_overload_name = aten_api_overload_name
        self.func_namespace = None


class TestPtdbgDispatch(unittest.TestCase):
    def setUp(self):
        self.dump_path = './dump_path'
        if not os.path.exists(self.dump_path):
            os.mkdir(self.dump_path)
        with patch('msprobe.pytorch.online_dispatch.dispatch.is_npu', new=True), \
            patch('msprobe.pytorch.online_dispatch.dispatch.torch_npu') as mock_torch_npu:
            mock_torch_npu._C = Mock()
            mock_torch_npu._C._npu_getDevice.return_value = 1
            self.ptdbg_dispatch = PtdbgDispatch(dump_mode='list', dump_path=self.dump_path, 
                                               api_list=['relu', 'aefa', 'rsqrt'], debug=True)
        self.yaml_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torch_ops_config.yaml')

    def tearDown(self):
        if os.path.exists(self.dump_path):
            shutil.rmtree(self.dump_path)

    @patch('msprobe.pytorch.online_dispatch.dispatch.logger.info')
    def test_exit(self, mock_info):
        with self.ptdbg_dispatch:
            pass
        
        mock_info.assert_called()
        args = mock_info.call_args[0]
        self.assertTrue(args[0].startswith('Dispatch exit'))

    @patch('torch.ops.aten')
    def test_check_fun_success(self, mock_aten):
        run_param = RunParam('my_api', 'my_overload')
        mock_func = Mock()
        mock_aten.my_api = Mock()
        mock_aten.my_api.my_overload = mock_func
        res = self.ptdbg_dispatch.check_fun(mock_func, run_param)

        self.assertTrue(res)
        self.assertEqual(run_param.func_namespace, 'aten')

    @patch('torch.ops.aten')
    def test_check_fun_failure(self, mock_aten):
        run_param = RunParam('invalid_api', 'invalid_overload')

        res = self.ptdbg_dispatch.check_fun(None, run_param)

        self.assertFalse(res)
        self.assertIsNone(run_param.func_namespace)

    def test_get_dir_name(self):
        res = self.ptdbg_dispatch.get_dir_name('my_tag')

        self.assertIn('msprobe_my_tag_rank', res)

    def test_get_ops(self):
        self.ptdbg_dispatch.get_ops(self.yaml_file_path)

        self.assertEqual(self.ptdbg_dispatch.aten_ops_blacklist, ['rand'])
        self.assertEqual(self.ptdbg_dispatch.npu_adjust_autograd, ['to'])

    def test_filter_dump_api(self):
        self.ptdbg_dispatch.filter_dump_api()

        self.assertEqual(self.ptdbg_dispatch.dump_api_list, ['relu', 'rsqrt'])

    def test_get_dump_flag(self):
        dump_flag, auto_dump_flag = self.ptdbg_dispatch.get_dump_flag('rsqrt')

        self.assertTrue(dump_flag)
        self.assertFalse(auto_dump_flag)

    def test_check_param_dump_mode(self):
        with self.assertRaises(DispatchException):
            self.ptdbg_dispatch.dump_mode = 'awfef'
            self.ptdbg_dispatch.check_param()

    def test_check_param_dump_api_list(self):
        with self.assertRaises(DispatchException):
            self.ptdbg_dispatch.dump_api_list = 'awfef'
            self.ptdbg_dispatch.check_param()

    def test_check_param_debug_flag(self):
        with self.assertRaises(DispatchException):
            self.ptdbg_dispatch.debug_flag = 'awfef'
            self.ptdbg_dispatch.check_param()

    def test_check_param_process_num(self):
        with self.assertRaises(DispatchException):
            self.ptdbg_dispatch.process_num = 'awfef'
            self.ptdbg_dispatch.check_param()

    @patch('torch._C._dispatch_tls_set_dispatch_key_excluded')
    def test_enable_autograd(self, mock__dispatch_tls_set_dispatch_key_excluded):
        self.ptdbg_dispatch.npu_adjust_autograd.append('to')
        self.ptdbg_dispatch.enable_autograd('to')

        mock__dispatch_tls_set_dispatch_key_excluded.assert_called_once()
