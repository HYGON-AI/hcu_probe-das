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
import inspect
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

from msprobe.pytorch.online_dispatch.utils import get_callstack, data_to_cpu, DispatchException, get_sys_info

cpu_device = torch._C.device("cpu")


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.stack = inspect.stack()
        self.data_path = ""
        self.file_name = "data"
        self.dispatch_exception = DispatchException(err_code=1, err_msg="messages")
        Path(os.path.join(self.data_path, f'{self.file_name}.npy')).touch()

    def tearDown(self):
        if os.path.exists(os.path.join(self.data_path, f'{self.file_name}.npy')):
            os.remove(os.path.join(self.data_path, f'{self.file_name}.npy'))

    @patch('inspect.stack')
    def test_get_callstack(self, mock_stack):
        mock_stack.return_value = [
            (None, '/path/to/file1.py', 10, 'function_a', ['line_of_code_a'], None),
            (None, '/path/to/file2.py', 20, 'function_b', ['line_of_code_b'], None),
            (None, '/path/to/file3.py', 30, 'function_c', ['line_of_code_c'], None),
        ]

        result = get_callstack()

        expected_callstack = [
            ['/path/to/file3.py', '30', 'function_c', 'line_of_code_c']
        ]

        self.assertEqual(result, expected_callstack)

    def test_data_to_cpu_tensor(self):
        data = torch.tensor([1, 2], device=cpu_device, dtype=torch.float16)
        deep = 1
        data_cpu = []
        self.assertEqual(data_to_cpu(data, deep, data_cpu).all(), data.clone().detach().float().all())

    def test_data_to_cpu_list(self):
        data = [1, 2]
        deep = 0
        data_cpu = []
        self.assertEqual(data_to_cpu(data, deep, data_cpu), data)

    def test_data_to_cpu_tuple(self):
        data = (1, 2)
        deep = 0
        data_cpu = []
        self.assertEqual(data_to_cpu(data, deep, data_cpu), data)

    def test_data_to_cpu_dict(self):
        data = {1: 2}
        deep = 0
        data_cpu = []
        self.assertEqual(data_to_cpu(data, deep, data_cpu), data)

    def test_data_to_cpu_c_device(self):
        data = torch.device("cpu")
        deep = 0
        data_cpu = []
        result = data_to_cpu(data, deep, data_cpu)
        self.assertEqual(result, cpu_device)

    def test_str(self):
        self.assertEqual(self.dispatch_exception.__str__(), "messages")

    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_get_sys_info(self, mock_cpu_percent, mock_virtual_memory):
        mock_virtual_memory.return_value = MagicMock(
            total=8 * 1024 * 1024 * 1024,
            available=4 * 1024 * 1024 * 1024,
            used=3 * 1024 * 1024 * 1024
        )
        mock_cpu_percent.return_value = 55.0

        result = get_sys_info()

        expected_output = 'Total: 8192.00MB Free: 4096.00 MB Used: 3072.00 MB CPU: 55.0%'
        self.assertIn(expected_output, result)
