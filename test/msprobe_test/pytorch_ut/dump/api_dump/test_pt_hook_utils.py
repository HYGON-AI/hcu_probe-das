# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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
from unittest.mock import MagicMock, patch

from msprobe.pytorch.dump.api_dump.utils import get_ops, dynamic_import_op


class MockPackage:
    __name__ = "mock_package"
    __file__ = "/fake_path/__init__.py"


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.yaml_content = {
            'functional': ['func1', 'func2'],
            'tensor': ['tensor_op1'],
            'torch': ['torch_op1', 'torch_op2'],
            'torch_npu': ['npu_op1']
        }

        self.mock_listdir = patch('os.listdir').start()
        self.mock_check_link = patch('msprobe.pytorch.dump.api_dump.utils.check_link').start()

    def tearDown(self):
        patch.stopall()

    def test_get_ops(self):
        with patch('msprobe.pytorch.dump.api_dump.utils.load_yaml') as mock_load:
            mock_load.return_value = self.yaml_content
            result = get_ops()
            self.assertEqual(
                result,
                {
                    'func1',
                    'func2',
                    'tensor_op1',
                    'torch_op1',
                    'torch_op2',
                    'npu_op1'
                }
            )

    @patch('msprobe.pytorch.dump.api_dump.utils.inspect')
    def test_dynamic_import_op_success(self, mock_inspect):
        mock_func = lambda x: x
        mock_inspect.getmembers = MagicMock()
        mock_inspect.getmembers.return_value = [['test_func', mock_func]]

        self.mock_listdir.return_value = ['valid.py', 'invalid.py']
        mock_module = MagicMock()

        with patch('importlib.import_module', return_value=mock_module) as mock_import:
            ops = dynamic_import_op(MockPackage(), white_list=['valid.py'])
            self.assertEqual(ops, {'valid.test_func': mock_func})
            mock_import.assert_called_once_with('mock_package.valid')

    def test_dynamic_import_op_failure(self):
        self.mock_listdir.return_value = ['fail.py']
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("Fake error")
            with patch('msprobe.pytorch.dump.api_dump.utils.logger.warning') as mock_logger:
                ops = dynamic_import_op(MockPackage(), white_list=['fail.py'])
                self.assertEqual(ops, {})
                mock_logger.assert_called_once_with("import mock_package.fail failed!")