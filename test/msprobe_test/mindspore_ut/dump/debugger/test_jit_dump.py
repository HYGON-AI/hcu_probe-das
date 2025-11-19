# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import os
import unittest
from unittest.mock import patch, MagicMock

import mindspore.common.dtype as mstype
import numpy as np
from mindspore.common.tensor import Tensor

from msprobe.mindspore.dump.dump_processor.jit_dump import JitDump, dump_jit


class TestJitDump(unittest.TestCase):
    @patch('os.getpid', return_value=12345)
    def test_dump_jit(self, mock_getpid):
        in_feat = Tensor(np.array([1, 2, 3]), mstype.float32)
        out_feat = Tensor(np.array([4, 5, 6]), mstype.float32)

        # Mock need_dump to return True
        with patch.object(JitDump, 'need_dump', return_value=True):
            # Add a mock data_collector to the JitDump class
            JitDump.data_collector = MagicMock()

            # Call the function to be tested
            dump_jit('sample_name', in_feat, out_feat, True)

            # Verify the expected calls
            self.assertTrue(JitDump.data_collector.update_api_or_module_name.called)
            self.assertTrue(JitDump.data_collector.forward_data_collect.called)
            JitDump.data_collector.forward_data_collect.assert_called_once()

    @patch('os.listdir', return_value=['tensor1', 'tensor2', 'tensor3', 'tensor4', 'tensor5'])
    def test_dump_tensor_data_files_count(self, mock_listdir):
        dir_path = "/absolute_path/step0/rank/dump_tensor_data/"
        expected_file_count = 5
        actual_file_count = len(os.listdir(dir_path))
        self.assertEqual(actual_file_count, expected_file_count)
