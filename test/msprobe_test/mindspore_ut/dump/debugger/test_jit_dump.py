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
