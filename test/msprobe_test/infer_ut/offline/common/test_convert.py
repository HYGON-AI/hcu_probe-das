# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import argparse
import unittest
from unittest.mock import patch
import os
import numpy as np
from msprobe.infer.offline.compare.msquickcmp.common.convert import convert_npy_to_bin, convert_bin_file_to_npy


class TestConvertNpyToBin(unittest.TestCase):
    def setUp(self):
        self.npy_path = 'convert_test.npy'
        self.bin_path = 'convert_test.bin'
        self.args = argparse.Namespace(input_path=self.npy_path)

    def tearDown(self):
        if os.path.exists(self.npy_path):
            os.remove(self.npy_path)
        if os.path.exists(self.bin_path):
            os.remove(self.bin_path)

    def test_convert_npy_to_bin(self):
        # create a test npy file
        npy_data = np.array([1, 2, 3])
        np.save(self.npy_path, npy_data)

        # call the function to convert npy to bin
        convert_npy_to_bin(self.args.input_path)

        # check if the bin file is generated
        assert os.path.exists(self.bin_path)


class TestConvertFunctions(unittest.TestCase):

    @patch("msprobe.infer.offline.compare.msquickcmp.common.convert.logger")
    @patch("msprobe.infer.offline.compare.msquickcmp.common.convert.execute_command")
    @patch("msprobe.infer.offline.compare.msquickcmp.common.convert.os.path.join")
    @patch("msprobe.infer.offline.compare.msquickcmp.common.convert.sys.executable", new="/usr/bin/python3")
    def test_convert_bin_file_to_npy(
        self,
        mock_join,
        mock_execute,
        mock_logger
    ):
        # Setup
        bin_file_path = "/some/file.bin"
        npy_dir_path = "/some/output"
        cann_path = "/opt/cann"
        mock_join.side_effect = lambda *args: "/".join(args)

        convert_bin_file_to_npy(bin_file_path, npy_dir_path, cann_path)

        expected_cmd = [
            "python3",
            "/opt/cann/toolkit/tools/operator_cmp/compare/msaccucmp.py",
            "convert",
            "-d",
            bin_file_path,
            "-out",
            npy_dir_path,
        ]

        mock_logger.info.assert_called_once()
        mock_execute.assert_called_once_with(expected_cmd)



