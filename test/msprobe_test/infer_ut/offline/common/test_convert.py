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



