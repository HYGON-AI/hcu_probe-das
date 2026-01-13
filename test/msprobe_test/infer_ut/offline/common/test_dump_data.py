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

import unittest
from unittest.mock import patch

from msprobe.infer.offline.compare.msquickcmp.common.dump_data import DumpData  # 假设 DumpData 类在 your_module.py 文件中


class TestDumpData(unittest.TestCase):

    def test_to_valid_name(self):
        # 测试 _to_valid_name 方法
        dump_data = DumpData()
        valid_name = dump_data._to_valid_name("model.name/with.slash")
        self.assertEqual(valid_name, "model_name_with_slash")

    @patch('os.path.exists')
    @patch('os.access')
    def test_check_path_exists_valid(self, mock_access, mock_exists):
        # 测试路径存在且可读
        mock_exists.return_value = True
        mock_access.return_value = True

        dump_data = DumpData()
        dump_data._check_path_exists("./mock/path.om", extentions=[".om"])

    @patch('time.time', return_value=1627551000)
    def test_generate_dump_data_file_name(self, mock_time):
        # 测试生成文件名
        dump_data = DumpData()
        file_name = dump_data._generate_dump_data_file_name("model.name", 1)
        expected_file_name = "model_name.1.1627551000000000.npy"
        self.assertEqual(file_name, expected_file_name)

    def test_check_input_data_path_valid(self):
        # 测试输入数据路径检查有效
        dump_data = DumpData()
        input_paths = ["/mock/input1.bin", "/mock/input2.bin"]
        inputs_tensor_info = [{"name": "input1", "shape": (1, 1)}, {"name": "input2", "shape": (1, 1)}]

        with patch('os.path.exists', return_value=True):
            dump_data._check_input_data_path(input_paths, inputs_tensor_info)

