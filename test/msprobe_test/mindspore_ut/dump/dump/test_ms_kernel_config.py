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
from unittest.mock import patch

from msprobe.core.dump.kernel_dump.kernel_config import create_kernel_config_json


class TestMsKernelConfig(unittest.TestCase):
    @patch("msprobe.core.dump.kernel_dump.kernel_config.save_json")
    def test_create_kernel_config_json(self, mock_save_json):
        dump_path = "./step0"
        cur_pid = os.getpid()
        kernel_config_path = create_kernel_config_json(dump_path, cur_pid)
        self.assertEqual(kernel_config_path, f"./step0/kernel_config_{cur_pid}.json")
        config_info = {
            "dump": {
                "dump_list": [],
                "dump_path": dump_path,
                "dump_mode": "all",
                "dump_op_switch": "on"
            }
        }
        mock_save_json.assert_called_once_with(kernel_config_path, config_info, indent=4)
