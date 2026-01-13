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

from msprobe.core.common.file_utils import save_json


def create_kernel_config_json(dump_path, cur_rank):
    kernel_config_name = "kernel_config.json" if cur_rank == '' else f"kernel_config_{cur_rank}.json"
    kernel_config_path = os.path.join(dump_path, kernel_config_name)
    config_info = {
        "dump": {
            "dump_list": [],
            "dump_path": dump_path,
            "dump_mode": "all",
            "dump_op_switch": "on"
        }
    }
    save_json(kernel_config_path, config_info, indent=4)
    return kernel_config_path
