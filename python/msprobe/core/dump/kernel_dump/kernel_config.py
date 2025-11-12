# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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
