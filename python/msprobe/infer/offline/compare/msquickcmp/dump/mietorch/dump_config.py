# -*- coding: utf-8 -*-
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

import json
import os.path

from msprobe.core.common.log import logger
from msprobe.infer.utils.file_open_check import ms_open


class DumpConfig:
    def __init__(
        self,
        dump_path=".",
        mode='all',
        op_switch="off",
        api_list=None,
    ):
        dump_list_config = dict(model_name="Graph")
        if api_list:
            dump_list_config["layer"] = [api for api in api_list.split(",") if api]
        self.config = dict(
            dump=dict(
                dump_path=dump_path,
                dump_mode=mode,
                dump_op_switch=op_switch,
                dump_list=[dump_list_config]
            )
        )
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(cur_dir, "acl.json")
        try:
            with ms_open(config_path, "w") as f:
                json.dump(self.config, f, indent=4)
        except FileNotFoundError:
            logger.error("File not found.")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error:{e}")
            raise


