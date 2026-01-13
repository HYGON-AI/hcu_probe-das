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

from msprobe.core.common.file_utils import create_directory, save_json
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig


class KernelGraphExceptionDump:

    def __init__(self, config: DebuggerConfig):
        self.dump_json = dict()
        self.dump_json["common_dump_settings"] = dict()
        self.dump_json["common_dump_settings"]["dump_mode"] = 0
        self.dump_json["common_dump_settings"]["path"] = ""
        self.dump_json["common_dump_settings"]["net_name"] = "Net"
        self.dump_json["common_dump_settings"]["iteration"] = "all"
        self.dump_json["common_dump_settings"]["saved_data"] = "tensor"
        self.dump_json["common_dump_settings"]["input_output"] = 0
        self.dump_json["common_dump_settings"]["kernels"] = []
        self.dump_json["common_dump_settings"]["support_device"] = [0, 1, 2, 3, 4, 5, 6, 7]
        self.dump_json["common_dump_settings"]["op_debug_mode"] = 4
        self.dump_json["common_dump_settings"]["file_format"] = "npy"
        self.dump_json["e2e_dump_settings"] = dict()
        self.dump_json["e2e_dump_settings"]["enable"] = not config.async_dump
        self.dump_json["e2e_dump_settings"]["trans_flag"] = True

        if config.stat_cal_mode and config.device_stat_precision_mode:
            self.dump_json["e2e_dump_settings"]["stat_calc_mode"] = config.stat_cal_mode
            self.dump_json["e2e_dump_settings"]["device_stat_precision_mode"] = config.device_stat_precision_mode
        self.dump_json["common_dump_settings"]["path"] = config.dump_path
        if len(config.step) > 0:
            logger.warning("Step would change to all in this task.")
        if len(config.rank) > 0:
            self.dump_json["common_dump_settings"]["support_device"] = config.rank

    def handle(self):
        json_path = self.dump_json["common_dump_settings"]["path"]
        create_directory(json_path)
        json_path = os.path.join(json_path, "kernel_graph_exception_check.json")
        save_json(json_path, self.dump_json, indent=4)
        logger.info(json_path + " has been created.")
        os.environ["MINDSPORE_DUMP_CONFIG"] = json_path
