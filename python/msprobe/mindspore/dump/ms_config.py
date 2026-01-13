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

from msprobe.core.common.const import Const
from msprobe.core.dump.common_config import BaseConfig, CommonConfig


class TensorConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.check_mode = None
        self.file_format = json_config.get("file_format")
        self.check_config()
        self._check_summary_mode()
        self._check_config()

    def _check_config(self):
        if self.file_format and self.file_format not in ["npy", "bin"]:
            raise Exception("file_format is invalid")


class StatisticsConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.file_format = None
        self.check_mode = None
        self.check_config()
        self._check_summary_mode()

        self.tensor_list = json_config.get("tensor_list", [])
        self._check_str_list_config(self.tensor_list, "tensor_list")
        self.stat_cal_mode = json_config.get("device", "host")
        self.device_stat_precision_mode = json_config.get("precision", "high")
        self._check_stat_params()

    def _check_stat_params(self):
        if self.stat_cal_mode not in ["device", "host"]:
            raise Exception("Config param [device] is invalid, expected from [\"device\", \"host\"]")
        if self.device_stat_precision_mode not in ["high", "low"]:
            raise Exception("Config param [precision] is invalid, expected from [\"high\", \"low\"]")

    def _check_summary_mode(self):
        muti_opt = ["max", "min", "mean", "count", "negative zero count", "positive zero count", "nan count",
                    "negative inf count", "positive inf count", "zero count", "l2norm", "hash", "md5"]
        if isinstance(self.summary_mode, str) and self.summary_mode not in Const.SUMMARY_MODE:
            raise Exception("summary_mode is an invalid string")
        if isinstance(self.summary_mode, list) and not all(opt in muti_opt for opt in self.summary_mode):
            raise Exception("summary_mode contains invalid option(s)")


class OverflowCheckConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.data_mode = ["all"]
        self._check_config()

    def _check_config(self):
        if self.check_mode and self.check_mode not in ["all", "aicore", "atomic"]:
            raise Exception("check_mode is invalid")


class ExceptionDumpConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.data_mode = ["all"]


class StructureConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)


TaskDict = {
    Const.TENSOR: TensorConfig,
    Const.STATISTICS: StatisticsConfig,
    Const.OVERFLOW_CHECK: OverflowCheckConfig,
    Const.STRUCTURE: StructureConfig,
    Const.EXCEPTION_DUMP: ExceptionDumpConfig
}


def parse_common_config(json_config):
    return CommonConfig(json_config)


def parse_task_config(task, json_config):
    task_map = json_config.get(task)
    if not task_map:
        task_map = dict()
    if task not in TaskDict:
        raise Exception("task is invalid.")
    return TaskDict.get(task)(task_map)
