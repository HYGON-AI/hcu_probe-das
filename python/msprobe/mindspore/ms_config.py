# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
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

from msprobe.core.common.const import Const
from msprobe.core.common.utils import is_int
from msprobe.core.common_config import BaseConfig, CommonConfig
from msprobe.core.grad_probe.constant import level_adp
from msprobe.core.grad_probe.utils import check_numeral_list_ascend
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.common.log import logger


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
        if self.overflow_nums is not None and not is_int(self.overflow_nums):
            raise Exception("overflow_nums is invalid, it should be an integer")
        if self.overflow_nums is not None and self.overflow_nums != -1 and self.overflow_nums <= 0:
            raise Exception("overflow_nums should be -1 or positive integer")
        if self.check_mode and self.check_mode not in ["all", "aicore", "atomic"]:
            raise Exception("check_mode is invalid")


class ExceptionDumpConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.data_mode = ["all"]


class FreeBenchmarkConfig(BaseConfig):
    def __init__(self, task_config):
        super().__init__(task_config)
        self._check_config()

    def _check_config(self):
        if self.fuzz_device and self.fuzz_device not in FreeBenchmarkConst.DEVICE_LIST:
            raise Exception("fuzz_device must be npu or empty")
        if self.pert_mode and self.pert_mode not in FreeBenchmarkConst.PERT_TYPE_LIST:
            raise Exception("pert_mode must be improve_precision, add_noise, bit_noise, "
                            "no_change, change_value or empty")
        if self.handler_type and self.handler_type not in FreeBenchmarkConst.HANDLER_TYPE_LIST:
            raise Exception("handler_type must be check, fix or empty")
        if self.fuzz_level and self.fuzz_level not in FreeBenchmarkConst.DUMP_LEVEL_LIST:
            raise Exception("fuzz_level must be L1 or empty")
        if self.fuzz_stage and self.fuzz_stage not in FreeBenchmarkConst.STAGE_LIST:
            raise Exception("fuzz_stage must be forward, backward or empty")
        if self.if_preheat or self.preheat_step or self.max_sample:
            logger.warning("'if_preheat', 'preheat_step' and 'max_sample' settings "
                           "are not supported for mindspore free benchmark task.")


class GradProbeConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.grad_level = json_config.get("grad_level", "L1")
        self.param_list = json_config.get("param_list", [])
        self.bounds = json_config.get("bounds", [-1, 0, 1])
        self._check_config()

    def _check_config(self):
        if self.grad_level not in level_adp.keys():
            raise Exception(f"grad_level must be one of {level_adp.keys()}")
        if not isinstance(self.param_list, list):
            raise Exception("param_list must be a list")
        check_numeral_list_ascend(self.bounds)


class StructureConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)


TaskDict = {
    Const.TENSOR: TensorConfig,
    Const.STATISTICS: StatisticsConfig,
    Const.OVERFLOW_CHECK: OverflowCheckConfig,
    Const.FREE_BENCHMARK: FreeBenchmarkConfig,
    Const.GRAD_PROBE: GradProbeConfig,
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
