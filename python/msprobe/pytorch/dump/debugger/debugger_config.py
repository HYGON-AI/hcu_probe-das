# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import is_torch_nn_module


class DebuggerConfig:
    def __init__(self, common_config, task_config, task, dump_path, level):
        self.dump_path = dump_path if dump_path else common_config.dump_path
        self.task = task or common_config.task or Const.STATISTICS
        self.rank = common_config.rank if common_config.rank else []
        self.step = common_config.step if common_config.step else []
        self.level = level or common_config.level or Const.LEVEL_L1
        self.scope = task_config.scope if task_config.scope else []
        self.list = task_config.list if task_config.list else []
        self.data_mode = task_config.data_mode if task_config.data_mode else ["all"]
        self.summary_mode = task_config.summary_mode if task_config.summary_mode else Const.STATISTICS
        self.framework = Const.PT_FRAMEWORK
        self.async_dump = common_config.async_dump if common_config.async_dump else False
        self.precision = common_config.precision if common_config.precision else Const.DUMP_PRECISION_LOW

        self.check()
        self._check_statistics_config(task_config)

        if self.level == Const.LEVEL_L2:
            self.is_backward_kernel_dump = False
            self._check_and_adjust_config_with_l2()

    def check(self):
        if self.task and self.task not in Const.TASK_LIST:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"The task <{self.task}> is not in the {Const.TASK_LIST}.")
        if self.level and self.level not in Const.LEVEL_LIST:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"The level <{self.level}> is not in the {Const.LEVEL_LIST}.")
        if not self.dump_path:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"The dump_path not found.")
        if not isinstance(self.async_dump, bool):
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"The parameters async_dump should be bool.")
        if self.task == Const.STRUCTURE and self.level not in [Const.LEVEL_L0, Const.LEVEL_MIX]:
            logger.warning_on_rank_0(
                f"When the task is set to structure, the level should be one of {[Const.LEVEL_L0, Const.LEVEL_MIX]}. "
                f"If not, the default level is {Const.LEVEL_MIX}."
            )
            self.level = Const.LEVEL_MIX
        if self.async_dump:
            if self.task == Const.TENSOR:
                if self.level == Const.LEVEL_DEBUG:
                    self.list = []  # async_dump + debug level case ignore list
                if not self.list and self.level != Const.LEVEL_DEBUG:
                    raise MsprobeException(
                        MsprobeException.INVALID_PARAM_ERROR,
                        f"The parameters async_dump is true in tensor task, the parameters list cannot be empty."
                    )
            if self.summary_mode == Const.MD5:
                raise MsprobeException(
                    MsprobeException.INVALID_PARAM_ERROR,
                    f"The parameters async_dump is true, the parameters summary_mode cannot be md5."
                )
        return True

    def check_model(self, models, token_range=None):
        if token_range and not models:
            error_info = "The 'model' parameter must be provided when token_range is not None"
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, error_info)

        if self.level not in [Const.LEVEL_L0, Const.LEVEL_MIX] and token_range is None:
            return

        if models is None:
            logger.error_on_rank_0(
                f"For level {self.level} or non-empty token_range, "
                f"PrecisionDebugger or start interface must receive a 'model' parameter.")
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, f"missing the parameter 'model'")

        if is_torch_nn_module(models):
            return

        if isinstance(models, (list, tuple)):
            error_model = None
            for model in models:
                if not is_torch_nn_module(model):
                    error_model = model
                    break
            if error_model is not None:
                error_info = (f"The 'model' parameter must be a torch.nn.Module or list[torch.nn.Module] "
                              f"type, currently there is an unsupported {type(error_model)} type.")
                raise MsprobeException(
                    MsprobeException.INVALID_PARAM_ERROR, error_info)
        else:
            error_info = (f"The 'model' parameter must be a torch.nn.Module or list[torch.nn.Module] "
                          f"type, currently there is an unsupported {type(models)} type.")
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, error_info)

    def _check_and_adjust_config_with_l2(self):
        if self.scope:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"When level is set to L2, the scope cannot be configured.")
        if not self.list or len(self.list) != 1:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"When level is set to L2, the list must be configured as a list with one api name.")
        if self.task != Const.TENSOR:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"When level is set to L2, the task must be set to tensor.")

        api_name = self.list[0]
        if api_name.endswith(Const.BACKWARD):
            self.is_backward_kernel_dump = True
            api_forward_name = api_name[:-len(Const.BACKWARD)] + Const.FORWARD
            self.list.append(api_forward_name)

    def _check_statistics_config(self, task_config):
        if self.task != Const.STATISTICS:
            return
        self.tensor_list = []
        if not hasattr(task_config, "tensor_list"):
            return
        if self.level == Const.LEVEL_DEBUG and task_config.tensor_list:
            logger.warning_on_rank_0("When level is set to debug, the tensor_list will be invalid.")
            return
        self.tensor_list = task_config.tensor_list
