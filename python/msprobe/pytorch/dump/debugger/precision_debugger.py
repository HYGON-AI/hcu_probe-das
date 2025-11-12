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

from msprobe.core.dump.debugger.precision_debugger import BasePrecisionDebugger
from msprobe.pytorch.dump.debugger.debugger_config import DebuggerConfig
from msprobe.pytorch.dump.pt_config import parse_task_config
from msprobe.pytorch.dump.pytorch_service import PytorchService

from msprobe.core.common.const import Const, MsgConst
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.utils import check_token_range, ThreadSafe
from msprobe.pytorch.common.utils import check_save_param, is_torch_nn_module
from msprobe.pytorch.dump.module_dump.module_dump import ModuleDumper


class PrecisionDebugger(BasePrecisionDebugger):

    def __init__(
        self,
        config_path=None,
        task=None,
        dump_path=None,
        level=None,
        step=None
    ):
        if self.initialized:
            return
        super().__init__(config_path, task, dump_path, level, step)
        self.config = DebuggerConfig(
            self.common_config,
            self.task_config,
            task,
            dump_path,
            level
        )
        self.service = PytorchService(self.config)
        self.module_dumper = ModuleDumper(self.service)
        self.ori_customer_func = {}

    @staticmethod
    def _get_task_config(task, json_config):
        return parse_task_config(task, json_config)

    @classmethod
    @ThreadSafe.synchronized
    def start(cls, model=None, token_range=None):
        instance = cls._get_instance()
        if instance is None:
            return

        check_token_range(token_range)
        instance.config.check_model(model, token_range)
        instance.service.start(model, token_range)

    @classmethod
    @ThreadSafe.synchronized
    def stop(cls):
        instance = cls._get_instance()
        if instance is None:
            return
        instance.service.stop()

    @classmethod
    @ThreadSafe.synchronized
    def step(cls):
        instance = cls._get_instance()
        if instance is None:
            return
        cls._instance.service.step()

    @classmethod
    @ThreadSafe.synchronized
    def save(cls, variable, name, save_backward=True):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if instance.task not in [Const.TENSOR, Const.STATISTICS] or instance.config.level != Const.LEVEL_DEBUG:
            return
        try:
            check_save_param(variable, name, save_backward)
        except ValueError:
            return
        instance.service.save(variable, name, save_backward)


@ThreadSafe.synchronized
def module_dump(module, dump_name):
    if not is_torch_nn_module(module):
        raise MsprobeException(
            MsprobeException.INVALID_PARAM_ERROR,
            f"the module argument in module_dump must be a torch.nn.Module type, "
            f"but currently there is an unsupported {type(module)} type."
        )
    if not isinstance(dump_name, str):
        raise MsprobeException(
            MsprobeException.INVALID_PARAM_ERROR,
            f"the dump_name argument in module_dump must be a str type"
        )
    instance = PrecisionDebugger._instance
    if not instance:
        raise MsprobeException(
            MsprobeException.INTERFACE_USAGE_ERROR,
            f"PrecisionDebugger must be instantiated before using module_dump interface"
        )
    instance.module_dumper.start_module_dump(module, dump_name)


@ThreadSafe.synchronized
def module_dump_end():
    instance = PrecisionDebugger._instance
    if not instance:
        raise MsprobeException(
            MsprobeException.INTERFACE_USAGE_ERROR,
            f"PrecisionDebugger must be instantiated before using module_dump_end interface"
        )
    instance.module_dumper.stop_module_dump()
