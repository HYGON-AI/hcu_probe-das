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
import time

from msprobe.core.dump.debugger.precision_debugger import BasePrecisionDebugger
from msprobe.pytorch.dump.debugger.debugger_config import DebuggerConfig
from msprobe.pytorch.dump.pt_config import parse_task_config
from msprobe.pytorch.dump.pytorch_service import PytorchService

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.utils import check_token_range, ThreadSafe, check_rank_id, get_real_step_or_rank
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import check_save_param, is_torch_nn_module
from msprobe.pytorch.dump.module_dump.module_dump import ModuleDumper


class PrecisionDebugger(BasePrecisionDebugger):
    _CONFIG_CHECK_INTERVAL_ENABLED_S = 0.5
    _CONFIG_CHECK_INTERVAL_DISABLED_S = 3.0

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
        self._overrides = {
            "config_path": config_path,
            "task": task,
            "dump_path": dump_path,
            "level": level,
            "step": step,
        }
        self._reload_state = {
            "last_check_ts": 0.0,
            "signature": self._get_config_signature(),
        }
        self._dynamic_dump_enable_active = self.common_config.dump_enable is not None
        self.config = self._create_debugger_config(self.common_config, self.task_config)
        self.service = PytorchService(self.config)
        self.module_dumper = ModuleDumper(self.service)
        self.ori_customer_func = {}

    @staticmethod
    def _get_task_config(task, json_config):
        return parse_task_config(task, json_config)

    @classmethod
    @ThreadSafe.synchronized
    def start(cls, model=None, token_range=None, rank_id=None):
        instance = cls._instance_with_reload()

        check_token_range(token_range)
        check_rank_id(rank_id)
        instance.config.check_model(model, token_range)
        instance.service.start(model, token_range, rank_id)

    @classmethod
    @ThreadSafe.synchronized
    def stop(cls):
        cls._run_with_reload(lambda instance: instance.service.stop())

    @classmethod
    @ThreadSafe.synchronized
    def step(cls):
        cls._run_with_reload(lambda instance: instance.service.step())

    @classmethod
    @ThreadSafe.synchronized
    def save(cls, variable, name, save_backward=True):
        instance = cls._get_instance()
        if not instance._is_debug_save_enabled():
            return
        try:
            check_save_param(variable, name, save_backward)
        except ValueError:
            return
        instance.service.save(variable, name, save_backward)

    @classmethod
    def _instance_with_reload(cls):
        instance = cls._get_instance()
        instance._maybe_reload_config()
        return instance

    @classmethod
    def _run_with_reload(cls, action):
        instance = cls._get_instance()
        action(instance)
        instance._maybe_reload_config()

    def _is_debug_save_enabled(self):
        return self.task in [Const.TENSOR, Const.STATISTICS] and self.config.level == Const.LEVEL_DEBUG

    def _get_config_signature(self):
        config_path = self._overrides.get("config_path")
        if not config_path:
            return None
        try:
            stat_result = os.stat(config_path)
        except OSError:
            return None
        return stat_result.st_mtime_ns, stat_result.st_size

    def _get_changed_config_signature(self, force=False):
        now = time.monotonic()
        check_interval = self._get_config_check_interval_s()
        if not force and now - self._reload_state["last_check_ts"] < check_interval:
            return None
        self._reload_state["last_check_ts"] = now
        current_signature = self._get_config_signature()
        if current_signature is None or current_signature == self._reload_state["signature"]:
            return None
        return current_signature

    def _get_config_check_interval_s(self):
        dump_enable = getattr(self.config, "dump_enable", None)
        return (
            self._CONFIG_CHECK_INTERVAL_ENABLED_S
            if self._is_dump_enabled(dump_enable)
            else self._CONFIG_CHECK_INTERVAL_DISABLED_S
        )

    def _maybe_reload_config(self, force=False):
        if not self._dynamic_dump_enable_active:
            return False
        pending_signature = self._get_changed_config_signature(force=force)
        if pending_signature is None:
            return False
        return self._reload_config(pending_signature)

    def _create_debugger_config(self, common_config, task_config):
        return DebuggerConfig(
            common_config,
            task_config,
            self._overrides["task"],
            self._overrides["dump_path"],
            self._overrides["level"]
        )

    def _reload_config(self, pending_signature):
        try:
            common_config, task_config = self._parse_config_path(
                self._overrides["config_path"],
                self._overrides["task"]
            )
            if self._overrides["step"] is not None:
                common_config.step = get_real_step_or_rank(self._overrides["step"], Const.STEP)
            new_config = self._create_debugger_config(common_config, task_config)
        except Exception as ex:
            self._fail_close_dump()
            logger.warning(f"Config hot reload skipped because parsing failed: {ex}")
            return False

        self._apply_reloaded_config(common_config, task_config, new_config, pending_signature)
        logger.info("PrecisionDebugger detected config change and reloaded runtime settings.")
        return True

    def _apply_reloaded_config(self, common_config, task_config, new_config, signature):
        previous_dump_enable = getattr(self.config, "dump_enable", None)
        # In dynamic mode, deleting dump_enable keeps the previous state.
        if self._dynamic_dump_enable_active and new_config.dump_enable is None:
            new_config.dump_enable = previous_dump_enable
            common_config.dump_enable = previous_dump_enable

        if self._is_dump_enabled(previous_dump_enable) and not self._is_dump_enabled(new_config.dump_enable):
            # Use unified stop path to flush existing buffered data before turning dump off.
            self.service.stop()

        self.common_config = common_config
        self.task_config = task_config
        self.task = common_config.task
        self.config = new_config
        self.service.apply_runtime_config(new_config)
        self._reload_state["signature"] = signature

    def _fail_close_dump(self):
        previous_dump_enable = getattr(self.config, "dump_enable", None)
        if self._is_dump_enabled(previous_dump_enable):
            self.service.stop()
        self.config.dump_enable = False
        self.common_config.dump_enable = False
        self.service.apply_runtime_config(self.config)

    @staticmethod
    def _is_dump_enabled(dump_enable):
        return True if dump_enable is None else dump_enable


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
    instance = _get_debugger_instance()
    instance.module_dumper.start_module_dump(module, dump_name)


@ThreadSafe.synchronized
def module_dump_end():
    instance = _get_debugger_instance()
    instance.module_dumper.stop_module_dump()


def _get_debugger_instance():
    instance = PrecisionDebugger._instance
    if instance:
        return instance
    raise MsprobeException(
        MsprobeException.INTERFACE_USAGE_ERROR,
        "PrecisionDebugger must be instantiated before using module_dump interfaces"
    )
