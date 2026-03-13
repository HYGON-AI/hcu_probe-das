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
import threading
import functools
import importlib
import types
import weakref

import torch

from msprobe.core.common.log import logger
from msprobe.pytorch.common.utils import torch_version_above_or_equal_2
from msprobe.pytorch.dump.api_dump.api_register import get_api_register
from msprobe.core.common.runtime import Runtime
from msprobe.core.common.utils import ThreadSafe
from msprobe.core.dump.hook_manager import BaseHookManager
from msprobe.core.dump.data_dump.data_processor.base import ModuleForwardInputsOutputs
from msprobe.pytorch.dump.api_dump.hook_module import HOOKModule
from msprobe.core.common.const import Const


if torch_version_above_or_equal_2:
    from torch._dynamo.convert_frame import convert_frame as _orig_convert_frame, Hooks


def wrap_jit_script_func():
    def patched_script(*args, **kwargs):
        all_api_registered = api_register.all_api_registered
        if all_api_registered:
            api_register.restore_all_api()
        result = original_script(*args, **kwargs)
        if all_api_registered:
            api_register.register_all_api()
        return result

    original_script = torch.jit.script
    api_register = get_api_register()
    torch.jit.script = patched_script


def wrap_compile_script_func():
    def _patched_convert_frame(compiler_fn, hooks):
        """
        在调用原 convert_frame 生成的 _convert_frame 之前恢复 API，
        调用完之后再重新注册所有 API。
        """
        # 拿到原来 inner 版的 _convert_frame
        inner_convert = _orig_convert_frame(compiler_fn, hooks)

        def _wrapped(frame: types.FrameType, cache_size: int, hooks: Hooks, frame_state):
            reg = get_api_register()
            # 进入前 restore
            reg.restore_all_api()
            try:
                result = inner_convert(frame, cache_size, hooks, frame_state)
            except Exception:
                # 异常时也要确保 register
                reg.register_all_api()
                raise
            # 正常结束后 register
            reg.register_all_api()
            return result

        # 保留原属性以兼容
        _wrapped._torchdynamo_orig_callable = compiler_fn  # type: ignore[attr-defined]

        _wrapped._clone_with_backend =\
            lambda backend: _patched_convert_frame(backend, hooks)  # type: ignore[attr-defined]

        return _wrapped

    import torch._dynamo.convert_frame as _cf_mod
    _cf_mod.convert_frame = _patched_convert_frame


def patch_dynamo_compile():
    cf = importlib.import_module("torch._dynamo.convert_frame")
    if not hasattr(cf, "_compile"):
        logger.warning("No found torch._dynamo.convert_frame._compile")

    original = cf._compile
    if getattr(original, "__msprobe_patched__", False):
        return

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        result = None
        try:
            reg = get_api_register()
            reg.restore_all_api()
        except Exception as e:
            logger.warning(f"[msprobe] Pre restore_all_api failed: {e}")
            return result

        try:
            result = original(*args, **kwargs)
        except Exception:
            logger.warning("[msprobe] _compile execution failed (returning None)")
            result = None
        finally:
            try:
                reg = get_api_register()
                reg.register_all_api()  # 改成注册hook
            except Exception as e:
                logger.warning(f"[msprobe] Post register_all_api failed: {e}")
        return result
    wrapped.__msprobe_patched__ = True
    wrapped.__msprobe_original__ = original
    cf._compile = wrapped


def unpatch_dynamo_compile() -> bool:
    # 预留取消patch接口
    cf = importlib.import_module("torch._dynamo.convert_frame")
    current = getattr(cf, "_compile", None)
    if current is None:
        return False
    original = getattr(current, "__msprobe_original__", None)
    if original is None:
        return False
    cf._compile = original
    return True


def preprocess_func():
    try:
        from torch.utils._device import _device_constructors
        _device_constructors()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to execute _device_constructors. Error Details: {str(e)}")


_service_ref = None

def set_current_service(service):
    """由 PytorchService 在初始化时注入，避免 script_wrapper 反向 import。"""
    global _service_ref
    _service_ref = weakref.ref(service)

def get_current_service():
    return _service_ref() if _service_ref else None

def patch_triton_jitfunction_run():
    try:
        from triton.runtime import JITFunction
    except Exception as e:
        logger.warning(f"[msprobe] Triton not available, skip patch JITFunction.run: {e}")
        return

    original_run = getattr(JITFunction, "run", None)
    if original_run is None:
        logger.warning("[msprobe] triton.runtime.JITFunction has no attribute 'run', skip patch.")
        return

    if getattr(original_run, "__msprobe_patched__", False):
        return

    @functools.wraps(original_run)
    def wrapped_run(self, *args, **kwargs):
        # ===== 0) 开关 + 防递归 =====
        tid = threading.get_ident()

        # 不在运行态，不采集
        if not Runtime.is_running:
            return original_run(self, *args, **kwargs)

        # 防止 hook 内部再次触发 hook
        if BaseHookManager.inner_switch.get(tid, False):
            return original_run(self, *args, **kwargs)

        service = get_current_service()

        if service is None:
            return original_run(self, *args, **kwargs)

        data_collector = service.data_collector

        # stop/step 会更新 is_running，但这里再加一道 should_stop_service 判断更稳
        if getattr(service, "should_stop_service", False):
            return original_run(self, *args, **kwargs)

        data_collector = service.data_collector
        pid = os.getpid()

        # ===== 1) 生成 api_name + per-step 计数 =====
        try:
            api_name = f"Triton.{self.fn.__qualname__}"
        except Exception:
            api_name = str(getattr(self, "fn", "unknown_triton_fn"))

        # ✅ 复用 HOOKModule.module_count（每 step 已经 reset）
        count = HOOKModule.get_module_count(api_name)
        HOOKModule.add_module_count(api_name)

        # 命名风格尽量对齐：API + count + forward
        full_name = f"{api_name}{Const.SEP}{count}{Const.SEP}{Const.FORWARD}"

        # ===== 2) forward_pre: 输入采集（线程安全 + inner_switch）=====
        with ThreadSafe():
            BaseHookManager.inner_switch[tid] = True
            try:
                data_collector.update_api_or_module_name(full_name)
                module_io = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=None)
                data_collector.forward_input_data_collect(full_name, self, pid, module_io)
            except Exception as e:
                logger.warning(f"[msprobe] Triton forward_input_data_collect failed: {e}")
            finally:
                BaseHookManager.inner_switch[tid] = False

        # ===== 3) 执行原始 triton run =====
        out = original_run(self, *args, **kwargs)

        # ===== 4) forward_hook: 输出采集 =====
        with ThreadSafe():
            BaseHookManager.inner_switch[tid] = True
            try:
                data_collector.update_api_or_module_name(full_name)
                module_io = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=args)
                data_collector.forward_output_data_collect(full_name, self, pid, module_io)
            except Exception as e:
                logger.warning(f"[msprobe] Triton forward_output_data_collect failed: {e}")
            finally:
                BaseHookManager.inner_switch[tid] = False

        return out

    wrapped_run.__msprobe_patched__ = True
    wrapped_run.__msprobe_original__ = original_run
    setattr(JITFunction, "run", wrapped_run)
    logger.info("[msprobe] Patched triton.runtime.JITFunction.run successfully.")

def unpatch_triton_jitfunction_run() -> bool:
    try:
        from triton.runtime import JITFunction
    except Exception:
        return False
    current = getattr(JITFunction, "run", None)
    if current is None:
        return False
    original = getattr(current, "__msprobe_original__", None)
    if original is None:
        return False
    setattr(JITFunction, "run", original)
    return True


def adapt_megatron_distributed_mappings():
    """
    megatron框架定义的变量dist_all_gather_func和dist_reduce_scatter_func会随着import链路初始化，指向msprobe工具patch前的dist接口

    源码示例:
    -------------------------------------------------------------------------------------------------------------
    import ...
    from ... import ...
    if is_torch_min_version("1.13.0"):
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
        dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
    else:
        dist_all_gather_func = torch.distributed._all_gather_base
        dist_reduce_scatter_func = torch.distributed._reduce_scatter_base

    # 此时dist_all_gather_func和dist_reduce_scatter_func已初始化完成，指向msprobe工具patch前的dist接口
    ... ...

    dist_all_gather_func(...)
    dist_reduce_scatter_func(...) # 实际调用msprobe工具patch前的dist接口，无法dump数据
    -------------------------------------------------------------------------------------------------------------

    本函数将megatron涉及的模块变量dist_all_gather_func和dist_reduce_scatter_func重新指向msprobe工具patch后的dist接口，
    避免dist接口无法dump
    """

    # 到megatron core_v0.12.1版本共有6个模块涉及
    megatron_module_path = [
        "megatron.core.tensor_parallel.mappings",
        "megatron.core.tensor_parallel.utils",
        "megatron.core.tensor_parallel.layers",
        "megatron.core.distributed.param_and_grad_buffer",
        "megatron.core.utils",
        "megatron.core.timers",
    ]

    def adapt_single_module(module):
        """适配单个模块中的分布式函数映射"""
        # 更新all_gather函数映射
        all_gather_func = getattr(module, "dist_all_gather_func", None)
        if all_gather_func:
            if str(all_gather_func).startswith('<function all_gather_into_tensor'):
                module.dist_all_gather_func = torch.distributed.all_gather_into_tensor
            elif str(all_gather_func).startswith('<function _all_gather_base'):
                module.dist_all_gather_func = torch.distributed._all_gather_base

        # 更新reduce_scatter函数映射
        reduce_scatter_func = getattr(module, "dist_reduce_scatter_func", None)
        if reduce_scatter_func:
            if str(reduce_scatter_func).startswith('<function reduce_scatter_tensor'):
                module.dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
            elif str(reduce_scatter_func).startswith('<function _reduce_scatter_base'):
                module.dist_reduce_scatter_func = torch.distributed._reduce_scatter_base

    try:
        import megatron
    except ImportError:
        return

    for module_path in megatron_module_path:
        try:
            megatron_module = importlib.import_module(module_path)
            adapt_single_module(megatron_module)
        except ImportError:
            logger.warning(f'Import {module_path} failed, skip mapping.')
            continue
        except Exception as e:
            logger.warning(f'An unexpected error occurred in the function "adapt_megatron_distributed_mappings", '
                           f'skip mapping: {e}')
            continue

def wrap_script_func():
    wrap_jit_script_func()
    if torch_version_above_or_equal_2:
        patch_dynamo_compile()
    patch_triton_jitfunction_run()
    adapt_megatron_distributed_mappings()
