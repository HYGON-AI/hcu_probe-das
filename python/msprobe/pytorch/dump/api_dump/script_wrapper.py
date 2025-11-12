# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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

import functools
import importlib
import types

import torch

from msprobe.core.common.log import logger
from msprobe.pytorch.common.utils import torch_version_above_or_equal_2
from msprobe.pytorch.dump.api_dump.api_register import get_api_register

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


def wrap_script_func():
    wrap_jit_script_func()
    if torch_version_above_or_equal_2:
        patch_dynamo_compile()
