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

from collections import defaultdict
from functools import wraps

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.log import logger

# 记录工具函数递归的深度
recursion_depth = defaultdict(int)


def recursion_depth_decorator(func_info, max_depth=Const.MAX_DEPTH):
    """装饰一个函数，当函数递归调用超过限制时，抛出异常并打印函数信息。"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_id = id(func)
            recursion_depth[func_id] += 1
            if recursion_depth[func_id] > max_depth:
                msg = f"call {func_info} exceeds the recursion limit."
                logger.error_log_with_exp(
                    msg,
                    MsprobeException(
                        MsprobeException.RECURSION_LIMIT_ERROR, msg
                    ),
                )
            try:
                result = func(*args, **kwargs)
            finally:
                recursion_depth[func_id] -= 1
            return result

        return wrapper

    return decorator
