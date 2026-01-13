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
