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

from msprobe.core.common.const import Data2DBConst
def process_tensor_value(value):
    """处理统计量值  限制在float范围"""
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    if value == float("inf"):
        return Data2DBConst.MAX_FLOAT_VALUE + 1  # 大于最大float值
    elif value == float("-inf"):
        return Data2DBConst.MIN_FLOAT_VALUE - 1  # 小于最小float值
    elif value > Data2DBConst.MAX_FLOAT_VALUE:
        return Data2DBConst.MAX_FLOAT_VALUE  # 最大float值
    elif value < Data2DBConst.MIN_FLOAT_VALUE:
        return Data2DBConst.MIN_FLOAT_VALUE  # 最小float值
    else:
        return float(value)
