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

from typing import Any

CHECK_FIELDS = ['Max', 'Min', 'Mean']
OVERFLOW_VALUES = ['inf', '-inf', 'nan']


def has_nan_inf(value: Any) -> bool:
    """检查值是否包含NaN或Inf"""
    if isinstance(value, dict):
        for k, v in value.items():
            if k in CHECK_FIELDS and str(v).lower() in OVERFLOW_VALUES:
                return True
    return False
