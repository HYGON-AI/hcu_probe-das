# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

from enum import Enum, auto


# 统计值类型枚举
class ValType(Enum):
    NUM = auto()
    NAN = auto()
    INF = auto()
    NEG_INF = auto()
    DEVICE = auto()
    NA = auto()
    OTHER = auto()


ALL_TYPES = [
    ValType.NUM,
    ValType.NAN,
    ValType.INF,
    ValType.NEG_INF,
    ValType.DEVICE,
    ValType.NA,
    ValType.OTHER,
]
