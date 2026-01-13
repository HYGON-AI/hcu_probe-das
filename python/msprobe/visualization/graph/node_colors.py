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

from enum import Enum
from msprobe.visualization.utils import GraphConst, ToolTip
from msprobe.core.common.const import CompareConst

SUMMARY_DESCRIPTION = "此节点所有输入输出的统计量误差, 颜色越深代表调试侧与标杆侧的偏差越大"
REAL_DATA_DESCRIPTION = "此节点所有输入输出的 tensor 误差, 颜色越深代表调试侧与标杆侧的偏差越大"
MD5_DESCRIPTION_N = "与标杆相比, 此节点任意输入输出的md5值不同"
MD5_DESCRIPTION_Y = "与标杆相比, 此节点所有输入输出的md5值相同"
NOT_MATCHED = "比对过程中节点未匹配上"


class NodeColors(Enum):
    # 枚举值后缀数字越小, 颜色越浅
    # value值左闭右开, 两个值相同代表固定值
    YELLOW_1 = ("#FFFCF3", {
        GraphConst.SUMMARY_COMPARE: {GraphConst.VALUE: [0, 0.3], GraphConst.DESCRIPTION: SUMMARY_DESCRIPTION,
                                     GraphConst.ACCURACY_LEVEL: CompareConst.PASS},
        GraphConst.REAL_DATA_COMPARE: {GraphConst.VALUE: [0, 0.3], GraphConst.DESCRIPTION: REAL_DATA_DESCRIPTION,
                                       GraphConst.ACCURACY_LEVEL: CompareConst.PASS},
        GraphConst.MD5_COMPARE: {GraphConst.VALUE: [0, 0.3], GraphConst.DESCRIPTION: MD5_DESCRIPTION_Y,
                                 GraphConst.ACCURACY_LEVEL: CompareConst.PASS},
    })
    ORANGE_1 = ("#FFDC7F", {
        GraphConst.SUMMARY_COMPARE: {GraphConst.VALUE: [0.3, 0.6], GraphConst.DESCRIPTION: SUMMARY_DESCRIPTION,
                                     GraphConst.ACCURACY_LEVEL: CompareConst.WARNING},
        GraphConst.REAL_DATA_COMPARE: {GraphConst.VALUE: [0.3, 0.6], GraphConst.DESCRIPTION: REAL_DATA_DESCRIPTION,
                                       GraphConst.ACCURACY_LEVEL: CompareConst.WARNING},
        GraphConst.MD5_COMPARE: {GraphConst.VALUE: [0.3, 0.6], GraphConst.DESCRIPTION: MD5_DESCRIPTION_N,
                                 GraphConst.ACCURACY_LEVEL: CompareConst.WARNING},
    })
    RED = ("#FF704D", {
        GraphConst.SUMMARY_COMPARE: {GraphConst.VALUE: [0.6, 1], GraphConst.DESCRIPTION: SUMMARY_DESCRIPTION,
                                     GraphConst.ACCURACY_LEVEL: CompareConst.ERROR},
        GraphConst.REAL_DATA_COMPARE: {GraphConst.VALUE: [0.6, 1], GraphConst.DESCRIPTION: REAL_DATA_DESCRIPTION,
                                       GraphConst.ACCURACY_LEVEL: CompareConst.ERROR},
        GraphConst.MD5_COMPARE: {GraphConst.VALUE: [0.6, 1], GraphConst.DESCRIPTION: MD5_DESCRIPTION_N,
                                 GraphConst.ACCURACY_LEVEL: CompareConst.ERROR},
    })
    GREY = ("#C7C7C7", {
        GraphConst.VALUE: [], GraphConst.DESCRIPTION: NOT_MATCHED, GraphConst.ACCURACY_LEVEL: GraphConst.UNMATCHED
    })

    def __init__(self, hex_value, mode_info):
        self.hex_value = hex_value
        self.mode_info = mode_info

    @staticmethod
    def get_node_colors(mode):
        """
        获取不同比对模式下的颜色说明
        Args:
            mode: 比对模式
        Returns: 颜色说明
        """
        return {
            color.hex_value: color.get_info_by_mode(mode) for color in NodeColors if color.get_info_by_mode(mode)
        }

    @staticmethod
    def get_node_error_status(mode, value):
        """
        判断精度数据比对指标是否大于基准值
        Args:
            mode: 比对模式
            value: 精度数据比对指标
        Returns: bool
        """
        info = NodeColors.ORANGE_1.get_info_by_mode(mode)
        if info and GraphConst.VALUE in info:
            value_range = info[GraphConst.VALUE]
            return value > value_range[0]
        return False

    def get_info_by_mode(self, mode):
        if isinstance(self.mode_info, dict):
            # 检查是否是模式特定的信息
            if isinstance(next(iter(self.mode_info.values())), dict):
                return self.mode_info.get(mode, {})
            else:
                # 所有模式共享相同的信息
                return self.mode_info
        return {}
