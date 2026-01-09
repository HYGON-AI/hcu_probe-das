# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

import math
import json
from msprobe.core.common.const import CompareConst, Const
from msprobe.visualization.utils import ToolTip, GraphConst, str2float


class ModeAdapter:
    def __init__(self, compare_mode, parallel_merge=False):
        self.compare_mode = compare_mode
        self.csv_data = []
        self.compare_nodes = []
        self.parallel_merge = parallel_merge

    @staticmethod
    def _is_invalid(value):
        if not isinstance(value, float):
            return False
        return math.isnan(value) or math.isinf(value)

    @staticmethod
    def _add_compare_data(node_data, compare_data_dict, key_list, headers):
        for key, data_info in node_data.items():
            if not isinstance(data_info, dict):
                continue
            compare_data = compare_data_dict.get(key)
            if compare_data:
                # 对应比对结果csv的列
                id_list = [headers.index(x) for x in key_list]
                ModeAdapter._match_data(data_info, compare_data, key_list, id_list)

    @staticmethod
    def _match_data(data_dict, compare_data, key_list, id_list):
        """
        绑定精度指标到node的input_data和output_data
        """
        if len(key_list) != len(id_list):
            return
        for id_val, key in zip(id_list, key_list):
            data_dict[key] = compare_data[id_val]

    @staticmethod
    def _check_list_len(data_list, len_num):
        if len(data_list) < len_num:
            raise ValueError(f"compare_data_dict_list must contain at least {len_num} items.")

    def parse_result(self, node, compare_data_dict, api_indicator):
        """
        根据结果返回precision_index
        """
        if self.compare_mode == GraphConst.MD5_COMPARE:
            key_list = GraphConst.MD5_INDEX_LIST
            headers = CompareConst.MD5_COMPARE_RESULT_HEADER
        elif self.compare_mode == GraphConst.SUMMARY_COMPARE:
            key_list = GraphConst.SUMMARY_INDEX_LIST
            headers = CompareConst.SUMMARY_COMPARE_RESULT_HEADER
        else:
            key_list = GraphConst.REAL_DATA_INDEX_LIST
            headers = CompareConst.COMPARE_RESULT_HEADER
        ModeAdapter._add_compare_data(node.input_data, compare_data_dict, key_list, headers)
        ModeAdapter._add_compare_data(node.output_data, compare_data_dict, key_list, headers)
        precision_index = GraphConst.COMPARE_INDICATOR_TO_PRECISION_INDEX_MAPPING.get(api_indicator,
                                                                                      CompareConst.PASS)
        precision_index = self._ignore_precision_index(node.id, precision_index)
        return precision_index

    def prepare_real_data(self, node):
        """
        为真实数据比较模式准备节点信息
        """
        if self.compare_mode == GraphConst.REAL_DATA_COMPARE:
            self.compare_nodes.append(node)
            return True
        return False

    def add_csv_data(self, compare_result_list):
        if self.compare_mode != GraphConst.REAL_DATA_COMPARE:
            return
        self.csv_data.extend(compare_result_list)

    def get_tool_tip(self):
        """
        用于前端展示字段的具体含义
        """
        if self.compare_mode == GraphConst.SUMMARY_COMPARE:
            tips = {
                CompareConst.MAX_DIFF: ToolTip.MAX_DIFF,
                CompareConst.MIN_DIFF: ToolTip.MIN_DIFF,
                CompareConst.MEAN_DIFF: ToolTip.MEAN_DIFF,
                CompareConst.NORM_DIFF: ToolTip.NORM_DIFF}
        elif self.compare_mode == GraphConst.MD5_COMPARE:
            tips = {Const.MD5: ToolTip.MD5}
        else:
            tips = {
                CompareConst.ONE_THOUSANDTH_ERR_RATIO: ToolTip.ONE_THOUSANDTH_ERR_RATIO,
                CompareConst.FIVE_THOUSANDTHS_ERR_RATIO: ToolTip.FIVE_THOUSANDTHS_ERR_RATIO,
                CompareConst.COSINE: ToolTip.COSINE,
                CompareConst.MAX_ABS_ERR: ToolTip.MAX_ABS_ERR,
                CompareConst.MAX_RELATIVE_ERR: ToolTip.MAX_RELATIVE_ERR}
        return json.dumps(tips)

    def _ignore_precision_index(self, node_id, precision_index):
        node_id_split = node_id.split(Const.SEP)
        if len(node_id_split) < 2:
            return precision_index
        if node_id.split(Const.SEP)[1] in GraphConst.IGNORE_PRECISION_INDEX:
            return GraphConst.MAX_INDEX_KEY if self.compare_mode == GraphConst.MD5_COMPARE else GraphConst.MIN_INDEX_KEY
        return precision_index
