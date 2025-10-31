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

from dataclasses import dataclass
from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.core.common.const import CompareConst
from msprobe.core.compare.find_first.utils import RankPath, DiffAnalyseConst


@dataclass
class DataNode:
    op_name: str
    rank: int
    inputs: dict
    outputs: dict
    op_data: list
    layer: int = 0  # 和communication_node的layer保持一致
    sub_layer: int = 0  # 调用顺序，越小表示越先调用

    def __init__(self, op_name, rank, op_data, **kwargs):
        self.op_name = op_name
        self.rank = rank
        self.stack = None
        self.inputs = {}
        self.outputs = {}
        self.is_diff = False
        self.parse_data(op_data)
        self.sub_layer = kwargs.get('sub_layer', 0)

    def find_stack(self):
        for item in self.stack:
            if len(item) >= 2 and self.op_name in item[0]:
                return item[1]
        return {}

    def parse_data(self, op_data):
        self.is_diff = not op_data.get("is_same", True)
        self.op_data = op_data.get("op_items") # 这里拿到的是比对column，是一个list，有若干行
        metrics = {}
        for cmp_data in self.op_data:
            name = cmp_data.get(CompareConst.NPU_NAME)
            # 构建度量指标字典
            metrics = {}
            
            if CompareConst.NPU_MAX in cmp_data:
                metrics = {CompareConst.NPU_MAX: cmp_data.get(CompareConst.NPU_MAX),
                        CompareConst.NPU_MIN: cmp_data.get(CompareConst.NPU_MIN),
                        CompareConst.NPU_MEAN: cmp_data.get(CompareConst.NPU_MEAN),
                        CompareConst.NPU_NORM: cmp_data.get(CompareConst.NPU_NORM)}
            elif CompareConst.NPU_MD5 in cmp_data:
                metrics[CompareConst.NPU_MD5] = cmp_data.get(CompareConst.NPU_MD5)
                
            if CompareConst.NPU_P2POP_PEER in cmp_data:
                metrics[CompareConst.NPU_P2POP_PEER] = cmp_data.get(CompareConst.NPU_P2POP_PEER)

            if cmp_data.get(CompareConst.STACK) != CompareConst.N_A and not self.stack:
                self.stack = cmp_data.get(CompareConst.STACK)
            if cmp_data.get('state') == "input":
                self.inputs[name] = metrics
            elif cmp_data.get('state') == "output":
                self.outputs[name] = metrics

    def gen_node_info(self, path: RankPath):
        data_info_list = {Const.INPUT: self.inputs, Const.OUTPUT: self.outputs}
        return {'op_name': self.op_name,
                'data_info': data_info_list,
                'stack_info': self.stack}


class CommunicationNode:
    def __init__(self, node_id, rank, data: DataNode, layer=0, **kwargs):
        self.node_id = node_id
        self.rank = rank
        self.data = data
        self.is_diff = data.is_diff
        self.layer = layer
        op_name_split = self.data.op_name.split(Const.SEP)
        if len(op_name_split) < 4:
            logger.error(f'invalid op_name: {self.data.op_name}')
            raise RuntimeError(f'invalid op_name: {self.data.op_name}')
        self.api = op_name_split[1]
        self.call_cnt = op_name_split[2]
        self.pre_node = kwargs.get('pre_node')
        self.link_nodes = kwargs.get('link_nodes', {})
        self.dst_nodes = kwargs.get('dst_nodes', {})
        self.src_nodes = kwargs.get('src_nodes', {})
        self.next_nodes = kwargs.get('next_nodes', {})
        self.compute_ops = kwargs.get('compute_ops', [])
        self.type = self._resolve_type()
        self.connected = False

    def add_next(self, node):
        self.next_nodes[node.node_id] = node
        node.pre_node = self
        node.layer = self.layer + 1
        node.data.layer = node.layer

    def add_link(self, node):
        self.link_nodes[node.node_id] = node
        node.link_nodes[self.node_id] = self
        node.layer = self.layer
        node.data.layer = node.layer
        self.connected = True
        node.connected = True

    def add_dst(self, node):
        self.dst_nodes[node.node_id] = node
        node.src_nodes[self.node_id] = self
        node.layer = self.layer
        node.data.layer = node.layer
        self.connected = True
        node.connected = True

    def delete(self):
        for node in self.next_nodes.values():
            node.pre_node = None
        for node in self.dst_nodes.values():
            if node.src_nodes:
                node.src_nodes.pop(self.node_id)
        for node in self.src_nodes.values():
            if node.dst_nodes:
                node.dst_nodes.pop(self.node_id)
        for node in self.link_nodes.values():
            if node.link_nodes:
                node.link_nodes.pop(self.node_id)
        if self.pre_node:
            if self.pre_node.next_nodes:
                self.pre_node.next_nodes.pop(self.node_id)

    def find_connected_nodes(self):
        """
        根据 api/类型/入参/调用次数 确定相连接的node的op_name
        """
        tar_api = DiffAnalyseConst.P2P_API_MAPPING.get(self.api, self.api)
        ranks = set()
        # 遍历DST和SRC相关的input，获取对应的rank值
        # 遍历inputs获取所有rank值
        for input_name, v in self.data.inputs.items():
            # 检查key是否包含DST/SRC相关标识
            target_types = [DiffAnalyseConst.DST, DiffAnalyseConst.DST_GROUP,
                          DiffAnalyseConst.SRC, DiffAnalyseConst.SRC_GROUP]
            if any(keyword in input_name for keyword in target_types):
                # 优先使用MD5值，如果没有则使用NPU_MAX值
                rank_val = 0
                if CompareConst.NPU_MD5 in v:
                    rank_val = int(v.get(CompareConst.NPU_MD5, 0))
                else:
                    rank_val = int(v.get(CompareConst.NPU_MAX, 0))
                if rank_val:
                    ranks.add(rank_val)
            elif input_name.endswith('.group'):
                # 优先使用MD5值，如果没有则使用NPU_MAX值
                val = v.get(CompareConst.NPU_MD5) if CompareConst.NPU_MD5 in v else v.get(CompareConst.NPU_MAX)
                if val and val.startswith('[') and val.endswith(']'):
                    val = [int(part) for part in val.strip('[]').split(',')]
                    ranks.update(val)
            elif v.get(CompareConst.NPU_P2POP_PEER) != "None":
                ranks.add(v.get(CompareConst.NPU_P2POP_PEER))

        return {'ranks': ranks, 'api': f'Distributed.{tar_api}',
                'type': DiffAnalyseConst.OPPOSITE_DIR.get(self.type, DiffAnalyseConst.LINK)}


    def _resolve_type(self):
        # 遍历SRC和DST相关的输入，根据rank值判断节点类型
        for prefix, node_type in [(DiffAnalyseConst.SRC, DiffAnalyseConst.SRC), 
                                (DiffAnalyseConst.DST, DiffAnalyseConst.DST)]:
            for k, v in self.data.inputs.items():
                if prefix in k or f"{prefix}_GROUP" in k:
                    # 优先使用MD5值，如果没有则使用NPU_MAX值
                    compare_val = v.get(CompareConst.NPU_MD5) if CompareConst.NPU_MD5 in v \
                                  else v.get(CompareConst.NPU_MAX)
                    return node_type if compare_val == self.rank \
                           else DiffAnalyseConst.OPPOSITE_DIR[node_type]
        return DiffAnalyseConst.LINK
