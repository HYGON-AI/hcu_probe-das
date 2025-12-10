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

from msprobe.core.overflow_check.level import OverflowLevel
from msprobe.visualization.utils import GraphConst
from msprobe.core.common.log import logger


class BaseNode:
    def __init__(self, node_op, node_id, up_node=None):
        self.op = node_op
        self.id = node_id
        self.data = {}
        self.output_data = {}
        self.input_data = {}
        self.upnode = None
        self.add_upnode(up_node)
        self.subnodes = []
        self.matched_node_link = []
        self.suggestions = {}
        self.stack_info = []
        self.micro_step_id = None
        self.overflow_level = None
        self.matched_distributed = {}
        self.batch_p2p_info = []
        self.rank = 0
        self.parallel_merge_info = []

    def __str__(self):
        info = f'id:\t{self.id}'
        return info

    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False

    def set_input_output(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def set_overflow_level(self, level):
        if not level or not isinstance(level, OverflowLevel):
            return
        self.overflow_level = level
        self.data[GraphConst.OVERFLOW_LEVEL] = self.overflow_level.value

    def add_upnode(self, node):
        """
        绑定upnode，用于对两个节点进行上下级关联
        """
        if not node or node.id == self.id or self.upnode:
            return
        self.upnode = node
        node.subnodes.append(self)

    def add_link(self, node, ancestors):
        """
        在节点匹配成功后进行匹配数据的录入
        Args:
            node: 和self相互匹配的节点
            ancestors: 对面节点的祖先信息
        """
        self.matched_node_link = ancestors
        node.matched_node_link = ancestors

    def get_ancestors(self):
        """
        获取节点所有祖先的列表
        """
        ancestors = []
        current_node = self.upnode
        seen_nodes = set()
        while current_node:
            if current_node.id in seen_nodes:
                logger.warning(f'Detected a cycle in the node structure and cannot get node ancestors, '
                               f'current node is {current_node.id}.')
                return []
            seen_nodes.add(current_node.id)
            ancestors.append(current_node.id)
            current_node = current_node.upnode
        return list(reversed(ancestors))
