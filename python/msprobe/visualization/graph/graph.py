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
from msprobe.core.overflow_check.checker import AnomalyDetector
from msprobe.visualization.graph.base_node import BaseNode
from msprobe.visualization.graph.node_op import NodeOp
from msprobe.visualization.utils import GraphConst
from msprobe.core.common.log import logger
from msprobe.core.common.const import Const
from msprobe.core.common.decorator import recursion_depth_decorator


class Graph:
    def __init__(self, model_name, data_path='', dump_data=None, micro_step_num=None):
        self.node_map = {}
        self.node_id_map = {}
        self.add_node(NodeOp.module, model_name)
        self.root = self.get_node(model_name)
        self.data_path = data_path
        self.dump_data = dump_data
        self.data_source = GraphConst.JSON_NPU_KEY
        self.step = 0
        self.rank = 0
        self.compare_mode = GraphConst.SUMMARY_COMPARE
        self.micro_step_num = micro_step_num

    def __str__(self):
        infos = [f'{str(self.node_map.get(node_id))}' for node_id in self.node_map]
        info = "\n".join(infos)
        return info

    @staticmethod
    def match(graph_n, node_n, graph_b):
        """
        给定节点n，在另一个graph中匹配它对应的节点。前置条件是它的父节点匹配已经完成
        目前采用完全匹配的方式，后续可能在这里加入一定的模糊匹配逻辑
        返回匹配结果，匹配到的节点，以及祖先树。没匹配到则返回None, []
        """
        if not node_n or node_n.id not in graph_b.node_map:
            return None, []
        node_b = graph_b.node_map.get(node_n.id)
        if node_n != node_b:
            return None, []
        ancestors_n = node_n.get_ancestors()
        ancestors_b = node_b.get_ancestors()
        if ancestors_n != ancestors_b:
            return None, []
        return node_b, ancestors_n

    @staticmethod
    def mapping_match(node_n, graph_b, mapping_dict):
        """
        根据映射配置对节点进行匹配
        """
        node_b = graph_b.node_map.get(mapping_dict.get(node_n.id, node_n.id))
        if not node_b:
            return None, [], []
        ancestors_n = node_n.get_ancestors()
        ancestors_b = node_b.get_ancestors()
        return node_b, ancestors_n, ancestors_b

    @staticmethod
    def fuzzy_match(node_n, node_b, check_shape=True):
        if not node_n or not node_b:
            return None, [], []
        if check_shape and not node_n.fuzzy_eq(node_b):
            return None, [], []
        ancestors_n = node_n.get_ancestors()
        ancestors_b = node_b.get_ancestors()
        return node_b, ancestors_n, ancestors_b

    @staticmethod
    def split_nodes_by_micro_step(nodes):
        """
        根据Module名称, 区分一个step中的多个micro steps.
        一个micro step必须是一次完整的前反向过程
        Example::
            =============== micro step0
            Module.forward
            Module.forward
            ...
            Module.backward
            Module.backward
            =============== micro step1
            Module.forward
            Module.forward
            ...
            Module.backward
            Module.backward
            =============== micro step2
            Module.forward
            Module.forward
            ...
            Module.backward
            Module.backward

        如果是非Module节点，分类到前一个Module节点所在的micro step.
        """
        result = {}
        micro_step = 0
        result[micro_step] = []
        backward_flag = False

        for node in nodes:
            if node.op == NodeOp.module:
                if f'{Const.SEP}{Const.FORWARD}{Const.SEP}' in node.id:
                    if backward_flag:
                        micro_step += 1
                        result[micro_step] = []
                        backward_flag = False
                else:
                    backward_flag = True
            result[micro_step].append(node)
        return result

    def get_sorted_nodes(self):
        """
        通过深度优先遍历graph，获得排过序的node列表
        """
        visited = set()
        order = []

        @recursion_depth_decorator('msprobe.visualization.graph.graph.Graph.get_nodes_order.visit', max_depth=500)
        def visit(node):
            if node.id in visited:
                return
            visited.add(node.id)
            for sub_node in node.subnodes:
                visit(sub_node)
            order.append(node)

        visit(self.root)
        return order

    def add_node(self, node_op, node_id, up_node=None, id_accumulation=False):
        """
        在graph中进行节点的添加
        Args:
            node_op: 需要添加的节点类型
            node_id: 需要添加的节点id
            up_node：对应节点的父节点
            id_accumulation: 是否对传入的重复node_id进行累加
        """
        if node_id in self.node_map:
            if id_accumulation:
                self.node_id_map[node_id] = 0
            else:
                return node_id
        if id_accumulation:
            if node_id in self.node_id_map:
                self.node_id_map[node_id] += 1
            else:
                self.node_id_map[node_id] = 0
            node_id = f'{node_id}.{self.node_id_map[node_id]}'
        node = BaseNode(node_op, node_id, up_node)
        self.node_map[node_id] = node
        return node_id

    def get_node(self, node_id):
        """
        返回节点，不存在返回None
        """
        return self.node_map.get(node_id, None)

    def paging_by_micro_step(self, graph_other=None):
        """
        给graph首层节点增加micro step标记，供前端分页展示，有助于在处理大规模图数据时进行优化和管理
        比对场景中，同步更新另一个图graph_other中相应节点的micro step信息
        Args:
            self: 当前graph
            graph_other: 可选参数，另一个graph
        Returns: 分批的数量
        """

        @recursion_depth_decorator(
            'msprobe.visualization.graph.graph.Graph.paging_by_micro_step.propagate_micro_step_id', max_depth=500)
        def propagate_micro_step_id(node):
            if node.upnode is not None and node.micro_step_id is None:
                node.micro_step_id = node.upnode.micro_step_id
            for sub_node in node.subnodes:
                propagate_micro_step_id(sub_node)

        if self.micro_step_num is not None:
            return self.micro_step_num + 1

        batches_n = Graph.split_nodes_by_micro_step(self.root.subnodes)
        for batch_number, nodes in batches_n.items():
            for node in nodes:
                node.micro_step_id = batch_number
                # 在graph_other中更新已匹配节点的micro_step_id
                if graph_other and node.matched_node_link:
                    node_other = graph_other.get_node(node.matched_node_link[-1])
                    if node_other:
                        node_other.micro_step_id = batch_number
        propagate_micro_step_id(self.root)
        # 遍历graph_other根节点下的所有子节点，确保未匹配节点也有micro_step_id
        if graph_other:
            for node in graph_other.root.subnodes:
                if node.micro_step_id is None:
                    try:
                        micro_step_id = int(node.id.split(Const.SEP)[-1])
                    except ValueError:
                        micro_step_id = 0
                    node.micro_step_id = micro_step_id
            propagate_micro_step_id(graph_other.root)
        return len(batches_n)

    def overflow_check(self):
        detector = AnomalyDetector(self.dump_data)
        detector.analyze().filter()

        for node_id, _node in self.node_map.items():
            if detector.has_overflow(node_id):
                lv = detector.get_overflow_level(node_id)
                _node.set_overflow_level(lv)
