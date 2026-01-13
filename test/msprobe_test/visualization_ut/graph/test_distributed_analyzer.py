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

import unittest
from unittest.mock import patch
from msprobe.visualization.graph.distributed_analyzer import (
    CommunicationType, DistributedType, CANNOT_MATCH, DistributedAnalyzer
)
from msprobe.visualization.utils import GraphConst, Const
from msprobe.core.common.log import logger
from msprobe.visualization.graph.graph import Graph, BaseNode, NodeOp


class TestDistributedAnalyzer(unittest.TestCase):
    @staticmethod
    def _create_base_node(node_id, up_node=None, op=NodeOp.module):
        """创建BaseNode实例的工厂方法"""
        node = BaseNode(op, node_id, up_node)
        node.data = {
            GraphConst.JSON_INDEX_KEY: 100,
            GraphConst.OVERFLOW_LEVEL: 0.5
        }
        node.matched_distributed = None
        return node

    @staticmethod
    def _create_graph(rank, nodes=None):
        """创建Graph实例的工厂方法"""
        graph = Graph(model_name=f"model_rank_{rank}")
        graph.rank = rank
        if nodes:
            for node_id, node in nodes.items():
                graph.node_map[node_id] = node
        return graph

    def setUp(self):
        """初始化测试数据，每个测试方法执行前都会调用"""
        # ------------------------ 基础工厂方法 ------------------------
        self.base_node_factory = lambda node_id, up_node=None, op=NodeOp.module: self._create_base_node(node_id,
                                                                                                        up_node, op)
        self.graph_factory = lambda rank, nodes=None: self._create_graph(rank, nodes)

        # ------------------------ 构建测试用图 ------------------------
        self.distributed_info = self._build_test_distributed_info()

        # ------------------------ 创建分析器实例 ------------------------
        self.analyzer = DistributedAnalyzer(self.distributed_info, overflow_check=False)
        self.analyzer_overflow = DistributedAnalyzer(self.distributed_info, overflow_check=True)

    def _build_test_distributed_info(self):
        """构建多rank的测试图字典"""
        # ------------------------ Rank 0 节点 ------------------------
        # 1. P2P节点：isend
        isend_node = self.base_node_factory("Distributed.isend.0.forward")
        isend_node.input_data = {
            f"{isend_node.id}{GraphConst.INPUT}dst": {"value": 1}
        }
        isend_node.output_data = {
            f"{isend_node.id}.output.0": {
                Const.DTYPE: "float32",
                Const.SHAPE: [1, 2],
                Const.MAX: 1.0,
                Const.MIN: 0.0,
                Const.MEAN: 0.5,
                Const.NORM: 1.0
            }
        }

        # 2. P2P节点：recv
        recv_node = self.base_node_factory("Distributed.recv.0.forward")
        recv_node.input_data = {
            f"{recv_node.id}{GraphConst.INPUT}src": {"value": 1}
        }
        recv_node.output_data = {
            f"{recv_node.id}.output.0": {
                Const.DTYPE: "float32",
                Const.SHAPE: [1, 2],
                Const.MAX: 1.0,
                Const.MIN: 0.0,
                Const.MEAN: 0.5,
                Const.NORM: 1.0
            }
        }

        # 3. 集体通信节点：broadcast
        broadcast_node = self.base_node_factory("Distributed.broadcast.0.forward")
        broadcast_node.input_data = {
            f"{broadcast_node.id}{GraphConst.INPUT}1": {"value": 0},
            f"{broadcast_node.id}{GraphConst.INPUT}group": {
                "group_ranks": [0, 1],
                "group_id": "group_0"
            }
        }

        # 4. 批量P2P节点：batch_p2p
        batch_p2p_node = self.base_node_factory("Distributed.batch_p2p.0.forward")
        batch_p2p_node.batch_p2p_info = [
            {GraphConst.OP: "isend", GraphConst.PEER: 1, GraphConst.GROUP_ID: "g1"},
            {GraphConst.OP: "recv", GraphConst.PEER: 2, GraphConst.GROUP_ID: "g2"}
        ]

        # 5. 非通信节点
        normal_node = self.base_node_factory("Module.forward.0", op=NodeOp.module)

        # ------------------------ Rank 1 节点 ------------------------
        # 1. P2P节点：irecv
        irecv_node = self.base_node_factory("Distributed.irecv.0.forward")
        irecv_node.input_data = {
            f"{irecv_node.id}{GraphConst.INPUT}src": {"value": 0}
        }
        irecv_node.output_data = {
            f"{irecv_node.id}.output.0": {
                Const.DTYPE: "float32",
                Const.SHAPE: [1, 2],
                Const.MAX: 1.0,
                Const.MIN: 0.0,
                Const.MEAN: 0.5,
                Const.NORM: 1.0
            }
        }

        # 2. P2P节点：send
        send_node = self.base_node_factory("Distributed.send.0.forward")
        send_node.input_data = {
            f"{send_node.id}{GraphConst.INPUT}dst": {"value": 0}
        }

        # 3. 集体通信节点：broadcast
        broadcast_node_1 = self.base_node_factory("Distributed.broadcast.0.forward")
        broadcast_node_1.input_data = {
            f"{broadcast_node_1.id}{GraphConst.INPUT}1": {"value": 0},
            f"{broadcast_node_1.id}{GraphConst.INPUT}group": {
                "group_ranks": [0, 1],
                "group_id": "group_0"
            }
        }

        return {
            "0": {
                isend_node.id: isend_node,
                recv_node.id: recv_node,
                broadcast_node.id: broadcast_node,
                batch_p2p_node.id: batch_p2p_node,
                normal_node.id: normal_node
            },
            "1": {
                irecv_node.id: irecv_node,
                send_node.id: send_node,
                broadcast_node_1.id: broadcast_node_1
            }
        }

    # ------------------------ 测试枚举类 ------------------------
    def test_enum_values(self):
        """验证枚举值正确性"""
        self.assertEqual(CommunicationType.SEND.value, "send")
        self.assertEqual(CommunicationType.RECEIVE.value, "receive")
        self.assertEqual(CommunicationType.SEND_RECEIVE.value, "send_receive")
        self.assertEqual(DistributedType.P2P.value, "p2p")
        self.assertEqual(DistributedType.COLLECTIVE.value, "collective")

    # ------------------------ 测试静态方法 ------------------------
    def test_get_opposite_communication_type(self):
        """测试通信类型反转逻辑"""
        self.assertEqual(DistributedAnalyzer._get_opposite_communication_type("send"), "receive")
        self.assertEqual(DistributedAnalyzer._get_opposite_communication_type("receive"), "send")
        self.assertEqual(DistributedAnalyzer._get_opposite_communication_type("send_receive"), "send_receive")
        self.assertEqual(DistributedAnalyzer._get_opposite_communication_type("unknown"), "unknown")

    def test__node_output_all_equal(self):
        """测试节点输出数据对比"""
        # 全相等场景
        data1 = {
            Const.DTYPE: "float32",
            Const.SHAPE: [1, 2],
            Const.MAX: 1.0,
            Const.MIN: 0.0,
            Const.MEAN: 0.5,
            Const.NORM: 1.0
        }
        data2 = data1.copy()
        self.assertTrue(DistributedAnalyzer._node_output_all_equal(data1, data2))

        # 部分字段不等
        data3 = data1.copy()
        data3[Const.MAX] = 2.0
        self.assertFalse(DistributedAnalyzer._node_output_all_equal(data1, data3))

        # 字段缺失
        self.assertFalse(DistributedAnalyzer._node_output_all_equal(data1, {}))
        self.assertFalse(DistributedAnalyzer._node_output_all_equal({}, data2))

    def test_get_target_rank(self):
        """测试获取目标rank"""
        node = self.base_node_factory("test_node")
        node_id = node.id

        # 正常获取目标rank
        node.input_data = {f"{node_id}{GraphConst.INPUT}dst": {"value": 1}}
        self.assertEqual(DistributedAnalyzer._get_target_rank(node, 0, "dst"), 1)

        # 参数不存在（日志输出验证）
        node.input_data = {}
        with patch.object(logger, "debug") as mock_log:
            self.assertIsNone(DistributedAnalyzer._get_target_rank(node, 0, "dst"))
            mock_log.assert_called_with(
                f'The parameter dst of node {node_id} does not exist, {CANNOT_MATCH}0'
            )

    def test_get_group_info(self):
        """测试获取group信息"""
        node = self.base_node_factory("test_node")
        node_id = node.id

        # 正常获取group信息
        node.input_data = {
            f"{node_id}{GraphConst.INPUT}group": {
                "group_ranks": [0, 1],
                "group_id": "g1"
            }
        }
        self.assertEqual(DistributedAnalyzer._get_group_info(node, 0), ([0, 1], "g1"))

        # group参数不存在
        node.input_data = {}
        with patch.object(logger, "debug") as mock_log:
            self.assertEqual(DistributedAnalyzer._get_group_info(node, 0), (None, None))
            mock_log.assert_called_with(
                f'The kwarg group of node {node_id} does not exist, {CANNOT_MATCH}0'
            )

        # group_ranks不存在
        node.input_data = {
            f"{node_id}{GraphConst.INPUT}group": {"group_id": "g1"}
        }
        with patch.object(logger, "debug") as mock_log:
            self.assertEqual(DistributedAnalyzer._get_group_info(node, 0), (None, None))
            mock_log.assert_called_with(
                f'The group_ranks of node {node_id} does not exist, {CANNOT_MATCH}0'
            )

        # group_id不存在
        node.input_data = {
            f"{node_id}{GraphConst.INPUT}group": {"group_ranks": [0, 1]}
        }
        with patch.object(logger, "debug") as mock_log:
            self.assertEqual(DistributedAnalyzer._get_group_info(node, 0), (None, None))
            mock_log.assert_called_with(
                f'The group_id of node {node_id} does not exist, {CANNOT_MATCH}0'
            )

    # ------------------------ 测试初始化和映射构建 ------------------------
    def test_init(self):
        """测试初始化逻辑"""
        # 基础初始化
        self.assertEqual(self.analyzer.overflow_check, False)
        self.assertIn("send", self.analyzer.config)
        self.assertIn("broadcast", self.analyzer.config)
        self.assertNotEqual(self.analyzer.group_node_mapping, {})

        # 开启overflow_check
        self.assertEqual(self.analyzer_overflow.overflow_check, True)

    def test_make_group_node_mapping(self):
        """测试构建group节点映射"""
        # 清空原有映射
        self.analyzer.group_node_mapping = {}
        self.analyzer._make_group_node_mapping()

        # 验证rank0的映射
        rank0_mapping = self.analyzer.group_node_mapping.get("0")
        self.assertIsNotNone(rank0_mapping)

        # 验证P2P节点映射（isend_rank1_1）
        isend_node_id = "Distributed.isend.0.forward"
        unique_group_id = rank0_mapping.get(isend_node_id)
        self.assertIsNotNone(unique_group_id)
        self.assertIn("isendrank1", unique_group_id)
        self.assertEqual(rank0_mapping.get(unique_group_id), isend_node_id)

        # 验证集体通信节点映射（group_0broadcast_1）
        broadcast_node_id = "Distributed.broadcast.0.forward"
        broadcast_unique_id = rank0_mapping.get(broadcast_node_id)
        self.assertIsNotNone(broadcast_unique_id)
        self.assertIn("group_0broadcast", broadcast_unique_id)

    def test_make_batch_p2p_mapping(self):
        """测试批量P2P映射构建"""
        batch_p2p_node = self.distributed_info["0"]["Distributed.batch_p2p.0.forward"]
        batch_p2p_count = {}
        self.analyzer.group_node_mapping = {0: {}}

        # 正常构建映射
        self.analyzer._make_batch_p2p_mapping(batch_p2p_node, 0, batch_p2p_count)
        rank0_mapping = self.analyzer.group_node_mapping[0]

        # 验证生成的unique_group_id
        self.assertIn("isend_rank1_g1_1", rank0_mapping)
        self.assertIn("recv_rank2_g2_1", rank0_mapping)
        self.assertEqual(rank0_mapping[batch_p2p_node.id], ["isend_rank1_g1_1", "recv_rank2_g2_1"])
        self.assertEqual(rank0_mapping["isend_rank1_g1_1"], batch_p2p_node.id)

        # 参数缺失场景（op/peer为空）
        invalid_batch_node = self.base_node_factory("Distributed.batch_p2p.1.forward")
        invalid_batch_node.batch_p2p_info = [
            {GraphConst.OP: None, GraphConst.PEER: 1},
            {GraphConst.OP: "recv", GraphConst.PEER: None}
        ]
        with patch.object(logger, "debug") as mock_log:
            self.analyzer._make_batch_p2p_mapping(invalid_batch_node, 0, batch_p2p_count)
            mock_log.assert_called_with('Cannot get param op or peer.')

    def test_get_distributed_name_and_type(self):
        """测试解析分布式节点名称和类型"""
        # 正常P2P节点
        node_id = "Distributed.isend.0.forward"
        api_name, dist_type = self.analyzer._get_distributed_name_and_type(node_id)
        self.assertEqual(api_name, "isend")
        self.assertEqual(dist_type, DistributedType.P2P)

        # 正常集体通信节点
        node_id = "Distributed.broadcast.0.forward"
        api_name, dist_type = self.analyzer._get_distributed_name_and_type(node_id)
        self.assertEqual(api_name, "broadcast")
        self.assertEqual(dist_type, DistributedType.COLLECTIVE)

        # 未知API（默认集体通信）
        node_id = "Distributed.unknown_api.0.forward"
        api_name, dist_type = self.analyzer._get_distributed_name_and_type(node_id)
        self.assertEqual(api_name, "unknown_api")
        self.assertEqual(dist_type, DistributedType.COLLECTIVE)

        # 无效node_id（无分隔符）
        with self.assertRaises(ValueError) as excinfo:
            self.analyzer._get_distributed_name_and_type("invalid_node_id")
        self.assertIn("Invalid node id invalid_node_id.", str(excinfo.exception))

    # ------------------------ 测试核心匹配逻辑 ------------------------
    def test_get_target_node(self):
        """测试获取目标节点"""
        # 正常获取P2P目标节点（isend -> irecv）
        rank0_isend_unique_id = self.analyzer.group_node_mapping["0"]["Distributed.isend.0.forward"]
        target_node = self.analyzer._get_target_node(
            rank=0,
            unique_group_id=rank0_isend_unique_id,
            api_name="isend",
            target_rank=1,
            target_api_name="irecv"
        )
        self.assertIsNotNone(target_node)
        self.assertEqual(target_node.id, "Distributed.irecv.0.forward")

        # 目标rank不存在
        with patch.object(logger, "debug") as mock_log:
            target_node = self.analyzer._get_target_node(
                rank=0,
                unique_group_id="test",
                api_name="isend",
                target_rank=999,
                target_api_name="irecv"
            )
            self.assertIsNone(target_node)
            mock_log.assert_called_with(f'Node data does not exist, {CANNOT_MATCH}999')

        # 目标节点不存在
        with patch.object(logger, "debug") as mock_log:
            target_node = self.analyzer._get_target_node(
                rank=0,
                unique_group_id="invalid_id",
                api_name="isend",
                target_rank=1,
                target_api_name="irecv"
            )
            self.assertIsNone(target_node)
            mock_log.assert_called_with(f'Node  does not exist, {CANNOT_MATCH}1')

    def test_add_node_matched_distributed(self):
        """测试添加节点匹配信息"""
        # 创建测试节点
        source_node = self.base_node_factory("source_node")
        target_node = self.base_node_factory("target_node")
        target_node.data = {GraphConst.JSON_INDEX_KEY: 200}

        # 基础场景（不反转类型）
        self.analyzer._add_node_matched_distributed(source_node, target_node, "isend", 1)
        self.assertEqual(source_node.matched_distributed, {
            "communications_type": "send",
            "nodes_info": {1: ["200", "target_node"]}
        })

        # 反转通信类型
        self.analyzer._add_node_matched_distributed(source_node, target_node, "broadcast", 1, reversal_type=True)
        self.assertEqual(source_node.matched_distributed["communications_type"], "receive")

    def test_p2p_match(self):
        """测试P2P节点匹配"""
        isend_node = self.distributed_info["0"]["Distributed.isend.0.forward"]

        # 正常匹配场景
        self.analyzer._p2p_match(isend_node, 0, "isend")
        self.assertIsNotNone(isend_node.matched_distributed)
        self.assertIsNotNone(self.distributed_info["1"]["Distributed.irecv.0.forward"].matched_distributed)

        # 目标rank不存在
        isend_node.matched_distributed = {}
        isend_node.input_data[f"{isend_node.id}{GraphConst.INPUT}dst"]["value"] = 999
        self.analyzer._p2p_match(isend_node, 0, "isend")
        self.assertEqual(isend_node.matched_distributed, {})

        # 源rank不匹配
        isend_node.input_data[f"{isend_node.id}{GraphConst.INPUT}dst"]["value"] = 1
        irecv_node = self.distributed_info["1"]["Distributed.irecv.0.forward"]
        irecv_node.input_data[f"{irecv_node.id}{GraphConst.INPUT}src"]["value"] = 2
        with patch.object(logger, "debug") as mock_log:
            self.analyzer._p2p_match(isend_node, 0, "isend")
            self.assertIn("rank is inconsistent", mock_log.call_args[0][0])

    def test_collective_match(self):
        """测试集体通信节点匹配"""
        broadcast_node = self.distributed_info["0"]["Distributed.broadcast.0.forward"]

        # 正常匹配场景
        self.analyzer._collective_match(broadcast_node, 0, "broadcast")
        self.assertIsNotNone(broadcast_node.matched_distributed)
        self.assertIsNotNone(self.distributed_info["1"]["Distributed.broadcast.0.forward"].matched_distributed)

        # source_rank不匹配
        broadcast_node.matched_distributed = {}
        broadcast_node.input_data[f"{broadcast_node.id}{GraphConst.INPUT}1"]["value"] = 1
        self.analyzer._collective_match(broadcast_node, 0, "broadcast")
        self.assertEqual(broadcast_node.matched_distributed, {})

        # group信息缺失
        broadcast_node.input_data[f"{broadcast_node.id}{GraphConst.INPUT}1"]["value"] = 0
        broadcast_node.input_data.pop(f"{broadcast_node.id}{GraphConst.INPUT}group")
        self.analyzer._collective_match(broadcast_node, 0, "broadcast")
        self.assertEqual(broadcast_node.matched_distributed, {})

        # group_id不匹配
        broadcast_node.input_data[f"{broadcast_node.id}{GraphConst.INPUT}group"] = {
            "group_ranks": [0, 1],
            "group_id": "g2"
        }
        with patch.object(logger, "debug") as mock_log:
            self.analyzer._collective_match(broadcast_node, 0, "broadcast")
            self.assertIn("group id of the two nodes are different", mock_log.call_args[0][0])

    # ------------------------ 边界场景测试 ------------------------
    def test_edge_cases(self):
        """测试边界场景"""
        # 空图字典
        empty_graphs = {}
        analyzer = DistributedAnalyzer(empty_graphs, overflow_check=False)
        analyzer.distributed_match()  # 无异常

        # 单rank场景
        single_rank_node = self.base_node_factory("Distributed.broadcast.0.forward")
        single_rank_node.input_data = {
            f"{single_rank_node.id}{GraphConst.INPUT}group": {
                "group_ranks": [0],
                "group_id": "g1"
            }
        }
        analyzer = DistributedAnalyzer({0: {single_rank_node.id: single_rank_node}}, overflow_check=True)
        analyzer.distributed_match()  # 无异常

        # 空node_map
        analyzer = DistributedAnalyzer({0: {}}, overflow_check=False)
        analyzer.distributed_match()  # 无异常

        # 配置中不存在的API
        unknown_node = self.base_node_factory("Distributed.unknown_api.0.forward")
        analyzer = DistributedAnalyzer({0: {unknown_node.id: unknown_node}}, overflow_check=False)
        analyzer.distributed_match()  # 无异常
