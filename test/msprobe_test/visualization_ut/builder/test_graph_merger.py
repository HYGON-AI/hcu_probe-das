import unittest
from unittest.mock import patch, MagicMock, call
from msprobe.visualization.builder.graph_merger import (
    GraphMerger, BaseGraphMerger, PPMerger, TPMerger,
    NoParallelMerger, TPPPMerger, FullMerger
)
from msprobe.core.common.const import Const
from msprobe.visualization.utils import GraphConst, ParallelParam
from msprobe.visualization.graph.node_op import NodeOp
from msprobe.visualization.graph.graph import Graph
from msprobe.core.common.exceptions import MsprobeException


class TestGraphMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = MagicMock()
        self.parallel_param = ParallelParam(tp=1, pp=1, rank_size=1)
        self.is_bench = False

    def test_select_strategy_no_parallel(self):
        self.parallel_param.tp = self.parallel_param.pp = self.parallel_param.rank_size = 1
        merger = GraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        self.assertIsInstance(merger.strategy, NoParallelMerger)

    def test_select_strategy_tp(self):
        self.parallel_param.tp = self.parallel_param.rank_size = 2
        self.parallel_param.pp = 1
        merger = GraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        self.assertIsInstance(merger.strategy, TPMerger)

    def test_select_strategy_pp(self):
        self.parallel_param.pp = self.parallel_param.rank_size = 2
        self.parallel_param.tp = 1
        merger = GraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        self.assertIsInstance(merger.strategy, PPMerger)

    def test_select_strategy_tp_pp(self):
        self.parallel_param.tp = self.parallel_param.pp = 2
        self.parallel_param.rank_size = 4
        merger = GraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        self.assertIsInstance(merger.strategy, TPPPMerger)

    def test_select_strategy_full(self):
        self.parallel_param.tp = 2
        self.parallel_param.pp = 2
        self.parallel_param.rank_size = 8
        merger = GraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        self.assertIsInstance(merger.strategy, FullMerger)

    def test_merge_graph(self):
        merger = GraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        merger.strategy.merge_graphs = MagicMock()
        merger.merge_graph()
        merger.strategy.merge_graphs.assert_called_once()


class TestBaseGraphMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = [MagicMock(rank=i) for i in range(2)]
        self.parallel_param = ParallelParam(tp=1, pp=1, rank_size=2)
        self.is_bench = False
        self.merger = BaseGraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)

    def test_sort_merged_api_collection(self):
        graph = MagicMock()
        root = MagicMock()
        graph.root = root
        subnode1 = MagicMock(id=f"{GraphConst.APIS_BETWEEN_MODULES_ALL_RANKS}.0", op=NodeOp.api_collection)
        subnode1.subnodes = [MagicMock(id="op_Rank1.0"), MagicMock(id="op_Rank0.0")]
        root.subnodes = [subnode1]
        self.merger.sort_merged_api_collection(graph)
        self.assertEqual([n.id for n in subnode1.subnodes], ["op_Rank0.0", "op_Rank1.0"])

    def test_update_node_data_key(self):
        data_dict = {
            "old_id.input.0": {"full_op_name": "old_id.op"},
            "other_key": {"value": "test"}
        }
        new_dict = self.merger._update_node_data_key("old_id", "new_id", data_dict)
        self.assertEqual(new_dict, {
            "new_id.input.0": {"full_op_name": "new_id.op"},
            "other_key": {"value": "test"}
        })

    def test_compare_value_same(self):
        self.assertTrue(self.merger._compare_value_same(1, 1))
        self.assertFalse(self.merger._compare_value_same(1, 2))
        self.assertTrue(self.merger._compare_value_same("a", "a"))
        self.assertTrue(self.merger._compare_value_same(1, 1.00000001, has_uncertainty=True))
        self.assertFalse(self.merger._compare_value_same(1, 1.1, has_uncertainty=True))

    def test_merge_graph_api_collection(self):
        results = [MagicMock() for _ in range(2)]
        graph0, graph1 = Graph("name1"), Graph("name2")
        results[0].graph, results[1].graph = graph0, graph1
        root0, root1 = MagicMock(), MagicMock()
        graph0.root, graph1.root = root0, root1
        node0 = MagicMock(id=f"{GraphConst.APIS_BETWEEN_MODULES}.0")
        node0_sub1 = MagicMock(id="sub_op.0")
        node0.subnodes = [node0_sub1]
        node1 = MagicMock(id=f"{GraphConst.APIS_BETWEEN_MODULES}.0")
        node1_sub1 = MagicMock(id="sub_op.0")
        graph0.node_map = {f"{GraphConst.APIS_BETWEEN_MODULES}.0": node0}
        node1.subnodes = [node1_sub1]
        root0.subnodes = [node0]
        root1.subnodes = [node1]

        self.merger.merge_graph_api_collection(results)

        self.assertEqual(len(root0.subnodes), 1)
        self.assertTrue(root0.subnodes[0].id.startswith(GraphConst.APIS_BETWEEN_MODULES_ALL_RANKS))
        self.assertEqual(len(root0.subnodes[0].subnodes), 1)

    def test_split_graph_results_by_groups(self):
        groups = [[0, 1], [2, 3]]
        results = [MagicMock(rank=i) for i in range(4)]
        self.merger.build_graph_results = results
        split = self.merger.split_graph_results_by_groups(groups)
        self.assertEqual(len(split), 2)
        self.assertEqual([r.rank for r in split[0]], [0, 1])
        self.assertEqual([r.rank for r in split[1]], [2, 3])

    def test_compare_node_param_data(self):
        main_node = MagicMock()
        other_nodes = [MagicMock()]
        main_node.id = "id"
        other_nodes[0].id = "id"
        main_node.input_data = {"input.0": {Const.DTYPE: "torch.float16", Const.MAX: 1}}
        other_nodes[0].input_data = {"input.0": {Const.DTYPE: "torch.float16", Const.MAX: 2}}
        in_diff, out_diff = self.merger.compare_node_param_data(main_node, other_nodes)
        self.assertEqual(list(in_diff.keys()), ["input.0"])

    def test_compare_param_same(self):
        param1 = {Const.MAX: 1, Const.MIN: 0, Const.MEAN: 0.5, Const.NORM: 1}
        param2 = {Const.MAX: 1, Const.MIN: 0, Const.MEAN: 0.5, Const.NORM: 1}
        self.assertTrue(self.merger.compare_param_same(param1, param2))

        param2[Const.MAX] = 2
        self.assertFalse(self.merger.compare_param_same(param1, param2))

    def test_add_all_nodes_rank(self):
        graph0, graph1 = MagicMock(), MagicMock()
        node0, node1 = MagicMock(), MagicMock()
        graph0.node_map.values.return_value = [node0]
        graph1.node_map.values.return_value = [node1]
        self.build_graph_results[0].graph = graph0
        self.build_graph_results[1].graph = graph1

        self.merger._add_all_nodes_rank()

        self.assertEqual(node0.rank, 0)
        self.assertEqual(node1.rank, 1)

    def test_get_default_groups(self):
        self.parallel_param.tp = 4
        self.parallel_param.pp = 2
        self.parallel_param.rank_size = 8
        merger = BaseGraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        tp_groups, pp_groups = merger.get_default_groups()
        self.assertEqual(tp_groups, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(pp_groups, [[0, 4], [1, 5], [2, 6], [3, 7]])

        self.parallel_param.tp = 2
        self.parallel_param.pp = 2
        self.parallel_param.rank_size = 8
        merger = BaseGraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        tp_groups, pp_groups = merger.get_default_groups()
        self.assertEqual(tp_groups, [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual(pp_groups, [[0, 4], [1, 5], [2, 6], [3, 7]])

        self.parallel_param.tp = 2
        self.parallel_param.pp = 3
        self.parallel_param.rank_size = 8
        merger = BaseGraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        with self.assertRaises(MsprobeException):
            merger.get_default_groups()


class TestPPMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = [MagicMock(rank=i) for i in range(4)]
        self.parallel_param = ParallelParam(tp=1, pp=4, rank_size=4)
        self.is_bench = False
        self.merger = PPMerger(self.build_graph_results, self.parallel_param, self.is_bench)

    def test_trace_p2p_mapping(self):
        p2p_mapping = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 4, 7: 5}
        chains = self.merger._trace_p2p_mapping(p2p_mapping)
        self.assertEqual(len(chains), 2)
        self.assertIn([0, 2, 4, 6], chains)
        self.assertIn([1, 3, 5, 7], chains)

    @patch('msprobe.visualization.builder.graph_merger.PPMerger._merge_nodes')
    def test_merge_nodes(self, mock_merge):
        main_graph = MagicMock()
        main_node = MagicMock(id="module.layers.0.forward")
        other_graphs = [MagicMock() for _ in range(3)]
        for i, g in enumerate(other_graphs):
            g.get_node.return_value = MagicMock(id=f"module.layers.{i}.forward")

        self.merger._merge_nodes(main_graph, main_node, other_graphs)
        mock_merge.assert_called()

    def test_merge_graphs(self):
        self.merger.get_groups = MagicMock(return_value=[[0, 1, 2, 3]])
        self.merger.merge_pp_graphs = MagicMock(return_value=self.build_graph_results[:1])
        results = self.merger.merge_graphs()
        self.assertEqual(len(results), 1)

    def test_get_groups(self):
        for i, result in enumerate(self.build_graph_results):
            graph = MagicMock()
            node = MagicMock(id=f"Distributed.send.{i}.forward")
            node.input_data = {f"Distributed.send.{i}.forward.input.dst": {"value": (i + 1) % 4}}
            graph.node_map.values.return_value = [node]
            result.graph = graph

        groups = self.merger.get_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], [0, 1, 2, 3])

    def test_merge_other_unique_nodes(self):
        main_graph = MagicMock()
        main_node = MagicMock()
        other_nodes = [MagicMock()]
        main_node.subnodes = [MagicMock(id="main_sub.0")]
        other_nodes[0].subnodes = [MagicMock(id="other_sub.0")]

        self.merger._merge_other_unique_nodes(main_graph, main_node, other_nodes)
        self.assertEqual(len(main_node.subnodes), 2)

    def test_sort_nodes(self):
        graph = MagicMock()
        start_node = MagicMock(id="module.layers.0.forward%0%0")
        start_node.op = NodeOp.module
        api_node = MagicMock(id="Torch.mul.forward.0%0%0")
        graph.node_map = {"module.layers.0.forward%0%0": start_node, "Torch.mul.forward.0%0%0": api_node}
        parent_node = MagicMock()
        parent_node.subnodes = [start_node, api_node]
        start_node.upnode = parent_node

        self.merger._sort_nodes(graph, start_node)
        self.assertEqual(parent_node.subnodes[0].id, "module.layers.0.forward")
        self.assertEqual(parent_node.subnodes[1].id, "Torch.mul_rank0.forward.0")

    def test_add_node_to_main_graph(self):
        graph = MagicMock()
        node = MagicMock()
        subnode = MagicMock()
        node.subnodes = [subnode]

        self.merger._add_node_to_main_graph(graph, node)
        graph.node_map.__setitem__.assert_has_calls([call(node.id, node), call(subnode.id, subnode)])

    def test_get_node_sort_rule(self):
        node = MagicMock(id="module.layers.0.forward%1%2")
        self.assertEqual(self.merger._get_node_sort_rule(node), (2, 1))
        self.assertEqual(self.merger._get_node_sort_rule(node, rank_ascending=False), (-2, 1))

    def test_mark_node_id_position_rank(self):
        node = MagicMock()
        parent_node = MagicMock()
        parent_node.subnodes = [MagicMock(), node, MagicMock()]
        node.upnode = parent_node
        node.id = "module.layers.0.forward"

        self.merger._mark_node_id_position_rank(node, 2)
        self.assertEqual(node.id, "module.layers.0.forward%1%2")

    def test_update_node_id(self):
        graph = MagicMock()
        start_node = MagicMock(id="module.layers.0.forward%1%2")
        start_node.op = NodeOp.module
        start_node.pp_index = 1
        graph.node_map = {start_node.id: start_node}

        self.merger._update_node_id(graph, start_node)
        self.assertEqual(start_node.id, "module.layers.1.forward")


class TestTPMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = [MagicMock(rank=i) for i in range(4)]
        self.parallel_param = ParallelParam(tp=4, pp=1, rank_size=4)
        self.is_bench = False
        self.merger = TPMerger(self.build_graph_results, self.parallel_param, self.is_bench)

    def test_merge_params(self):
        params = {
            "input.0": [
                {Const.MAX: 1, Const.MIN: 0, Const.MEAN: 0.5, Const.NORM: 1},
                {Const.MAX: 2, Const.MIN: 0, Const.MEAN: 0.7, Const.NORM: 1.2}
            ]
        }
        merge_info = self.merger._merge_params(params)
        self.assertIn("The Max value merging method for input.0 is: max(1, 2) = 2", merge_info)
        self.assertIn("The Mean value merging method for input.0 is: (0.5 + 0.7) / 2 = 0.6", merge_info)

    def test_get_need_merge_node(self):
        main_node = MagicMock(id="module.matmul_rank0.forward")
        other_graphs = [MagicMock() for _ in range(3)]
        tp_merge_mapping = {0: [1, 2, 3]}

        for i, g in enumerate(other_graphs):
            g.node_map = {f"module.matmul_rank{i + 1}.forward": MagicMock()}

        nodes = self.merger._get_need_merge_node(main_node, other_graphs, tp_merge_mapping)
        self.assertEqual(len(nodes), 0)

    def test_merge_graphs(self):
        self.merger.get_groups = MagicMock(return_value=[[0, 1, 2, 3]])
        self.merger.merge_tp_graphs = MagicMock(return_value=self.build_graph_results[:1])
        results = self.merger.merge_graphs()
        self.assertEqual(len(results), 1)

    def test_get_groups(self):
        for i, result in enumerate(self.build_graph_results):
            graph = MagicMock()
            node = MagicMock(id=f"all_reduce.{i}")
            node.input_data = {f"all_reduce.{i}.input.group": {"group_ranks": [0, 1, 2, 3]}}
            graph.node_map.values.return_value = [node]
            result.graph = graph

        groups = self.merger.get_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], [0, 1, 2, 3])

    def test_handle_tp_matmul_reduce(self):
        node = MagicMock(id=f"module.RowParallelLinear.forward.0")
        node.op = NodeOp.module
        matmul_node = MagicMock(id="matmul.0")
        matmul_node.output_data = {"output.0": {Const.MAX: 1}}
        reduce_node = MagicMock(id="all_reduce.0")
        reduce_node.input_data = {"input.0": {Const.MAX: 1}}
        reduce_node.output_data = {"output.0": {Const.MAX: 2}}
        node.subnodes = [matmul_node, reduce_node]
        other_graphs = [MagicMock()]

        self.merger._handle_tp_matmul_reduce(node, other_graphs, {})
        self.assertEqual(matmul_node.output_data["output.0"][Const.MAX], 2)


class TestNoParallelMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = [MagicMock()]
        self.parallel_param = ParallelParam(tp=1, pp=1, rank_size=1)
        self.is_bench = False
        self.merger = NoParallelMerger(self.build_graph_results, self.parallel_param, self.is_bench)

    def test_merge_graphs(self):
        self.merger.merge_graph_api_collection = MagicMock()
        results = self.merger.merge_graphs()
        self.assertEqual(results, self.build_graph_results)
        self.merger.merge_graph_api_collection.assert_called_once_with(self.build_graph_results)


class TestTPPPMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = [MagicMock(rank=i) for i in range(4)]
        self.parallel_param = ParallelParam(tp=2, pp=2, rank_size=4)
        self.is_bench = False
        self.merger = TPPPMerger(self.build_graph_results, self.parallel_param, self.is_bench)

    @patch('msprobe.visualization.builder.graph_merger.TPMerger')
    @patch('msprobe.visualization.builder.graph_merger.PPMerger')
    def test_merge_graphs(self, mock_pp, mock_tp):
        tp_merger = MagicMock()
        pp_merger = MagicMock()
        mock_tp.return_value = tp_merger
        mock_pp.return_value = pp_merger

        pp_merger.get_groups.return_value = [[0, 1], [2, 3]]
        tp_merger.get_groups.return_value = [[0, 2], [1, 3]]
        tp_merger.merge_tp_graphs.return_value = [MagicMock()]

        results = self.merger.merge_graphs()
        self.assertEqual(len(results), 1)


class TestFullMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = [MagicMock(rank=i) for i in range(8)]
        self.parallel_param = ParallelParam(tp=2, pp=4, rank_size=8, vpp=1)
        self.is_bench = False
        self.merger = FullMerger(self.build_graph_results, self.parallel_param, self.is_bench)

    @patch('msprobe.visualization.builder.graph_merger.TPMerger')
    @patch('msprobe.visualization.builder.graph_merger.PPMerger')
    def test_merge_graphs(self, mock_pp, mock_tp):
        tp_merger = MagicMock()
        pp_merger = MagicMock()
        mock_tp.return_value = tp_merger
        mock_pp.return_value = pp_merger

        pp_merger.get_groups.return_value = [[0, 1, 2, 3], [4, 5, 6, 7]]
        tp_merger.get_groups.return_value = [[0, 4], [1, 5], [2, 6], [3, 7]]

        pp_result0 = MagicMock(rank=0)
        pp_result1 = MagicMock(rank=4)
        pp_merger.merge_pp_graphs.side_effect = [[pp_result0], [pp_result1]]

        tp_merger.merge_tp_graphs.side_effect = [[MagicMock()], [MagicMock()]]

        results = self.merger.merge_graphs()
        self.assertEqual(len(results), 1)


if __name__ == '__main__':
    unittest.main()
