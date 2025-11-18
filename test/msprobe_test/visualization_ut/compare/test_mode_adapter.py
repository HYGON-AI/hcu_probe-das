import json
import unittest
from unittest.mock import patch, MagicMock
from msprobe.visualization.compare.mode_adapter import ModeAdapter
from msprobe.visualization.graph.base_node import BaseNode
from msprobe.visualization.graph.node_op import NodeOp
from msprobe.visualization.utils import GraphConst, ToolTip
from msprobe.core.common.const import CompareConst


class TestModeAdapter(unittest.TestCase):

    def setUp(self):
        self.node_op = NodeOp.module
        self.node_id = "node_1"
        self.node = BaseNode(self.node_op, self.node_id)
        self.compare_mode = GraphConst.REAL_DATA_COMPARE
        self.adapter = ModeAdapter(self.compare_mode)
        self.compare_data_dict = [{}, {}]

    def test_add_md5_compare_data(self):
        node_data = {'md5_key': 'some_md5_value'}
        compare_data_dict = {'md5_key': 'expected_md5_value'}
        precision_index = ModeAdapter._add_md5_compare_data(node_data, compare_data_dict)
        self.assertEqual(precision_index, 1)

        node_data = {'Tensor.__imul__.0.forward.input.0': {'type': 'torch.Tensor', 'dtype': 'torch.int64', 'shape': [],
                                                           'Max': 16388, 'Min': 16388, 'Mean': 16388, 'Norm': 16388,
                                                           'requires_grad': 'False', 'md5': 'a563a4ea',
                                                           'full_op_name': 'Tensor.__imul__.0.forward.input.0',
                                                           'data_name': '-1', 'state': 'input'},
                     'Tensor.__imul__.0.forward.input.1': {'type': 'torch.Tensor', 'dtype': 'torch.int64', 'shape': [],
                                                           'Max': 4097, 'Min': 4097, 'Mean': 4097, 'Norm': 4097,
                                                           'requires_grad': 'False', 'md5': 'ce564339',
                                                           'full_op_name': 'Tensor.__imul__.0.forward.input.1',
                                                           'data_name': '-1', 'state': 'input'}}
        compare_dict = {'Tensor.__imul__.0.forward.input.0': ['Tensor.__imul__.0.forward.input.0',
                                                              'Tensor.__imul__.0.forward.input.0', 'torch.int64',
                                                              'torch.int64', [], [], 'False', 'False',
                                                              'a563a4ea', 'a563a4ea', True, 'pass', []],
                        'Tensor.__imul__.0.forward.input.1': ['Tensor.__imul__.0.forward.input.1',
                                                              'Tensor.__imul__.0.forward.input.1', 'torch.int64',
                                                              'torch.int64', [], [], 'False', 'False',
                                                              'ce564339', 'ce564559', True, 'diff', 'None']}
        precision_index = ModeAdapter._add_md5_compare_data(node_data, compare_dict)
        self.assertEqual(precision_index, 0)

    def test_add_real_compare_data(self):
        tensor_data = {'Module.module.Float16Module.forward.0.input.0':
                           ['Module.module.Float16Module.forward.0.input.0',
                            'Module.module.Float16Module.forward.0.input.0',
                            'torch.int64', 'torch.int64', [1, 1024], [1, 1024], 'False', 'False',
                            1.0, 0.0, 0.0, 1.0, 1.0,
                            29992.0, 1.0, 9100.3125, 474189.09375,
                            29992.0, 1.0, 9100.3125, 474189.09375,
                            True, 'Yes', '', None,
                            'Module.module.Float16Module.forward.0.input.0.pt'],
                       'Module.module.Float16Module.forward.0.input.1': [
                           'Module.module.Float16Module.forward.0.input.1',
                           'Module.module.Float16Module.forward.0.input.1',
                           'torch.int64', 'torch.int64', [1, 1024], [1, 1024], 'False', 'False',
                           1.0, 0.0, 0.0, None, 1.0,
                           1023.0, 0.0, 511.5, 18904.755859375,
                           1023.0, 0.0, 511.5, 18904.755859375,
                           True, 'Yes', '', 'None',
                           'Module.module.Float16Module.forward.0.input.1.pt'],
                       'Module.module.Float16Module.forward.0.input.2': [
                           'Module.module.Float16Module.forward.0.input.2',
                           'Module.module.Float16Module.forward.0.input.2',
                           'torch.bool', 'torch.bool', [1, 1, 1024, 1024], [1, 1, 1024, 1024], 'False', 'False',
                           1.0, 0.0, 0.0, 1.0, 1.0,
                           True, False, None, None, True, False, None, None,
                           True, 'Yes', '', 'None',
                           'Module.module.Float16Module.forward.0.input.2.pt'],
                       'Module.module.Float16Module.forward.0.kwargs.labels': [
                           'Module.module.Float16Module.forward.0.kwargs.labels',
                           'Module.module.Float16Module.forward.0.kwargs.labels',
                           'torch.int64', 'torch.int64', [1, 1024], [1, 1024], 'False', 'False',
                           1.0, 0.0, 0.0, 1.0, 1.0,
                           29992.0, 1.0, 9108.99609375, 474332.28125,
                           29992.0, 1.0, 9108.99609375, 474332.28125,
                           True, 'Yes', '', 'None',
                           'Module.module.Float16Module.forward.0.kwargs.labels.pt'],
                       'Module.module.Float16Module.forward.0.output.0': [
                           'Module.module.Float16Module.forward.0.output.0',
                           'Module.module.Float16Module.forward.0.output.0',
                           'torch.float32', 'torch.float32', [1, 1024], [1, 1024], 'False', 'False',
                           0.994182636336, 4.863566398621, 0.461487948895, 0.0068359375, 0.0234375,
                           15.402446746826172, 7.318280220031738, 11.375151634216309, 366.3365173339844,
                           10.538880348205566, 10.215872764587402, 10.378824234008789, 332.1264953613281,
                           True, 'No', '', 'None',
                           'Module.module.Float16Module.forward.0.output.0.pt']}
        node_data = {'Module.module.Float16Module.forward.0.input.0': {'type': 'torch.Tensor', 'dtype': 'torch.int64',
                                                                       'shape': [1, 1024], 'Max': 29992.0, 'Min': 1.0,
                                                                       'Mean': 9100.3125, 'Norm': 474189.09375,
                                                                       'requires_grad': 'False',
                                                                       'md5': '00000000'},
                     'Module.module.Float16Module.forward.0.input.1': {'type': 'torch.Tensor', 'dtype': 'torch.int64',
                                                                       'shape': [1, 1024], 'Max': 1023.0, 'Min': 0.0,
                                                                       'Mean': 511.5, 'Norm': 18904.755859375,
                                                                       'requires_grad': 'False',
                                                                       'md5': '00000000'},
                     'Module.module.Float16Module.forward.0.input.2': {'type': 'torch.Tensor', 'dtype': 'torch.bool',
                                                                       'shape': [1, 1, 1024, 1024], 'Max': True,
                                                                       'Min': False, 'Mean': None, 'Norm': None,
                                                                       'requires_grad': 'False',
                                                                       'md5': '00000000'},
                     'Module.module.Float16Module.forward.0.kwargs.labels': {'type': 'torch.Tensor',
                                                                             'dtype': 'torch.int64', 'shape': None,
                                                                             'Max': 29992.0, 'Min': 1.0,
                                                                             'Mean': 9108.99609375,
                                                                             'Norm': 474332.28125,
                                                                             'requires_grad': 'False',
                                                                             'md5': '00000000'},
                     'Module.module.Float16Module.forward.0.kwargs.None': None}
        min_thousandth = ModeAdapter._add_real_compare_data(node_data, tensor_data)
        self.assertEqual(min_thousandth, 1.0)

    def test_add_summary_compare_data(self):
        compare_data_dict = {
            'Module.module.Float16Module.forward.0.input.0': ['Module.module.Float16Module.forward.0.input.0',
                                                              'Module.module.Float16Module.forward.0.input.0',
                                                              'torch.int64', 'torch.int64', [4, 4096], [4, 4096],
                                                              'False', 'False',
                                                              0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
                                                              30119.0, 1.0, 8466.25, 1786889.625,
                                                              30119.0, 1.0, 8466.25, 1786889.625,
                                                              True, '', '', None],
            'Module.module.Float16Module.forward.0.input.1': ['Module.module.Float16Module.forward.0.input.1',
                                                              'Module.module.Float16Module.forward.0.input.1',
                                                              'torch.int64', 'torch.int64', [4, 4096], [4, 4096],
                                                              'False', 'False',
                                                              0.0, 0.0, 0.0, 0.0, '0.0%', 'N/A', '0.0%', '0.0%',
                                                              4095.0, 0.0, 2047.5, 302642.375,
                                                              4095.0, 0.0, 2047.5, 302642.375,
                                                              True, '', '', 'None'],
            'Module.module.Float16Module.forward.0.input.2': ['Module.module.Float16Module.forward.0.input.2',
                                                              'Module.module.Float16Module.forward.0.input.2',
                                                              'torch.bool', 'torch.bool',
                                                              [1, 1, 4096, 4096], [1, 1, 4096, 4096],
                                                              'False', 'False',
                                                              'N/A', 'N/A', 'N/A', 'N/A',
                                                              'N/A', 'N/A', 'N/A', 'N/A',
                                                              True, False, None, None, True, False, None, None,
                                                              True, '', '', 'None'],
            'Module.module.Float16Module.forward.0.input.labels': ['Module.module.Float16Module.forward.0.input.labels',
                                                                   'Module.module.Float16Module.forward.0.input.labels',
                                                                   'torch.float16', 'torch.float16',
                                                                   [4, 4096], [4, 4096],
                                                                   'False', 'False',
                                                                   0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
                                                                   30119.0, 0.00001, 8460.7685546875, 1786117.625,
                                                                   30119.0, 1.0, 8460.7685546875, 1786117.625,
                                                                   True, '', '', 'None']}
        node_data = {'Module.module.Float16Module.forward.0.input.0': {'type': 'torch.Tensor', 'dtype': 'torch.int64',
                                                                       'shape': [4, 4096], 'Max': 30119.0, 'Min': 1.0,
                                                                       'Mean': 8466.25, 'Norm': 1786889.625,
                                                                       'requires_grad': 'False',
                                                                       'data_name': '-1', 'md5': '00000000'},
                     'Module.module.Float16Module.forward.0.input.1': {'type': 'torch.Tensor', 'dtype': 'torch.int64',
                                                                       'shape': [4, 4096], 'Max': 4095.0, 'Min': 0.0,
                                                                       'Mean': 2047.5, 'Norm': 302642.375,
                                                                       'requires_grad': 'False',
                                                                       'data_name': '-1', 'md5': '00000000'},
                     'Module.module.Float16Module.forward.0.input.2': {'type': 'torch.Tensor', 'dtype': 'torch.bool',
                                                                       'shape': [1, 1, 4096, 4096], 'Max': True,
                                                                       'Min': False, 'Mean': None, 'Norm': None,
                                                                       'requires_grad': 'False',
                                                                       'data_name': '-1', 'md5': '00000000'},
                     'Module.module.Float16Module.forward.0.input.labels': {'type': 'torch.Tensor',
                                                                            'dtype': 'torch.float16',
                                                                            'shape': [4, 4096],
                                                                            'Max': 30119.0, 'Min': 0.00001,
                                                                            'Mean': 8460.7685546875,
                                                                            'Norm': 1786117.625,
                                                                            'requires_grad': 'False',
                                                                            'data_name': '-1', 'md5': '00000000'},
                     'Module.module.Float16Module.forward.0.kwargs.None': None}
        precision_index = ModeAdapter._add_summary_compare_data(node_data, compare_data_dict)
        self.assertEqual(precision_index, 0)

    def test_match_data(self):
        compare_data = ['Module.module.Float16Module.forward.0.input.0',
                        'Module.module.Float16Module.forward.0.input.0', 'torch.int64', 'torch.int64', [4, 4096],
                        [4, 4096], 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%', 30119.0, 1.0, 8466.25,
                        1786889.625, 30119.0, 1.0, 8466.25, 1786889.625, '', '']
        data_dict = {'type': 'torch.Tensor', 'dtype': 'torch.int64', 'shape': [4, 4096], 'Max': 30119.0, 'Min': 1.0,
                     'Mean': 8466.25, 'Norm': 1786889.625, 'requires_grad': False,
                     'full_op_name': 'Module.module.Float16Module.forward.0.input.0', 'data_name': '-1',
                     'md5': '00000000'}
        id_list = [6, 7, 8, 9, 10, 11, 12, 13]
        id_list1 = [6, 7, 8, 9, 10, 11, 12, 13, 14]
        key_list = ['Max diff', 'Min diff', 'Mean diff', 'L2norm diff', 'MaxRelativeErr', 'MinRelativeErr',
                    'MeanRelativeErr', 'NormRelativeErr']
        ModeAdapter._match_data(data_dict, compare_data, key_list, id_list1)
        self.assertNotIn('Max diff', data_dict)
        ModeAdapter._match_data(data_dict, compare_data, key_list, id_list)
        self.assertIn('Max diff', data_dict)

    def test_check_list_len(self):
        data_list = [1, 2]
        with self.assertRaises(ValueError):
            ModeAdapter._check_list_len(data_list, 3)

    @patch('msprobe.visualization.compare.mode_adapter.ModeAdapter')
    def test_parse_result(self, mock_mode_adapter):
        mock_mode_adapter._add_summary_compare_data.return_value = 0.5
        self.adapter.compare_mode = GraphConst.SUMMARY_COMPARE
        precision_index, other_dict = self.adapter.parse_result(self.node, self.compare_data_dict)
        self.assertEqual(precision_index, 0.5)
        self.assertEqual(other_dict, {})

        mock_mode_adapter._add_md5_compare_data.return_value = 1
        self.adapter.compare_mode = GraphConst.MD5_COMPARE
        precision_index, other_dict = self.adapter.parse_result(self.node, self.compare_data_dict)
        self.assertEqual(precision_index, 1)
        self.assertEqual(other_dict, {'Result': 'pass'})

        mock_mode_adapter._add_real_compare_data.return_value = 0.6
        self.adapter.compare_mode = GraphConst.REAL_DATA_COMPARE
        precision_index, other_dict = self.adapter.parse_result(self.node, self.compare_data_dict)
        self.assertEqual(precision_index, 0.0)
        self.assertEqual(other_dict, {})

    def test_prepare_real_data(self):
        result = self.adapter.prepare_real_data(self.node)
        self.assertTrue(result)

        self.adapter.compare_mode = GraphConst.SUMMARY_COMPARE
        result = self.adapter.prepare_real_data(self.node)
        self.assertFalse(result)

    def test_add_csv_data(self):
        compare_result_list = ['result1', 'result2']
        self.adapter.add_csv_data(compare_result_list)
        self.assertEqual(self.adapter.csv_data, compare_result_list)

    def test_get_tool_tip(self):
        self.adapter.compare_mode = GraphConst.MD5_COMPARE
        tips = self.adapter.get_tool_tip()
        self.assertEqual(tips, json.dumps({'md5': ToolTip.MD5}))

        self.adapter.compare_mode = GraphConst.SUMMARY_COMPARE
        tips = self.adapter.get_tool_tip()
        self.assertEqual(tips, json.dumps({
            CompareConst.MAX_DIFF: ToolTip.MAX_DIFF,
            CompareConst.MIN_DIFF: ToolTip.MIN_DIFF,
            CompareConst.MEAN_DIFF: ToolTip.MEAN_DIFF,
            CompareConst.NORM_DIFF: ToolTip.NORM_DIFF}))

        self.adapter.compare_mode = GraphConst.REAL_DATA_COMPARE
        tips = self.adapter.get_tool_tip()
        self.assertEqual(tips, json.dumps({
            CompareConst.ONE_THOUSANDTH_ERR_RATIO: ToolTip.ONE_THOUSANDTH_ERR_RATIO,
            CompareConst.FIVE_THOUSANDTHS_ERR_RATIO: ToolTip.FIVE_THOUSANDTHS_ERR_RATIO,
            CompareConst.COSINE: ToolTip.COSINE,
            CompareConst.MAX_ABS_ERR: ToolTip.MAX_ABS_ERR,
            CompareConst.MAX_RELATIVE_ERR: ToolTip.MAX_RELATIVE_ERR}))
