import unittest
from unittest.mock import patch
from msprobe.visualization.builder.msprobe_adapter import (
    get_compare_mode,
    run_real_data,
    get_input_output,
    compare_data,
    format_node_data,
    compare_node,
    _format_decimal_string,
    _format_data
)
from msprobe.visualization.utils import GraphConst
from msprobe.visualization.graph.base_node import BaseNode
import torch
from msprobe.core.common.const import Const


class TestMsprobeAdapter(unittest.TestCase):
    @patch('msprobe.visualization.builder.msprobe_adapter.set_dump_path')
    @patch('msprobe.visualization.builder.msprobe_adapter.get_dump_mode', return_value=Const.SUMMARY)
    def test_get_compare_mode_summary(self, mock_get_dump_mode, mock_set_dump_path):
        mode = get_compare_mode("dummy_param")
        self.assertEqual(mode, GraphConst.SUMMARY_COMPARE)

    def test_get_input_output(self):
        node_data = {
            'input_args': [{'type': 'torch.Tensor', 'dtype': 'torch.int64', 'shape': [5],
                            'Max': 2049.0, 'Min': 0.0, 'Mean': 410.20001220703125, 'Norm': 2049.0009765625,
                            'requires_grad': False, 'full_op_name': 'Distributed.broadcast.0.forward_input.0'},
                           {'type': 'int', 'value': 0}],
            'input_kwargs': {'group': None},
            'output': [{'type': 'torch.Tensor', 'dtype': 'torch.int64', 'shape': [5],
                        'Max': 2049.0, 'Min': 0.0, 'Mean': 410.20001220703125, 'Norm': 2049.0009765625,
                        'requires_grad': False, 'full_op_name': 'Distributed.broadcast.0.forward_output.0'},
                       {'type': 'int', 'value': 0}, None]
        }
        node_id = "Distributed.broadcast.0.forward"
        input_data, output_data = get_input_output(node_data, node_id)
        self.assertIn("Distributed.broadcast.0.forward.output.0", output_data)
        self.assertIn("Distributed.broadcast.0.forward.input.0", input_data)

    def test_compare_data(self):
        data_dict_list1 = {'key1': {'type': 'Type1', 'dtype': 'DType1', 'shape': 'Shape1'}}
        data_dict_list2 = {'key1': {'type': 'Type1', 'dtype': 'DType1', 'shape': 'Shape1'}}
        data_dict_list3 = {'key1': {'type': 'Type2', 'dtype': 'DType1', 'shape': 'Shape1'}}
        data_dict_list4 = {}
        self.assertTrue(compare_data(data_dict_list1, data_dict_list2))
        self.assertFalse(compare_data(data_dict_list1, data_dict_list3))
        self.assertFalse(compare_data(data_dict_list1, data_dict_list4))

    def test_format_node_data(self):
        data_dict = {'node1': {'data_name': 'data1', 'full_op_name': 'op1'}}
        result = format_node_data(data_dict)
        self.assertNotIn('requires_grad', result['node1'])

    @patch('msprobe.visualization.builder.msprobe_adapter.get_accuracy')
    def test_compare_node(self, mock_get_accuracy):
        node_n = BaseNode('', 'node1')
        node_b = BaseNode('', 'node2')
        result = compare_node(node_n, node_b, GraphConst.REAL_DATA_COMPARE)
        mock_get_accuracy.assert_called_once()
        self.assertIsInstance(result, list)

    def test__format_decimal_string(self):
        s = "0.123456789%"
        formatted_s = _format_decimal_string(s)
        self.assertIn("0.123457%", formatted_s)
        self.assertEqual('0.123457', _format_decimal_string('0.12345678'))
        self.assertEqual('-1', _format_decimal_string('-1'))
        self.assertEqual('0.0.25698548%', _format_decimal_string('0.0.25698548%'))

    def test__format_data(self):
        data_dict = {'value': 0.123456789, 'value1': None, 'value2': "<class 'str'>", 'value3': 1.123123123123e-11,
                     'value4': torch.inf, 'value5': -1}
        _format_data(data_dict)
        self.assertEqual(data_dict['value'], '0.123457')
        self.assertEqual(data_dict['value1'], 'null')
        self.assertEqual(data_dict['value2'], '<class str>')
        self.assertEqual(data_dict['value3'], '1.123123e-11')
        self.assertEqual(data_dict['value4'], 'inf')
        self.assertEqual(data_dict['value5'], '-1')

        all_none_dict = {'Max': None, 'Min': None, 'Mean': None, 'Norm': None, 'type': None}
        _format_data(all_none_dict)
        self.assertEqual({'value': 'null'}, all_none_dict)
