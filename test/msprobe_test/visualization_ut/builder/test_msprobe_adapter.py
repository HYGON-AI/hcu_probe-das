import os
import unittest
from unittest.mock import patch
from msprobe.visualization.builder.msprobe_adapter import (
    get_compare_mode,
    run_real_data,
    get_input_output,
    format_node_data,
    compare_node,
    _format_decimal_string,
    _format_data,
    MatchedNodeCalculator
)
from msprobe.visualization.utils import GraphConst
from msprobe.visualization.graph.base_node import BaseNode
import torch
from msprobe.core.common.const import Const

npu_data_in = {
    "Functional.conv2d.0.forward.input.0": {
        "type": "torch.Tensor",
        "dtype": "torch.float32",
        "shape": "[10, 3, 64, 64]",
        "Max": "4.350435",
        "Min": "-4.339223",
        "Mean": "-0.001572",
        "Norm": "350.247772",
        "requires_grad": "False",
        "data_name": "Functional.conv2d.0.forward.input.0.pt"
    },
    "Functional.conv2d.0.forward.input.1": {
        "type": "torch.Tensor",
        "dtype": "torch.float32",
        "shape": "[64, 3, 7, 7]",
        "Max": "0.099308",
        "Min": "-0.108559",
        "Mean": "0.000216",
        "Norm": "2.48812",
        "requires_grad": "True",
        "data_name": "Functional.conv2d.0.forward.input.1.pt"
    },
    "Functional.conv2d.0.forward.input.2": {
        "value": "null"
    },
    "Functional.conv2d.0.forward.input.3.0": {
        "type": "int",
        "value": "2",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "2",
        "Min": "2",
        "Mean": "2",
        "Norm": "2",
        "dtype": "<class int>",
        "shape": "[]"
    },
    "Functional.conv2d.0.forward.input.3.1": {
        "type": "int",
        "value": "2",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "2",
        "Min": "2",
        "Mean": "2",
        "Norm": "2",
        "dtype": "<class int>",
        "shape": "[]"
    },
    "Functional.conv2d.0.forward.input.4.0": {
        "type": "int",
        "value": "3",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "3",
        "Min": "3",
        "Mean": "3",
        "Norm": "3",
        "dtype": "<class int>",
        "shape": "[]"
    },
    "Functional.conv2d.0.forward.input.4.1": {
        "type": "int",
        "value": "3",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "3",
        "Min": "3",
        "Mean": "3",
        "Norm": "3",
        "dtype": "<class int>",
        "shape": "[]"
    },
    "Functional.conv2d.0.forward.input.5.0": {
        "type": "int",
        "value": "1",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "1",
        "Min": "1",
        "Mean": "1",
        "Norm": "1",
        "dtype": "<class int>",
        "shape": "[]"
    },
    "Functional.conv2d.0.forward.input.5.1": {
        "type": "int",
        "value": "1",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "1",
        "Min": "1",
        "Mean": "1",
        "Norm": "1",
        "dtype": "<class int>",
        "shape": "[]"
    },
    "Functional.conv2d.0.forward.input.6": {
        "type": "int",
        "value": "1",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "1",
        "Min": "1",
        "Mean": "1",
        "Norm": "1",
        "dtype": "<class int>",
        "shape": "[]"
    }
}

npu_data_out = {"Functional.conv2d.0.forward.output.0": {"type": "torch.Tensor", "dtype": "torch.float32",
                                                         "shape": "[10, 64, 32, 32]",
                                                         "Max": "inf", "Min": "-1.559762", "Mean": "inf", "Norm": "inf",
                                                         "requires_grad": "True",
                                                         "data_name": "Functional.conv2d.0.forward.output.0.pt"}}

npu_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))

bench_data_in = {
    "Functional.conv2d.0.forward.input.0": {
        "type": "torch.Tensor",
        "dtype": "torch.float32",
        "shape": "[10, 3, 64, 64]",
        "Max": "4.350435",
        "Min": "-4.339223",
        "Mean": "-0.001572",
        "Norm": "350.247772",
        "requires_grad": "False",
        "data_name": "Functional.conv2d.0.forward.input.0.pt"
    },
    "Functional.conv2d.0.forward.input.1": {
        "type": "torch.Tensor",
        "dtype": "torch.float32",
        "shape": "[64, 3, 7, 7]",
        "Max": "0.099308",
        "Min": "-0.108559",
        "Mean": "0.000216",
        "Norm": "2.48812",
        "requires_grad": "True",
        "data_name": "Functional.conv2d.0.forward.input.1.pt"
    },
    "Functional.conv2d.0.forward.input.2": {
        "value": "null"
    },
    "Functional.conv2d.0.forward.input.3.0": {
        "type": "int",
        "value": "2",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "2",
        "Min": "2",
        "Mean": "2",
        "Norm": "2",
        "dtype": "<class int>",
        "shape": "[]"
    },
    "Functional.conv2d.0.forward.input.3.1": {
        "type": "int",
        "value": "2",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "2",
        "Min": "2",
        "Mean": "2",
        "Norm": "2",
        "dtype": "<class int>",
        "shape": "[]"
    },
    "Functional.conv2d.0.forward.input.4.0": {
        "type": "int",
        "value": "3",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "3",
        "Min": "3",
        "Mean": "3",
        "Norm": "3",
        "dtype": "<class int>",
        "shape": "[]"
    },
    "Functional.conv2d.0.forward.input.4.1": {
        "type": "int",
        "value": "3",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "3",
        "Min": "3",
        "Mean": "3",
        "Norm": "3",
        "dtype": "<class int>",
        "shape": "[]"
    },
    "Functional.conv2d.0.forward.input.5.0": {
        "type": "int",
        "value": "1",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "1",
        "Min": "1",
        "Mean": "1",
        "Norm": "1",
        "dtype": "<class int>",
        "shape": "[]"
    },
    "Functional.conv2d.0.forward.input.5.1": {
        "type": "int",
        "value": "1",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "1",
        "Min": "1",
        "Mean": "1",
        "Norm": "1",
        "dtype": "<class int>",
        "shape": "[]"
    },
    "Functional.conv2d.0.forward.input.6": {
        "type": "int",
        "value": "1",
        "data_name": "-1",
        "requires_grad": "null",
        "Max": "1",
        "Min": "1",
        "Mean": "1",
        "Norm": "1",
        "dtype": "<class int>",
        "shape": "[]"
    }
}

bench_data_out = {"Functional.conv2d.0.forward.output.0": {"type": "torch.Tensor", "dtype": "torch.float32",
                                                           "shape": "[10, 64, 32, 32]",
                                                           "Max": "1.526799", "Min": "-1.559762", "Mean": "-0.000263",
                                                           "Norm": "244.860931",
                                                           "requires_grad": "True",
                                                           "data_name": "Functional.conv2d.0.forward.output.0.pt"}}

bench_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))

npu_data = {
    'input_data': npu_data_in,
    'output_data': npu_data_out,
    'dump_data_dir': npu_data_dir
}

bench_data = {
    'input_data': bench_data_in,
    'output_data': bench_data_out,
    'dump_data_dir': bench_data_dir
}

npu_data1 = {
    'input_data': npu_data_in,
    'output_data': npu_data_out,
    'dump_data_dir': ''
}

bench_data1 = {
    'input_data': bench_data_in,
    'output_data': bench_data_out,
    'dump_data_dir': ''
}


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

    def test_matched_node_calculator(self):
        template = {'success': False, 'error': [
            'The file path /home/louyujing/gitcode/MindStudio-Probe_3/test/msprobe_test/visualization_ut/builder/'
            'Functional.conv2d.0.forward.input.0.pt does not exist.',
            'The file path /home/louyujing/gitcode/MindStudio-Probe_3/test/msprobe_test/visualization_ut/builder/'
            'Functional.conv2d.0.forward.input.1.pt does not exist.',
            'The file path /home/louyujing/gitcode/MindStudio-Probe_3/test/msprobe_test/visualization_ut/builder/'
            'Functional.conv2d.0.forward.output.0.pt does not exist.'],
                    'data': {'precision_index': 1, 'input_info': {
                        'Functional.conv2d.0.forward.input.0': {'Cosine': 'unsupported', 'EucDist': 'unsupported',
                                                                'MaxAbsErr': 'unsupported',
                                                                'MaxRelativeErr': 'unsupported',
                                                                'One Thousandth Err Ratio': 'unsupported',
                                                                'Five Thousandths Err Ratio': 'unsupported',
                                                                'Requires_grad Consistent': 'True', 'Result': 'pass',
                                                                'Err_message': '[]'},
                        'Functional.conv2d.0.forward.input.1': {'Cosine': 'unsupported', 'EucDist': 'unsupported',
                                                                'MaxAbsErr': 'unsupported',
                                                                'MaxRelativeErr': 'unsupported',
                                                                'One Thousandth Err Ratio': 'unsupported',
                                                                'Five Thousandths Err Ratio': 'unsupported',
                                                                'Requires_grad Consistent': 'True', 'Result': 'pass',
                                                                'Err_message': '[]'},
                        'Functional.conv2d.0.forward.input.2': {'Cosine': '', 'EucDist': '', 'MaxAbsErr': '',
                                                                'MaxRelativeErr': '', 'One Thousandth Err Ratio': '',
                                                                'Five Thousandths Err Ratio': '',
                                                                'Requires_grad Consistent': '', 'Result': 'pass',
                                                                'Err_message': '[]'},
                        'Functional.conv2d.0.forward.input.3.0': {'Cosine': 'unsupported', 'EucDist': 'unsupported',
                                                                  'MaxAbsErr': 'unsupported',
                                                                  'MaxRelativeErr': 'unsupported',
                                                                  'One Thousandth Err Ratio': 'unsupported',
                                                                  'Five Thousandths Err Ratio': 'unsupported',
                                                                  'Requires_grad Consistent': 'True', 'Result': 'pass',
                                                                  'Err_message': '[]'},
                        'Functional.conv2d.0.forward.input.3.1': {'Cosine': 'unsupported', 'EucDist': 'unsupported',
                                                                  'MaxAbsErr': 'unsupported',
                                                                  'MaxRelativeErr': 'unsupported',
                                                                  'One Thousandth Err Ratio': 'unsupported',
                                                                  'Five Thousandths Err Ratio': 'unsupported',
                                                                  'Requires_grad Consistent': 'True', 'Result': 'pass',
                                                                  'Err_message': '[]'},
                        'Functional.conv2d.0.forward.input.4.0': {'Cosine': 'unsupported', 'EucDist': 'unsupported',
                                                                  'MaxAbsErr': 'unsupported',
                                                                  'MaxRelativeErr': 'unsupported',
                                                                  'One Thousandth Err Ratio': 'unsupported',
                                                                  'Five Thousandths Err Ratio': 'unsupported',
                                                                  'Requires_grad Consistent': 'True', 'Result': 'pass',
                                                                  'Err_message': '[]'},
                        'Functional.conv2d.0.forward.input.4.1': {'Cosine': 'unsupported', 'EucDist': 'unsupported',
                                                                  'MaxAbsErr': 'unsupported',
                                                                  'MaxRelativeErr': 'unsupported',
                                                                  'One Thousandth Err Ratio': 'unsupported',
                                                                  'Five Thousandths Err Ratio': 'unsupported',
                                                                  'Requires_grad Consistent': 'True', 'Result': 'pass',
                                                                  'Err_message': '[]'},
                        'Functional.conv2d.0.forward.input.5.0': {'Cosine': 'unsupported', 'EucDist': 'unsupported',
                                                                  'MaxAbsErr': 'unsupported',
                                                                  'MaxRelativeErr': 'unsupported',
                                                                  'One Thousandth Err Ratio': 'unsupported',
                                                                  'Five Thousandths Err Ratio': 'unsupported',
                                                                  'Requires_grad Consistent': 'True', 'Result': 'pass',
                                                                  'Err_message': '[]'},
                        'Functional.conv2d.0.forward.input.5.1': {'Cosine': 'unsupported', 'EucDist': 'unsupported',
                                                                  'MaxAbsErr': 'unsupported',
                                                                  'MaxRelativeErr': 'unsupported',
                                                                  'One Thousandth Err Ratio': 'unsupported',
                                                                  'Five Thousandths Err Ratio': 'unsupported',
                                                                  'Requires_grad Consistent': 'True', 'Result': 'pass',
                                                                  'Err_message': '[]'},
                        'Functional.conv2d.0.forward.input.6': {'Cosine': 'unsupported', 'EucDist': 'unsupported',
                                                                'MaxAbsErr': 'unsupported',
                                                                'MaxRelativeErr': 'unsupported',
                                                                'One Thousandth Err Ratio': 'unsupported',
                                                                'Five Thousandths Err Ratio': 'unsupported',
                                                                'Requires_grad Consistent': 'True', 'Result': 'pass',
                                                                'Err_message': '[]'}}, 'output_info': {
                        'Functional.conv2d.0.forward.output.0': {'Cosine': 'unsupported', 'EucDist': 'unsupported',
                                                                 'MaxAbsErr': 'unsupported',
                                                                 'MaxRelativeErr': 'unsupported',
                                                                 'One Thousandth Err Ratio': 'unsupported',
                                                                 'Five Thousandths Err Ratio': 'unsupported',
                                                                 'Requires_grad Consistent': 'True', 'Result': 'error',
                                                                 'Err_message': '["error: There is nan/inf/-inf in the '
                                                                                'maximum or minimum value of NPU."]'}}}}

        m = MatchedNodeCalculator(npu_data, bench_data)
        result = m.get_db_tensor_compare_result()
        self.assertEqual(result.get('data'), template.get('data'))
        self.assertEqual(result.get('success'), template.get('success'))
        self.assertEqual(len(result.get('error')), len(template.get('error')))

        template1 = {'success': False, 'error': ['The file path  does not exist.', '[msprobe] 非法文件路径： '],
                     'data': {}}
        m1 = MatchedNodeCalculator(npu_data1, bench_data1)
        result1 = m1.get_db_tensor_compare_result()
        self.assertEqual(result1, template1)

    def test_compare_index(self):
        tensor_index = ['Cosine', 'EucDist', 'MaxAbsErr', 'MaxRelativeErr', 'One Thousandth Err Ratio',
                        'Five Thousandths Err Ratio', 'Requires_grad Consistent', 'Result', 'Err_message']
        self.assertEqual(MatchedNodeCalculator.TENSOR_COMPARE_INDEX, tensor_index)
