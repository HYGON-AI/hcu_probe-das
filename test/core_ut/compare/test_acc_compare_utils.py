# coding=utf-8
import argparse
import json
import os
import shutil
import unittest
from unittest.mock import patch
import zlib
import tempfile

import numpy as np
import pandas as pd

from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.utils import ApiItemInfo, _compare_parser, check_and_return_dir_contents, extract_json, \
    count_struct, get_accuracy, get_rela_diff_summary_mode, merge_tensor, op_item_parse, read_op, result_item_init, \
    stack_column_process, table_value_is_valid, reorder_op_name_list, gen_op_item, ApiBatch, get_paired_dirs, \
    reorder_index, gen_api_batches

# test_read_op_1
op_data = {
    'input_args': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                    'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
                    'Norm': 2.2533628940582275, 'requires_grad': True},
                   {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                    'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
                    'Norm': 0.02844562754034996, 'requires_grad': False}],
    'input_kwargs': {'alpha': {'type': 'float', 'value': -0.1}},
    'output': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
                'Norm': 2.2533628940582275, 'requires_grad': True}]}
op_name = "Tensor.add_0.0.forward"
op_result = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'md5': '00000000',
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063, 'data_name': '-1',
     'Norm': 2.2533628940582275, 'requires_grad': 'True', 'full_op_name': 'Tensor.add_0.0.forward.input.0',
     'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'md5': '00000000',
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481, 'data_name': '-1',
     'Norm': 0.02844562754034996, 'requires_grad': 'False', 'full_op_name': 'Tensor.add_0.0.forward.input.1',
     'state': 'input'},
    {'full_op_name': 'Tensor.add_0.0.forward.input.alpha', 'dtype': "<class 'float'>", 'shape': '[]', 'md5': '0dae4479',
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'requires_grad': None, 'data_name': '-1', 'type': 'float',
     'value': -0.1, 'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'md5': '00000000',
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063, 'data_name': '-1',
     'Norm': 2.2533628940582275, 'requires_grad': 'True', 'full_op_name': 'Tensor.add_0.0.forward.output.0',
     'state': 'output'}]

# test_read_op_1
op_data_b = {
    'input': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
               'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
               'Norm': 2.2533628940582275, 'requires_grad': True},
              {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
               'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
               'Norm': 0.02844562754034996, 'requires_grad': False}],
    'output': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
                'Norm': 2.2533628940582275, 'requires_grad': True}]}
op_name_b = "Tensor.add_0.0.backward"
op_result_b = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'data_name': '-1', 'md5': '00000000',
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': 'True', 'full_op_name': 'Tensor.add_0.0.backward.input.0',
     'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'data_name': '-1', 'md5': '00000000',
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': 'False', 'full_op_name': 'Tensor.add_0.0.backward.input.1',
     'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'data_name': '-1', 'md5': '00000000',
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': 'True', 'full_op_name': 'Tensor.add_0.0.backward.output.0',
     'state': 'output'}]

# test_op_item_parse
parse_item = [
    {'Max': 4097.0, 'Mean': 820.2, 'Min': 0.0, 'Norm': 4097.0, 'dtype': 'torch.int64', 'requires_grad': False,
     'shape': [5], 'type': 'torch.Tensor'},
    {'type': 'int', 'value': 0},
    {'type': 'slice', 'value': [None, None, None]}
]
parse_op_name = 'Distributed.broadcast.0.forward.input'
parse_index = None
parse_item_list = None
parse_top_bool = True
o_result_parse = [
    {'Max': 4097.0, 'Mean': 820.2, 'Min': 0.0, 'Norm': 4097.0, 'dtype': 'torch.int64', 'requires_grad': 'False',
     'shape': [5], 'type': 'torch.Tensor', 'full_op_name': 'Distributed.broadcast.0.forward.input.0',
     'data_name': '-1', 'md5': '00000000', 'state': 'input'},
    {'full_op_name': 'Distributed.broadcast.0.forward.input.1', 'dtype': "<class 'int'>", 'shape': '[]',
     'md5': 'f4dbdf21', 'Max': 0, 'Min': 0, 'Mean': 0, 'Norm': 0, 'data_name': '-1', 'type': 'int', 'value': 0,
     'state': 'input', 'requires_grad': None},
    {'Max': None, 'Mean': None, 'Min': None, 'Norm': None, 'data_name': '-1', 'dtype': 'slice', 'type': 'slice',
     'full_op_name': 'Distributed.broadcast.0.forward.input.2', 'md5': '5fbbe87f', 'shape': '(3,)',
     'value': [None, None, None], 'state': 'input', 'requires_grad': None}
]

# test_resolve_api_special_parameters
data_dict = {
    "last_hidden_state":
        {"type": "torch.Tensor", "dtype": "torch.bfloat16"},
    "loss":
        {"type": "torch.Tensor", "dtype": "torch.float32"}
}
full_op_name = "Tensor.add_0.0.forward.input.0"
o_result_api_special = [
    {"type": "torch.Tensor", "dtype": "torch.bfloat16",
     "full_op_name": "Tensor.add_0.0.forward.input.last_hidden_state.0"},
    {"type": "torch.Tensor", "dtype": "torch.float32", "full_op_name": "Tensor.add_0.0.forward.input.loss.0"}
]

# test_get_accuracy
npu_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                        'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output.0',
                        'Functional.conv2d.0.forward.parameters.weight', 'Functional.conv2d.0.forward.parameters.bias',
                        'Functional.conv2d.0.parameters_grad.weight', 'Functional.conv2d.0.parameters_grad.bias'],
            'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                             ('torch.float32', [16])],
            'output_struct': [('torch.float32', [1, 16, 28, 28])],
            'params_struct': [('torch.float32', [1, 16, 28, 28]), ('torch.float32', [1, 16, 28, 28])],
            'params_grad_struct': [('torch.float32', [1, 16, 28, 28]), ('torch.float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029, 1.0],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 1.0],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 1.0],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0]],
            'stack_info': [],
            'requires_grad': [True, False, True, True, True, True, True, True]}
bench_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output.0',
                          'Functional.conv2d.0.forward.parameters.weight', 'Functional.conv2d.0.forward.parameters.bias',
                          'Functional.conv2d.0.parameters_grad.weight', 'Functional.conv2d.0.parameters_grad.bias'],
              'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                               ('torch.float32', [16])],
              'output_struct': [('torch.float32', [1, 16, 28, 28])],
              'params_struct': [('torch.float32', [1, 16, 28, 28]), ('torch.float32', [1, 16, 28, 28])],
              'params_grad_struct': [('torch.float32', [1, 16, 28, 28]), ('torch.float32', [1, 16, 28, 28])],
              'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029, 1.0],
                          [0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 1.0],
                          [0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 1.0],
                          [2.1166646480560303, -2.190781354904175, -0.003579073818400502, 1.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0]],
              'stack_info': [],
              'requires_grad': [True, False, True, True, True, True, True, True]}
highlight_dict = {'red_rows': [], 'yellow_rows': []}
o_result = [
    ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.0', 'torch.float32', 'torch.float32',
     [1, 1, 28, 28], [1, 1, 28, 28], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     3.029174327850342, -2.926689624786377, -0.06619918346405029, 1.0,
     3.029174327850342, -2.926689624786377, -0.06619918346405029, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.forward.input.1', 'Functional.conv2d.0.forward.input.1', 'torch.float32', 'torch.float32',
     [16, 1, 5, 5], [16, 1, 5, 5], False, False, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 1.0,
     0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.input.2', 'torch.float32', 'torch.float32',
     [16], [16], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 1.0,
     0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.forward.parameters.weight', 'Functional.conv2d.0.forward.parameters.weight', 'torch.float32',
     'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.forward.parameters.bias', 'Functional.conv2d.0.forward.parameters.bias', 'torch.float32',
     'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.forward.output.0', 'Functional.conv2d.0.forward.output.0', 'torch.float32', 'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     2.1166646480560303, -2.190781354904175, -0.003579073818400502, 1.0,
     2.1166646480560303, -2.190781354904175, -0.003579073818400502, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.parameters_grad.weight', 'Functional.conv2d.0.parameters_grad.weight', 'torch.float32',
     'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.parameters_grad.bias', 'Functional.conv2d.0.parameters_grad.bias', 'torch.float32',
     'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, '', '', 'None'],
]

# test_get_un_match_accuracy
o_result_unmatch_1 = [
    ['Functional.conv2d.0.forward.input.0', 'N/A', 'torch.float32', 'N/A', [1, 1, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A',
     'None'],
    ['Functional.conv2d.0.forward.input.1', 'N/A', 'torch.float32', 'N/A', [16, 1, 5, 5], 'N/A', 'N/A', 'N/A', 'N/A',
     'None'],
    ['Functional.conv2d.0.forward.input.2', 'N/A', 'torch.float32', 'N/A', [16], 'N/A', 'N/A', 'N/A', 'N/A', 'None'],
    ['Functional.conv2d.0.forward.parameters.weight', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A',
     'N/A', 'N/A',
     'None'],
    ['Functional.conv2d.0.forward.parameters.bias', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A',
     'N/A',
     'None'],
    ['Functional.conv2d.0.forward.output.0', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A',
     'None'],
    ['Functional.conv2d.0.parameters_grad.weight', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A',
     'N/A',
     'None'],
    ['Functional.conv2d.0.parameters_grad.bias', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A',
     'N/A',
     'None']
]
o_result_unmatch_2 = [
    ['Functional.conv2d.0.forward.input.0', 'N/A', 'torch.float32', 'N/A', [1, 1, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 3.029174327850342, -2.926689624786377, -0.06619918346405029, 1.0, 'N/A', 'N/A',
     'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.input.1', 'N/A', 'torch.float32', 'N/A', [16, 1, 5, 5], 'N/A', 'N/A', 'N/A', 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 1.0, 'N/A',
     'N/A',
     'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.input.2', 'N/A', 'torch.float32', 'N/A', [16], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 1.0, 'N/A', 'N/A',
     'N/A',
     'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.parameters.weight', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A',
     'N/A', 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 1.0, 1.0, 1.0, 1.0, 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.parameters.bias', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A',
     'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 1.0, 1.0, 1.0, 1.0, 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.output.0', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 2.1166646480560303, -2.190781354904175, -0.003579073818400502, 1.0, 'N/A',
     'N/A',
     'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.parameters_grad.weight', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A',
     'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 1.0, 1.0, 1.0, 1.0, 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.parameters_grad.bias', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A',
     'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 1.0, 1.0, 1.0, 1.0, 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None']
]
o_result_unmatch_3 = [
    ['Functional.conv2d.0.forward.input.0', 'N/A', 'torch.float32', 'N/A', [1, 1, 28, 28], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     3.029174327850342, -2.926689624786377, -0.06619918346405029, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.forward.input.1', 'N/A', 'torch.float32', 'N/A', [16, 1, 5, 5], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.forward.input.2', 'N/A', 'torch.float32', 'N/A', [16], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.forward.parameters.weight', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     1.0, 1.0, 1.0, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.forward.parameters.bias', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     1.0, 1.0, 1.0, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.forward.output.0', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     2.1166646480560303, -2.190781354904175, -0.003579073818400502, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.parameters_grad.weight', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     1.0, 1.0, 1.0, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.parameters_grad.bias', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     1.0, 1.0, 1.0, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None', ['-1', '-1']]
]

# test_merge_tensor
tensor_list = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'Max': 0.33033010363578796,
     'Min': -0.331031858921051, 'Mean': -0.030964046716690063, 'Norm': 2.2533628940582275, 'requires_grad': True,
     'full_op_name': 'Tensor.add_.0.forward.input.0', 'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_.0.forward.input.1',
     'state': 'input'},
    {'full_op_name': 'Tensor.add_.0.forward.input.alpha.0', 'dtype': "<class 'float'>", "shape": '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1', 'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_.0.forward.output.0',
     'state': 'output'}
]
result_op_dict = {'op_name': ['Tensor.add_.0.forward.input.0', 'Tensor.add_.0.forward.input.1',
                              'Tensor.add_.0.forward.input.alpha.0', 'Tensor.add_.0.forward.output.0'],
                  'input_struct': [('torch.float32', [16, 1, 3, 3]), ('torch.float32', [16, 1, 3, 3]),
                                   ("<class 'float'>", '[]')],
                  'output_struct': [('torch.float32', [16, 1, 3, 3])],
                  'params_struct': [],
                  'params_grad_struct': [],
                  'debug_struct': [],
                  'summary': [[0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275],
                              [0.003992878366261721, -0.008102823048830032, -0.0002002553956117481,
                               0.02844562754034996],
                              [-0.1, -0.1, -0.1, -0.1],
                              [0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275]],
                  'stack_info': [],
                  'state': ['input', 'input', 'input', 'output'],
                  'requires_grad': [True, False, None, True]}

tensor_list_md5 = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_.0.forward.input.0', 'md5': 1,
     'state': 'input'},
    {'full_op_name': 'Tensor.add_.0.forward.kwargs.alpha.0', 'dtype': "<class 'float'>", "shape": '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1', 'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_.0.forward.output.0', 'md5': 2,
     'state': 'output'}
]
result_op_dict_md5 = {'op_name': ['Tensor.add_.0.forward.input.0', 'Tensor.add_.0.forward.kwargs.alpha.0',
                                  'Tensor.add_.0.forward.output.0'],
                      'input_struct': [('torch.float32', [16, 1, 3, 3], 1), ("<class 'float'>", '[]', None)],
                      'output_struct': [('torch.float32', [16, 1, 3, 3], 2)],
                      'params_struct': [],
                      'params_grad_struct': [],
                      'debug_struct': [],
                      'summary': [
                          [0.003992878366261721, -0.008102823048830032, -0.0002002553956117481, 0.02844562754034996],
                          [-0.1, -0.1, -0.1, -0.1],
                          [0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275]],
                      'stack_info': [],
                      'state': ['input', 'input', 'output'],
                      'requires_grad': [False, None, True]
                      }

base_dir1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_utils1')
base_dir2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_utils2')


def create_json_files(base_dir):
    file_names = ['dump.json', 'stack.json', 'construct.json', 'debug.json']

    for file_name in file_names:
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump({}, f)


def create_rank_dirs(base_dir):
    folder_names = ['rank0', 'rank1']

    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        _compare_parser(self.parser)

        os.makedirs(base_dir1, mode=0o750, exist_ok=True)
        os.makedirs(base_dir2, mode=0o750, exist_ok=True)

    def tearDown(self):
        if os.path.exists(base_dir1):
            shutil.rmtree(base_dir1)
        if os.path.exists(base_dir2):
            shutil.rmtree(base_dir2)

    def test_extract_json_1(self):
        create_json_files(base_dir1)
        result = extract_json(base_dir1, Const.DUMP_JSON_FILE)
        self.assertEqual(result, os.path.join(base_dir1, 'dump.json'))

        result = extract_json(base_dir1, Const.STACK_JSON_FILE)
        self.assertEqual(result, os.path.join(base_dir1, 'stack.json'))

        result = extract_json(base_dir1, Const.DEBUG_JSON_FILE)
        self.assertEqual(result, os.path.join(base_dir1, 'debug.json'))

    def test_check_and_return_dir_contents(self):
        create_rank_dirs(base_dir2)
        result = check_and_return_dir_contents(base_dir2, 'rank')
        self.assertEqual(set(result), set(['rank0', 'rank1']))

    def test_read_op(self):
        result = read_op(op_data, op_name)
        self.assertEqual(result, op_result)

    def test_read_op_back(self):
        result = read_op(op_data_b, op_name_b)
        self.assertEqual(result, op_result_b)

    def test_op_item_parse(self):
        result = op_item_parse(parse_item, parse_op_name, 'input')
        self.assertEqual(result, o_result_parse)

    def test_op_item_parse_max_depth(self):
        with self.assertRaises(CompareException) as context:
            op_item_parse(parse_item, parse_op_name, 'input', depth=401)
        self.assertEqual(context.exception.code, CompareException.RECURSION_LIMIT_ERROR)

    def test_get_rela_diff_summary_mode_float_or_int(self):
        result_item = [0] * 16
        err_msg = ''
        npu_summary_data = [1, 1, 1, 1]
        bench_summary_data = [2, 2, 2, 2]
        result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                          bench_summary_data, err_msg)
        self.assertEqual(result_item, [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, '50.0%', '50.0%', '50.0%', '50.0%'])
        self.assertEqual(accuracy_check, '')
        self.assertEqual(err_msg, '')

    def test_get_rela_diff_summary_mode_bool(self):
        result_item = [0] * 16
        err_msg = ''
        npu_summary_data = [True, True, True, True]
        bench_summary_data = [True, True, True, True]
        result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                          bench_summary_data, err_msg)
        self.assertEqual(result_item, [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
        self.assertEqual(accuracy_check, '')
        self.assertEqual(err_msg, '')

    def test_get_rela_diff_summary_mode_nan(self):
        result_item = [0] * 16
        err_msg = ''
        npu_summary_data = [float('nan')]
        bench_summary_data = [float('nan')]
        result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                          bench_summary_data, err_msg)
        self.assertEqual(result_item, [0, 0, 0, 0, 0, 0, 0, 0, 'Nan', 0, 0, 0, 'Nan', 0, 0, 0])
        self.assertEqual(accuracy_check, '')
        self.assertEqual(err_msg, '')

    def test_count_struct_normal(self):
        op_dict = {
            CompareConst.OP_NAME: ['op1', 'op2', 'op3', 'op4', 'op5', 'op6', 'op7', 'op8'],
            CompareConst.INPUT_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
            CompareConst.OUTPUT_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
            CompareConst.PARAMS_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
            CompareConst.PARAMS_GRAD_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
        }

        result = count_struct(op_dict)

        self.assertEqual(result, (8, 2, 2, 2, 2))

    @patch('msprobe.core.compare.utils.logger')
    def test_mismatch_case(self, mock_logger):
        op_dict = {
            CompareConst.OP_NAME: ['op1', 'op2', 'op3', 'op4', 'op5', 'op6', 'op7', 'op8'],
            CompareConst.INPUT_STRUCT: [("torch.float32", [1])],
            CompareConst.OUTPUT_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
            CompareConst.PARAMS_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
            CompareConst.PARAMS_GRAD_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
        }

        with self.assertRaises(CompareException) as context:
            count_struct(op_dict)
        self.assertEqual(context.exception.code, CompareException.NAMES_STRUCTS_MATCH_ERROR)

    def test_get_accuracy(self):
        result = []
        get_accuracy(result, npu_dict, bench_dict, dump_mode=Const.SUMMARY)
        self.assertEqual(result, o_result)

    def test_merge_tensor_summary(self):
        op_dict = merge_tensor(tensor_list, dump_mode=Const.SUMMARY)
        self.assertEqual(op_dict, result_op_dict)

    def test_merge_tensor_md5(self):
        op_dict = merge_tensor(tensor_list_md5, dump_mode=Const.MD5)
        self.assertEqual(op_dict, result_op_dict_md5)

    def test_compare_parser_1(self):
        test_args = ["-i", "input.json", "-o", "output.json", "-s", "-c", "-f"]
        args = self.parser.parse_args(test_args)

        self.assertEqual(args.input_path, "input.json")
        self.assertEqual(args.output_path, "output.json")
        self.assertTrue(args.stack_mode)
        self.assertTrue(args.compare_only)
        self.assertTrue(args.fuzzy_match)

    def test_compare_parser_2(self):
        self.assertEqual(self.parser.parse_args('-i aaa -o'.split(' ')).output_path, './output')
        self.assertEqual(self.parser.parse_args('-i aaa'.split(' ')).output_path, './output')
        self.assertEqual(self.parser.parse_args('-i aaa -o ./aaa/output'.split(' ')).output_path, './aaa/output')

    def test_compare_parser_3(self):
        test_args = ["-i", "input.json", "-o", "output.json", "-cm", "cell_mapping.txt", "-dm",
                     "data_mapping.txt", "-lm", "layer_mapping.txt"]
        args = self.parser.parse_args(test_args)

        self.assertEqual(args.cell_mapping, "cell_mapping.txt")
        self.assertIsNone(args.api_mapping)  # 默认值应为 None
        self.assertEqual(args.data_mapping, "data_mapping.txt")
        self.assertEqual(args.layer_mapping, "layer_mapping.txt")

    def test_stack_column_process_stack_info(self):
        result_item = []
        has_stack = True
        index = 0
        key = CompareConst.INPUT_STRUCT
        npu_stack_info = ['abc']
        result_item = stack_column_process(result_item, has_stack, index, key, npu_stack_info)
        self.assertEqual(result_item, ['abc'])

    def test_stack_column_process_None(self):
        result_item = []
        has_stack = True
        index = 1
        key = CompareConst.INPUT_STRUCT
        npu_stack_info = ['abc']
        result_item = stack_column_process(result_item, has_stack, index, key, npu_stack_info)
        self.assertEqual(result_item, ['None'])

    def test_result_item_init_all_and_summary(self):
        n_name = 'Tensor.add.0.forward.input.0'
        n_struct = ('torch.float32', [96])
        npu_stack_info = ['abc']
        b_name = 'Tensor.add.0.forward.input.0'
        b_struct = ('torch.float32', [96])
        bench_stack_info = ['abc']
        requires_grad_pair = [True, True]
        n_info = ApiItemInfo(n_name, n_struct, npu_stack_info)
        b_info = ApiItemInfo(b_name, b_struct, bench_stack_info)

        dump_mode = Const.ALL
        result_item = result_item_init(n_info, b_info, requires_grad_pair, dump_mode)
        self.assertEqual(result_item, ['Tensor.add.0.forward.input.0', 'Tensor.add.0.forward.input.0',
                                       'torch.float32', 'torch.float32', [96], [96], True, True,
                                       ' ', ' ', ' ', ' ', ' ', ' '])

        dump_mode = Const.SUMMARY
        result_item = result_item_init(n_info, b_info, requires_grad_pair, dump_mode)
        self.assertEqual(result_item, ['Tensor.add.0.forward.input.0', 'Tensor.add.0.forward.input.0',
                                       'torch.float32', 'torch.float32', [96], [96], True, True,
                                       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])

    def test_result_item_init_md5(self):
        n_name = 'Tensor.add.0.forward.input.0'
        n_struct = ('torch.float32', [96], 'e87000dc')
        npu_stack_info = ['abc']
        b_name = 'Tensor.add.0.forward.input.0'
        b_struct = ('torch.float32', [96], 'e87000dc')
        bench_stack_info = ['abc']
        requires_grad_pair = [True, True]
        n_info = ApiItemInfo(n_name, n_struct, npu_stack_info)
        b_info = ApiItemInfo(b_name, b_struct, bench_stack_info)

        dump_mode = Const.MD5
        result_item = result_item_init(n_info, b_info, requires_grad_pair, dump_mode)
        self.assertEqual(result_item, ['Tensor.add.0.forward.input.0', 'Tensor.add.0.forward.input.0',
                                       'torch.float32', 'torch.float32', [96], [96], True, True,
                                       'e87000dc', 'e87000dc', True, 'pass'])

    def test_result_item_init_md5_index_error(self):
        n_name = 'Tensor.add.0.forward.input.0'
        n_struct = ('torch.float32', [96])
        npu_stack_info = ['abc']
        b_name = 'Tensor.add.0.forward.input.0'
        b_struct = ('torch.float32', [96])
        bench_stack_info = ['abc']
        requires_grad_pair = [True, True]
        n_info = ApiItemInfo(n_name, n_struct, npu_stack_info)
        b_info = ApiItemInfo(b_name, b_struct, bench_stack_info)

        dump_mode = Const.MD5
        with self.assertRaises(CompareException) as context:
            result_item = result_item_init(n_info, b_info, requires_grad_pair, dump_mode)
        self.assertEqual(context.exception.code, CompareException.INDEX_OUT_OF_BOUNDS_ERROR)

    def test_table_value_is_valid_int(self):
        result = table_value_is_valid(1)
        self.assertTrue(result)

    def test_table_value_is_valid_float(self):
        result = table_value_is_valid("-1.00")
        self.assertTrue(result)

        result = table_value_is_valid("+1.00")
        self.assertTrue(result)

    def test_table_value_is_valid_invalid_str(self):
        result = table_value_is_valid("=1.00")
        self.assertFalse(result)


class TestReorderIndex(unittest.TestCase):
    def test_reorder_index_mixed_states(self):
        op_parsed_list = [
            {Const.STATE: "OTHER"},
            {Const.STATE: Const.OUTPUT},
            {Const.STATE: Const.PARAMS},
            {Const.STATE: Const.PARAMS_GRAD},
            {Const.STATE: Const.INPUT},
            {"not_state": 123},  # 没有 STATE，算作 other
        ]

        reordered = reorder_index(op_parsed_list)
        self.assertTrue(reordered == [0, 4, 2, 1, 3])

    def test_reorder_index_all_params(self):
        op_parsed_list = [
            {Const.STATE: Const.PARAMS},
            {Const.STATE: Const.PARAMS},
            {Const.STATE: Const.PARAMS},
        ]
        reordered = reorder_index(op_parsed_list)
        self.assertTrue(reordered == [0, 1])

    def test_reorder_index_empty(self):
        op_parsed_list = []
        reordered = reorder_index(op_parsed_list)
        self.assertTrue(reordered == [])

    def test_reorder_index_single_element(self):
        op_parsed_list = [{Const.STATE: Const.PARAMS}]
        reordered = reorder_index(op_parsed_list)
        self.assertTrue(reordered == [])


class TestReorderOpNameList(unittest.TestCase):
    def test_reorder_op_name_list(self):
        # 标准顺序
        op_name_list = ["op.forward.input.0.0", "op.forward.output.0", "op.forward.output.1", "op.forward.parameters.1",
                        "op.forward.parameters.2", "op.parameters_grad.0"]
        state_list = ["input", "output", "output", "parameters", "parameters", "parameters_grad"]
        op_name_reorder, state_reorder = reorder_op_name_list(op_name_list, state_list)
        expected_result = ["op.forward.input.0.0", "op.forward.parameters.1", "op.forward.parameters.2",
                           "op.forward.output.0", "op.forward.output.1", "op.parameters_grad.0"]
        expected_state = ["input", "parameters", "parameters", "output", "output", "parameters_grad"]
        self.assertEqual(op_name_reorder, expected_result)
        self.assertEqual(state_reorder, expected_state)

        # 只有输入元素
        op_name_list = ["op.forward.input.0", "op.forward.input.1"]
        state_list = ["input", "input"]
        op_name_reorder, state_reorder = reorder_op_name_list(op_name_list, state_list)
        expected_result = ["op.forward.input.0", "op.forward.input.1"]
        expected_state = ["input", "input"]
        self.assertEqual(op_name_reorder, expected_result)
        self.assertEqual(state_reorder, expected_state)

        # 输入为空
        op_name_list = []
        state_list = []
        op_name_reorder, state_reorder = reorder_op_name_list(op_name_list, state_list)
        expected_result = []
        expected_state = []
        self.assertEqual(op_name_reorder, expected_result)
        self.assertEqual(state_reorder, expected_state)


class TestGenOpItem(unittest.TestCase):
    def test_gen_op_item_with_data_name(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'torch.Tensor',
            'dtype': 'torch.int64',
            'shape': [3],
            'value': [1, 2, 3],
            'Max': 3,
            'Min': 1,
            'Mean': 2,
            'Norm': 2
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        self.assertEqual(result['data_name'], 'test_data')
        self.assertEqual(result['full_op_name'], 'test_data')
        self.assertEqual(result['dtype'], 'torch.int64')
        self.assertEqual(result['shape'], [3])
        self.assertEqual(result['Max'], 3)
        self.assertEqual(result['Min'], 1)
        self.assertEqual(result['Mean'], 2)
        self.assertEqual(result['Norm'], 2)
        self.assertEqual(result['md5'], f"{zlib.crc32(str(op_data['value']).encode()):08x}")
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_empty_data_name(self):
        op_data = {
            'data_name': '',
            'type': 'torch.Tensor',
            'value': [1, 2, 3]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        # data_name为空时，应该被设置为'-1'
        self.assertEqual(result['data_name'], '-1')
        self.assertEqual(result['full_op_name'], op_name)
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_none_data_name(self):
        op_data = {
            'data_name': None,
            'type': 'torch.Tensor',
            'value': [1, 2, 3]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        # data_name为None时，应该被设置为'-1'
        self.assertEqual(result['data_name'], '-1')
        self.assertEqual(result['full_op_name'], op_name)
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_type_torch_size(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'torch.Size',
            'value': [2, 3, 4]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        self.assertEqual(result['dtype'], 'torch.Size')
        self.assertEqual(result['shape'], '[2, 3, 4]')
        self.assertEqual(result['Max'], None)
        self.assertEqual(result['Min'], None)
        self.assertEqual(result['Mean'], None)
        self.assertEqual(result['Norm'], None)
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_type_slice(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'slice',
            'value': [1, 2, 3]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        self.assertEqual(result['dtype'], 'slice')
        self.assertEqual(result['shape'], str(np.shape(np.array(op_data['value']))))
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_type_ellipsis(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'ellipsis',
            'value': '...'
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        self.assertEqual(result['dtype'], 'ellipsis')
        self.assertEqual(result['shape'], '[]')
        self.assertEqual(result['Max'], '...')
        self.assertEqual(result['Min'], '...')
        self.assertEqual(result['Mean'], '...')
        self.assertEqual(result['Norm'], '...')
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_type_torch_process_group(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'torch.ProcessGroup',
            'group_ranks': [0, 1]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        self.assertEqual(result['dtype'], 'torch.ProcessGroup')
        self.assertEqual(result['shape'], '[]')
        self.assertEqual(result['Max'], '[0, 1]')
        self.assertEqual(result['Min'], '[0, 1]')
        self.assertEqual(result['Mean'], '[0, 1]')
        self.assertEqual(result['Norm'], '[0, 1]')
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_default_dtype(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'other_type',
            'value': [1, 2, 3]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        self.assertEqual(result['dtype'], str(type(op_data['value'])))
        self.assertEqual(result['shape'], '[]')
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_md5(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'torch.Tensor',
            'value': [1, 2, 3]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        expected_md5 = f"{zlib.crc32(str(op_data['value']).encode()):08x}"
        self.assertEqual(result['md5'], expected_md5)
        self.assertEqual(result['state'], 'input')


class TestApiBatch(unittest.TestCase):
    def test_ApiBatch_increment_input(self):
        api_name = "functional.conv2d"
        start = 2
        api_batch = ApiBatch(api_name, start)

        api_batch.increment(Const.INPUT)

        self.assertEqual(api_batch._state, Const.INPUT)
        self.assertEqual(api_batch.input_len, 2)
        self.assertEqual(api_batch.params_end_index, 4)
        self.assertEqual(api_batch.output_end_index, 4)
        self.assertEqual(api_batch.params_grad_end_index, 4)

    def test_ApiBatch_increment_output(self):
        api_name = "functional.conv2d"
        start = 2
        api_batch = ApiBatch(api_name, start)

        api_batch.increment(Const.OUTPUT)

        self.assertEqual(api_batch._state, Const.OUTPUT)
        self.assertEqual(api_batch.input_len, 1)
        self.assertEqual(api_batch.params_end_index, 3)
        self.assertEqual(api_batch.output_end_index, 4)
        self.assertEqual(api_batch.params_grad_end_index, 4)

    def test_ApiBatch_increment_kwargs(self):
        api_name = "functional.conv2d"
        start = 2
        api_batch = ApiBatch(api_name, start)

        api_batch.increment(Const.KWARGS)

        self.assertEqual(api_batch._state, Const.KWARGS)
        self.assertEqual(api_batch.input_len, 2)
        self.assertEqual(api_batch.params_end_index, 4)
        self.assertEqual(api_batch.output_end_index, 4)
        self.assertEqual(api_batch.params_grad_end_index, 4)

    def test_ApiBatch_increment_params(self):
        api_name = "functional.conv2d"
        start = 2
        api_batch = ApiBatch(api_name, start)

        api_batch.increment(Const.PARAMS)

        self.assertEqual(api_batch._state, Const.PARAMS)
        self.assertEqual(api_batch.input_len, 1)
        self.assertEqual(api_batch.params_end_index, 4)
        self.assertEqual(api_batch.output_end_index, 4)
        self.assertEqual(api_batch.params_grad_end_index, 4)

    def test_ApiBatch_increment_multiple_input(self):
        api_name = "functional.conv2d"
        start = 2
        api_batch = ApiBatch(api_name, start)

        api_batch.increment(Const.INPUT)
        api_batch.increment(Const.INPUT)

        self.assertEqual(api_batch._state, Const.INPUT)
        self.assertEqual(api_batch.input_len, 3)
        self.assertEqual(api_batch.params_end_index, 5)
        self.assertEqual(api_batch.output_end_index, 5)
        self.assertEqual(api_batch.params_grad_end_index, 5)

    def test_ApiBatch_increment_multiple_output(self):
        api_name = "functional.conv2d"
        start = 2
        api_batch = ApiBatch(api_name, start)

        api_batch.increment(Const.OUTPUT)
        api_batch.increment(Const.OUTPUT)

        self.assertEqual(api_batch._state, Const.OUTPUT)
        self.assertEqual(api_batch.input_len, 1)
        self.assertEqual(api_batch.params_end_index, 3)
        self.assertEqual(api_batch.output_end_index, 5)
        self.assertEqual(api_batch.params_grad_end_index, 5)


class TestGenApiBatches(unittest.TestCase):
    def test_gen_api_batches_normal(self):
        result_df_part1 = pd.DataFrame(o_result)
        result_df_part1.columns = CompareConst.SUMMARY_COMPARE_RESULT_HEADER_STACK
        new_columns = [
            ['input', 'Functional.conv2d.0.forward'],
            ['input', 'Functional.conv2d.0.forward'],
            ['input', 'Functional.conv2d.0.forward'],
            ['parameters', 'Functional.conv2d.0.forward'],
            ['parameters', 'Functional.conv2d.0.forward'],
            ['output', 'Functional.conv2d.0.forward'],
            ['parameters_grad', 'Functional.conv2d.0.forward'],
            ['parameters_grad', 'Functional.conv2d.0.forward']
        ]
        result_df_part2 = pd.DataFrame(new_columns)
        result_df_part2.columns = [Const.STATE, Const.API_ORIGIN_NAME]
        result_df = pd.concat([result_df_part1, result_df_part2], axis=1)
        result = result_df.values
        header = result_df.columns.tolist()
        result_api_batches = gen_api_batches(result, header)

        api_batch = ApiBatch('Functional.conv2d.0.forward', 0)
        api_batch.input_len = 3
        api_batch.output_end_index = 6
        api_batch.params_end_index = 5
        api_batch.params_grad_end_index = 8
        api_batch._state = 'parameters_grad'

        result_api_batch = result_api_batches[0]
        self.assertEqual(result_api_batch.api_name, api_batch.api_name)
        self.assertEqual(result_api_batch.start, api_batch.start)
        self.assertEqual(result_api_batch.input_len, api_batch.input_len)
        self.assertEqual(result_api_batch.params_end_index, api_batch.params_end_index)
        self.assertEqual(result_api_batch.params_grad_end_index, api_batch.params_grad_end_index)
        self.assertEqual(result_api_batch._state, api_batch._state)


class TestGetPairedSteps(unittest.TestCase):
    def setUp(self):
        self.npu_dir = tempfile.TemporaryDirectory()
        self.bench_dir = tempfile.TemporaryDirectory()

        self.npu_files = ['step1', 'step2']
        for name in self.npu_files:
            open(os.path.join(self.npu_dir.name, name), 'w').close()

        self.bench_files = ['step2', 'step3']
        for name in self.bench_files:
            open(os.path.join(self.bench_dir.name, name), 'w').close()

    def tearDown(self):
        self.npu_dir.cleanup()
        self.bench_dir.cleanup()

    def test_get_paired_steps(self):
        paired = get_paired_dirs(self.npu_dir.name, self.bench_dir.name)
        self.assertEqual(set(paired), {'step2'})
