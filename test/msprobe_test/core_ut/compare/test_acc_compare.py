# coding=utf-8
import json
import os
import shutil
import threading
import unittest
from unittest.mock import patch
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import torch

from msprobe.core.common.file_utils import load_json
from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.acc_compare import ModeConfig, MappingConfig, MappingDict, Comparator, ParseData, \
    ProcessDf, Match, CreateTable, CalcStatsDiff
from msprobe.core.compare.stats_diff_calc import ValType, ALL_TYPES


npu_op_item_data_fuzzy = {
    'op_name': 'Functional.conv2d.0.forward.input.0',
    'dtype': 'torch.float32',
    'shape': [1, 1, 28, 28],
    'summary': [3.029174327850342, -2.926689624786377, -0.06619918346405029],
    'stack_info': [],
    'data_name': 'Functional.conv2d.0.forward.input.0.pt',
    'compare_key': 'Functional.conv2d.0.forward.input.0',
    'compare_shape': [1, 1, 28, 28],
}
npu_op_item_fuzzy = pd.Series(npu_op_item_data_fuzzy)
npu_op_item_data_fuzzy_2 = {
    'op_name': 'Functional.conv2d.0.forward.input.1',
    'dtype': 'torch.float32',
    'shape': [1, 1, 28, 28],
    'summary': [3.029174327850342, -2.926689624786377, -0.06619918346405029],
    'stack_info': [],
    'data_name': 'Functional.conv2d.0.forward.input.1.pt',
    'compare_key': 'Functional.conv2d.0.forward.input.1',
    'compare_shape': [1, 1, 28, 28],
}
npu_op_item_fuzzy_2 = pd.Series(npu_op_item_data_fuzzy_2)
bench_op_item_data_fuzzy = {
    'op_name': 'Functional.conv2d.1.forward.input.0',
    'dtype': 'torch.float32',
    'shape': [1, 1, 28, 28],
    'summary': [3.029174327850342, -2.926689624786377, -0.06619918346405029],
    'stack_info': [],
    'data_name': 'Functional.conv2d.1.forward.input.0.pt',
    'compare_key': 'Functional.conv2d.1.forward.input.0',
    'compare_shape': [1, 1, 28, 28],
}
bench_op_item_fuzzy = pd.Series(bench_op_item_data_fuzzy)

npu_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                        'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
            'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                             ('torch.float32', [16])],
            'output_struct': [('torch.float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

npu_dict2 = {'op_name': ['Functional.conv2d.1.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                         'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
             'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                              ('torch.float32', [16])],
             'output_struct': [('torch.float32', [1, 16, 28, 28])],
             'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                         [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                         [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                         [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

bench_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
              'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                               ('torch.float32', [16])],
              'output_struct': [('torch.float32', [1, 16, 28, 28])],
              'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                          [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                          [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                          [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

tensor_list = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'Max': 0.33033010363578796,
     'Min': -0.331031858921051, 'Mean': -0.030964046716690063, 'Norm': 2.2533628940582275, 'requires_grad': True,
     'full_op_name': 'Tensor.add_.0.forward.input.0'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_.0.forward.input.1'},
    {'full_op_name': 'Tensor.add_.0.forward.input.alpha.0', 'dtype': "<class 'float'>", "shape": '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_.0.forward.output.0'}
]

result_op_dict = {'op_name': ['Tensor.add_.0.forward.input.0', 'Tensor.add_.0.forward.input.1',
                              'Tensor.add_.0.forward.input.alpha.0', 'Tensor.add_.0.forward.output.0'],
                  'input_struct': [('torch.float32', [16, 1, 3, 3]), ('torch.float32', [16, 1, 3, 3]),
                                   ("<class 'float'>", '[]')],
                  'output_struct': [('torch.float32', [16, 1, 3, 3])],
                  'summary': [[0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275],
                              [0.003992878366261721, -0.008102823048830032, -0.0002002553956117481,
                               0.02844562754034996],
                              [-0.1, -0.1, -0.1, -0.1],
                              [0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275]],
                  'stack_info': []}

o_result = [
    ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.0', 'torch.float32', 'torch.float32',
     [1, 1, 28, 28], [1, 1, 28, 28], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 3.029174327850342,
     -2.926689624786377,
     -0.06619918346405029, 3.029174327850342, -2.926689624786377, -0.06619918346405029, '', '', 'None'],
    ['Functional.conv2d.0.forward.input.1', 'Functional.conv2d.0.forward.input.1', 'torch.float32', 'torch.float32',
     [16, 1, 5, 5], [16, 1, 5, 5], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 0.19919930398464203,
     -0.19974489510059357,
     0.006269412115216255, 0.19919930398464203, -0.19974489510059357, 0.006269412115216255, '', '', 'None'],
    ['Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.input.2', 'torch.float32', 'torch.float32',
     [16], [16], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 0.19734230637550354, -0.18177609145641327,
     0.007903944700956345,
     0.19734230637550354, -0.18177609145641327, 0.007903944700956345, '', '', 'None'],
    ['Functional.conv2d.0.forward.output', 'Functional.conv2d.0.forward.output', 'torch.float32', 'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 2.1166646480560303,
     -2.190781354904175,
     -0.003579073818400502, 2.1166646480560303, -2.190781354904175, -0.003579073818400502, '', '', 'None']]

npu_dict_aten = {'op_name': ['Aten__native_batch_norm_legit_functional.default_0_forward.input.0',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.input.1',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.input.2',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.input.3',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.input.4',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.0',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.1',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.2',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.3',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.4'],
                 'input_struct': [('torch.float16', [256, 256, 14, 14]), ('torch.float32', [256]),
                                  ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256])],
                 'output_struct': [('torch.float16', [256, 256, 14, 14]), ('torch.float32', [256]),
                                   ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256])],
                 'summary': [[139.625, -127.5625, -0.0103607177734375],
                             [2.5276029109954834, -2.1788690090179443, -0.0008259844034910202],
                             [2.472219944000244, -2.845968723297119, -0.008756577968597412],
                             [2.763145923614502, -3.398397922515869, -0.052132632583379745],
                             [2.673110008239746, -3.149275064468384, 0.01613386906683445],
                             [13.5546875, -10.640625, -0.008758544921875],
                             [0.30550330877304077, -0.24485322833061218, -0.010361209511756897],
                             [623.9192504882812, 432.96826171875, 520.2276611328125],
                             [2.4797861576080322, -3.055997371673584, -0.04795549064874649],
                             [61.7945556640625, 42.59713363647461, 52.03831481933594]]}

bench_dict_functional = {
    'op_name': ['Functional_batch_norm_0_forward.input.0', 'Functional_batch_norm_0_forward.input.1',
                'Functional_batch_norm_0_forward.input.2', 'Functional_batch_norm_0_forward.input.3',
                'Functional_batch_norm_0_forward.input.4', 'Functional_batch_norm_0_forward.output'],
    'input_struct': [('torch.float32', [256, 256, 14, 14]), ('torch.float32', [256]), ('torch.float32', [256]),
                     ('torch.float32', [256]), ('torch.float32', [256])],
    'output_struct': [('torch.float32', [256, 256, 14, 14])],
    'summary': [[3.061628818511963, -3.22507381439209, 3.634914173744619e-05],
                [0.0005779837374575436, -0.0006301702815108001, 3.634906533989124e-06],
                [0.9338104128837585, 0.9277191162109375, 0.930335283279419],
                [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                [5.397906303405762, -5.796811580657959, 2.5283952709287405e-10]]
}

aten_result = [
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.0', 'Functional_batch_norm_0_forward.input.0',
     'torch.float16', 'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 136.56337118148804, -124.33742618560791,
     -0.010397066915174946, ' ', '4460.480981749501%', '3855.335826136584%', '28603.33536971545%', ' ', 139.625,
     -127.5625, -0.0103607177734375, 3.061628818511963, -3.22507381439209, 3.634914173744619e-05, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.1', 'Functional_batch_norm_0_forward.input.1',
     'torch.float32', 'torch.float32', [256], [256], 2.527024927258026, -2.1782388387364335, -0.0008296193100250093,
     ' ', '437213.84590749856%', '345658.76916858414%', '22823.676544842117%', ' ', 2.5276029109954834,
     -2.1788690090179443, -0.0008259844034910202, 0.0005779837374575436, -0.0006301702815108001, 3.634906533989124e-06,
     'Warning', 'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.2', 'Functional_batch_norm_0_forward.input.2',
     'torch.float32', 'torch.float32', [256], [256], 1.5384095311164856, -3.7736878395080566, -0.9390918612480164, ' ',
     '164.74538192025793%', '406.7705163736246%', '100.94122819224167%', ' ', 2.472219944000244, -2.845968723297119,
     -0.008756577968597412, 0.9338104128837585, 0.9277191162109375, 0.930335283279419, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.3', 'Functional_batch_norm_0_forward.input.3',
     'torch.float32', 'torch.float32', [256], [256], 1.763145923614502, -4.398397922515869, -1.0521326325833797, ' ',
     '176.3145923614502%', '439.8397922515869%', '105.21326325833797%', ' ', 2.763145923614502, -3.398397922515869,
     -0.052132632583379745, 1.0, 1.0, 1.0, 'Warning', 'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.4', 'Functional_batch_norm_0_forward.input.4',
     'torch.float32', 'torch.float32', [256], [256], 2.673110008239746, -3.149275064468384, 0.01613386906683445, ' ',
     'N/A', 'N/A', 'N/A', ' ', 2.673110008239746, -3.149275064468384, 0.01613386906683445, 0.0, 0.0, 0.0, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.0', 'Functional_batch_norm_0_forward.output',
     'torch.float16', 'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 8.156781196594238, -4.843813419342041,
     -0.008758545174714527, ' ', '151.11009228611078%', '83.55995967687207%', '3464072756.115108%', ' ', 13.5546875,
     -10.640625, -0.008758544921875, 5.397906303405762, -5.796811580657959, 2.5283952709287405e-10, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.1', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', ' ', 0.30550330877304077, -0.24485322833061218, -0.010361209511756897, 'Nan', 'Nan',
     'Nan',
     'Yes', '', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.2', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', ' ', 623.9192504882812, 432.96826171875, 520.2276611328125, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.3', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', ' ', 2.4797861576080322, -3.055997371673584, -0.04795549064874649, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.4', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', ' ', 61.7945556640625, 42.59713363647461, 52.03831481933594, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None']]

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
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_0.0.forward.input.0'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_0.0.forward.input.1'},
    {'full_op_name': 'Tensor.add_0.0.forward.input.alpha.0', 'dtype': "<class 'float'>", 'shape': '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_0.0.forward.output.0'}]

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_data')
base_dir2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_data2')
base_dir3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_data3')
pt_dir = os.path.join(base_dir3, f'dump_data_dir')


def generate_dump_json(base_dir):
    data_path = os.path.join(base_dir, 'dump.json')
    data = {
        'task': 'statistics',
        'framework': 'pytorch',
        'level': 'L1',
        'dump_data_dir': '',
        'data': {
            'Functional.linear.0.forward': {
                'input_args': [
                    {'type': 'torch.Tensor',
                     'dtype': 'torch.float32',
                     'shape': [2, 2],
                     'Max': 2,
                     'Min': 0,
                     'Mean': 1,
                     'Norm': 1,
                     'requires_grad': False,
                     'data_name': 'Functional.linear.0.forward.input.0.pt'
                     }
                ]
            }
        }
    }
    with open(data_path, 'w') as json_file:
        json.dump(data, json_file)


def generate_dump_json_md5(base_dir):
    data_path = os.path.join(base_dir, 'dump_md5.json')
    data = {
        'task': 'statistics',
        'level': 'L1',
        'dump_data_dir': '',
        'data': {
            'Functional.linear.0.forward': {
                'input_args': [
                    {'type': 'torch.Tensor',
                     'dtype': 'torch.float32',
                     'shape': [2, 2],
                     'Max': 2,
                     'Min': 0,
                     'Mean': 1,
                     'Norm': 1,
                     'requires_grad': False,
                     'md5': 123456
                     }
                ]
            }
        }
    }
    with open(data_path, 'w') as json_file:
        json.dump(data, json_file)


def generate_stack_json(base_dir):
    data_path = os.path.join(base_dir, 'stack.json')
    data = {'Functional.linear.0.forward': ['File']}
    with open(data_path, 'w') as json_file:
        json.dump(data, json_file)


def generate_pt(base_dir):
    data_path = os.path.join(base_dir, 'Functional.linear.0.forward.input.0.pt')
    data = torch.Tensor([1, 2, 3, 4])
    torch.save(data, data_path)


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        os.makedirs(base_dir, mode=0o750, exist_ok=True)
        os.makedirs(base_dir2, mode=0o750, exist_ok=True)
        os.makedirs(base_dir3, mode=0o750, exist_ok=True)
        os.makedirs(pt_dir, mode=0o750, exist_ok=True)

        self.lock = threading.Lock()

    def tearDown(self):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        if os.path.exists(base_dir2):
            shutil.rmtree(base_dir2)
        if os.path.exists(pt_dir):
            shutil.rmtree(pt_dir)
        if os.path.exists(base_dir3):
            shutil.rmtree(base_dir3)

    def test_gen_merge_list(self):
        op_data = {
            'input_args': [
                {
                    'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [2, 2],
                    'Max': 1, 'Min': 1, 'Mean': 1, 'Norm': 1, 'requires_grad': False,
                    'data_name': 'Functional.linear.0.forward.input.0.pt',
                    'full_op_name': 'Functional.linear.0.forward.input.0'
                }
            ]
        }
        json_data = {'data': {'Functional.linear.0.forward': op_data}}
        op_name = 'Functional.linear.0.forward'
        stack_json_data = {'Functional.linear.0.forward': ['File']}
        target_merge_list = [
            {
                'full_op_name': 'Functional.linear.0.forward.input.0',
                'type': 'torch.Tensor',
                'dtype': 'torch.float32',
                'shape': [2, 2],
                'requires_grad': 'False',
                'Max': 1,
                'Min': 1,
                'Mean': 1,
                'Norm': 1,
                'md5': '00000000',
                'data_name': 'Functional.linear.0.forward.input.0.pt',
                'state': 'input'
            },
            {
                'full_op_name': 'Functional.linear.0.forward',
                'full_info': ['File']
            }
        ]

        config_dict = {
            'stack_mode':  True,
            'auto_analyze': True,
            'fuzzy_match': False,
            'dump_mode': Const.SUMMARY,
        }
        mode_config = ModeConfig(**config_dict)

        result = ParseData(mode_config, 'rank0').gen_merge_list(json_data, op_name, stack_json_data)
        self.assertEqual(result, target_merge_list)

    def test_check_op_item_fuzzy(self):
        config_dict = {
            'stack_mode': False,
            'auto_analyze': True,
            'fuzzy_match': True,
            'dump_mode': Const.SUMMARY,
        }
        mode_config = ModeConfig(**config_dict)
        mapping_config = MappingConfig()

        match = Match(mode_config, mapping_config, cross_frame=False)
        result = match.check_op_item(npu_op_item_fuzzy, bench_op_item_fuzzy)
        self.assertEqual(result, True)

    def test_compare_statistics(self):
        generate_dump_json(base_dir)
        generate_stack_json(base_dir)
        file_list = [os.path.join(base_dir, 'dump.json'), os.path.join(base_dir, 'dump.json'),
                     os.path.join(base_dir, 'stack.json')]

        config_dict = {
            'stack_mode': True,
            'auto_analyze': True,
            'fuzzy_match': False,
            'dump_mode': Const.SUMMARY,
        }
        mode_config = ModeConfig(**config_dict)
        mapping_config = MappingConfig()

        from msprobe.pytorch.compare.pt_compare import read_real_data
        comparator = Comparator(read_real_data, mode_config, mapping_config)
        parse_data = ParseData(mode_config, '')
        npu_df, bench_df = parse_data.parse(file_list)
        result = comparator.compare_statistics(npu_df, bench_df)
        o_data = [
            ['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
             'torch.float32', 'torch.float32', '[2, 2]', '[2, 2]', 'False', 'False',
             '0', '0', '0', '0', '0', '0', '0', '0', 2, 0, 1, 1, 2, 0, 1, 1,
             True, '', '', ['File'], 'input', 'Functional.linear.0.forward'
             ]
        ]
        columns = CompareConst.SUMMARY_COMPARE_RESULT_HEADER + ['NPU_Stack_Info'] + ['state', 'api_origin_name']
        o_result = pd.DataFrame(o_data, columns=columns, dtype=object)
        self.assertTrue(np.array_equal(result.to_numpy(), o_result.to_numpy()))


class TestParseData(unittest.TestCase):

    def setUp(self):
        os.makedirs(base_dir, mode=0o750, exist_ok=True)
        generate_dump_json(base_dir)
        generate_dump_json_md5(base_dir)
        generate_stack_json(base_dir)

        self.lock = threading.Lock()

        mode_config = ModeConfig(stack_mode=True)
        self.parser = ParseData(mode_config, "rank0")

    def tearDown(self):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

    def test_parse(self):
        file_list = [os.path.join(base_dir, 'dump.json'), os.path.join(base_dir, 'dump.json'),
                     os.path.join(base_dir, 'stack.json')]

        stack_mode = True
        mode_config = ModeConfig(stack_mode=stack_mode)
        parse_data = ParseData(mode_config, 'rank0')
        npu_df, bench_df = parse_data.parse(file_list)

        target_df = pd.DataFrame(
            [
                ['Functional.linear.0.forward.input.0', 'torch.float32', [2, 2], [2, 0, 1, 1], ['File'], 'input',
                 'Functional.linear.0.forward', 'False',
                 'forward', '0.forward', 'Functional.linear', 0, 0, '.input.0',]
            ],
            columns=[
                'op_name', 'dtype', 'shape', 'summary', 'stack_info', 'state',
                'api_origin_name', 'requires_grad',
                'direction', 'call_direction', 'op_no_number', 'forward_call_order', 'backward_call_order', 'suffix'
            ]
        )
        self.assertTrue(npu_df.equals(target_df))
        self.assertTrue(bench_df.equals(target_df))

    def test_gen_data_df_summary(self):
        npu_json_path = os.path.join(base_dir, 'dump.json')
        stack_json_path = os.path.join(base_dir, 'stack.json')
        npu_json_data = load_json(npu_json_path)
        stack_json_data = load_json(stack_json_path)

        stack_mode = True
        mode_config = ModeConfig(stack_mode=stack_mode)
        parse_data = ParseData(mode_config, 'rank0')
        npu_df = parse_data.gen_data_df(npu_json_data, stack_json_data, 'NPU')

        target_df = pd.DataFrame(
            [
                ['Functional.linear.0.forward.input.0', 'torch.float32', [2, 2], [2, 0, 1, 1], ['File'], 'input',
                 'Functional.linear.0.forward', 'False',
                 'forward', '0.forward', 'Functional.linear', 0, 0, '.input.0',]
            ],
            columns=[
                'op_name', 'dtype', 'shape', 'summary', 'stack_info', 'state',
                'api_origin_name', 'requires_grad',
                'direction', 'call_direction', 'op_no_number', 'forward_call_order', 'backward_call_order', 'suffix'
            ]
        )
        self.assertTrue(npu_df.equals(target_df))

    def test_gen_data_df_all(self):
        npu_json_path = os.path.join(base_dir, 'dump.json')
        stack_json_path = os.path.join(base_dir, 'stack.json')
        npu_json_data = load_json(npu_json_path)
        stack_json_data = load_json(stack_json_path)

        stack_mode = True
        mode_config = ModeConfig(stack_mode=stack_mode, dump_mode=Const.ALL)
        parse_data = ParseData(mode_config, 'rank0')
        npu_df = parse_data.gen_data_df(npu_json_data, stack_json_data, 'NPU')

        target_df = pd.DataFrame(
            [
                ['Functional.linear.0.forward.input.0', 'torch.float32', [2, 2], [2, 0, 1, 1], ['File'], 'input',
                 'Functional.linear.0.forward', 'False',
                 'forward', '0.forward', 'Functional.linear', 0, 0, '.input.0',
                 'Functional.linear.0.forward.input.0.pt']
            ],
            columns=[
                'op_name', 'dtype', 'shape', 'summary', 'stack_info', 'state',
                'api_origin_name', 'requires_grad',
                'direction', 'call_direction', 'op_no_number', 'forward_call_order', 'backward_call_order', 'suffix',
                'data_name'
            ]
        )
        self.assertTrue(npu_df.equals(target_df))

    def test_gen_data_df_md5(self):
        npu_json_path = os.path.join(base_dir, 'dump_md5.json')
        stack_json_path = os.path.join(base_dir, 'stack.json')
        npu_json_data = load_json(npu_json_path)
        stack_json_data = load_json(stack_json_path)

        stack_mode = True
        mode_config = ModeConfig(stack_mode=stack_mode, dump_mode=Const.MD5)
        parse_data = ParseData(mode_config, 'rank0')
        npu_df = parse_data.gen_data_df(npu_json_data, stack_json_data, 'NPU')

        target_df = pd.DataFrame(
            [
                ['Functional.linear.0.forward.input.0', 'torch.float32', [2, 2], [2, 0, 1, 1], ['File'], 'input',
                 'Functional.linear.0.forward', 'False',
                 'forward', '0.forward', 'Functional.linear', 0, 0, '.input.0',
                 123456]
            ],
            columns=[
                'op_name', 'dtype', 'shape', 'summary', 'stack_info', 'state',
                'api_origin_name', 'requires_grad',
                'direction', 'call_direction', 'op_no_number', 'forward_call_order', 'backward_call_order', 'suffix',
                'md5'
            ]
        )
        self.assertTrue(npu_df.equals(target_df))

    def test_gen_merge_list(self):
        npu_json_path = os.path.join(base_dir, 'dump.json')
        stack_json_path = os.path.join(base_dir, 'stack.json')
        npu_json_data = load_json(npu_json_path)
        stack_json_data = load_json(stack_json_path)

        stack_mode = True
        mode_config = ModeConfig(stack_mode=stack_mode)
        parse_data = ParseData(mode_config, 'rank0')
        merge_list = parse_data.gen_merge_list(npu_json_data, 'Functional.linear.0.forward', stack_json_data)

        target_merge_list = [
            {
                'full_op_name': 'Functional.linear.0.forward.input.0',
                'type': 'torch.Tensor',
                'dtype': 'torch.float32',
                'shape': [2, 2],
                'requires_grad': 'False',
                'Max': 2,
                'Min': 0,
                'Mean': 1,
                'Norm': 1,
                'md5': '00000000',
                'data_name': 'Functional.linear.0.forward.input.0.pt',
                'state': 'input'
            },
            {
                'full_op_name': 'Functional.linear.0.forward',
                'full_info': ['File']
            }
        ]
        self.assertEqual(merge_list, target_merge_list)

    def test_get_direction_and_call_direction_backward_at_last(self):
        direction, call_direction = self.parser.get_direction_and_call_direction(
            "conv.1.backward"
        )
        self.assertEqual(direction, Const.BACKWARD)
        self.assertEqual(call_direction, "1.backward")

    def test_get_direction_and_call_direction_backward_at_second_last(self):
        direction, call_direction = self.parser.get_direction_and_call_direction(
            "conv.backward.1"
        )
        self.assertEqual(direction, Const.BACKWARD)
        self.assertEqual(call_direction, "backward.1")

    def test_get_direction_and_call_direction_forward_at_last(self):
        direction, call_direction = self.parser.get_direction_and_call_direction(
            "matmul.2.forward"
        )
        self.assertEqual(direction, Const.FORWARD)
        self.assertEqual(call_direction, "2.forward")

    def test_get_direction_and_call_direction_forward_at_second_last(self):
        direction, call_direction = self.parser.get_direction_and_call_direction(
            "matmul.forward.2"
        )
        self.assertEqual(direction, Const.FORWARD)
        self.assertEqual(call_direction, "forward.2")

    def test_get_direction_and_call_direction_less_than_two_parts(self):
        direction, call_direction = self.parser.get_direction_and_call_direction(
            "backward"
        )
        self.assertEqual((direction, call_direction), (None, None))

    def test_get_direction_and_call_direction_no_direction_keyword(self):
        direction, call_direction = self.parser.get_direction_and_call_direction(
            "conv.1.2"
        )
        self.assertEqual((direction, call_direction), (None, None))

    def test_get_direction_and_call_direction_direction_not_in_last_two(self):
        direction, call_direction = self.parser.get_direction_and_call_direction(
            "backward.conv.1"
        )
        self.assertEqual((direction, call_direction), (None, None))

    def test_get_op_no_number__three_parts(self):
        result = self.parser.get_op_no_number("conv.1.forward")
        self.assertEqual(result, "conv")

    def test_get_op_no_number__more_than_three_parts(self):
        result = self.parser.get_op_no_number("conv.bn.relu.1.forward")
        self.assertEqual(result, "conv.bn.relu")

    def test_get_op_no_number__exact_two_parts(self):
        result = self.parser.get_op_no_number("conv.1")
        self.assertEqual(result, "")

    def test_get_op_no_number__single_part(self):
        result = self.parser.get_op_no_number("conv")
        self.assertEqual(result, "")

    def test_get_op_no_number__empty_string(self):
        result = self.parser.get_op_no_number("")
        self.assertEqual(result, "")


class TestProcessDf(unittest.TestCase):

    def setUp(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        self.process_df = ProcessDf(mode_config, mapping_config, mapping_dict)

    def test_get_api_name_success(self):
        api_list = ['Functional', 'linear', '0', 'forward', 'input', '0']

        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)
        api_name = process_df.get_api_name(api_list)

        target_api_name = 'Functional.linear'
        self.assertEqual(api_name, target_api_name)

    @patch('msprobe.core.compare.acc_compare.logger')
    def test_get_api_name_index_error(self, mock_logger):
        api_list = ['Functional']
        with self.assertRaises(CompareException) as context:
            mode_config = ModeConfig()
            mapping_config = MappingConfig()
            mapping_dict = MappingDict(mapping_config)
            process_df = ProcessDf(mode_config, mapping_config, mapping_dict)
            api_name = process_df.get_api_name(api_list)
        self.assertEqual(context.exception.code, CompareException.INDEX_OUT_OF_BOUNDS_ERROR)
        mock_logger.error.assert_called_once_with('Failed to retrieve API name, please check if the dump data is reasonable')

    def test_process_compare_key_and_shape(self):
        npu_df_o = bench_df_o = pd.DataFrame(
            [['Functional.linear.0.forward.input.0', 'torch.float32', [2, 2], [2, 0, 1, 1], ['File'], 'Functional.linear.0.forward.input.0']],
            columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info', 'op_name_update']
        )

        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)
        npu_df, bench_df = process_df.process_compare_key_and_shape(npu_df_o, bench_df_o)

        target_df = pd.DataFrame(
            [['Functional.linear.0.forward.input.0', 'torch.float32', [2, 2], [2, 0, 1, 1], ['File'], 'Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0', [2, 2]]],
            columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info', 'op_name_update', 'compare_key', 'compare_shape']
        )
        self.assertTrue(npu_df.equals(target_df))
        self.assertTrue(bench_df.equals(target_df))

    def test_process_internal_api_mapping(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)

        # mint to torch
        npu_op_name = 'Mint.mean.0.input.0'
        target_name = 'Torch.mean.0.input.0'
        name = process_df.process_internal_api_mapping(npu_op_name)
        self.assertEqual(name, target_name)

        # mintfunctional to functional
        npu_op_name = 'MintFunctional.mean.0.input.0'
        target_name = 'Functional.mean.0.input.0'
        name = process_df.process_internal_api_mapping(npu_op_name)
        self.assertEqual(name, target_name)

        # inner mapping exists
        npu_op_name = 'Functional.abs.0.input.0'
        mapping_dict.ms_to_pt_mapping = {'Functional.abs': 'Torch.abs'}
        target_name = 'Torch.abs.0.input.0'
        name = process_df.process_internal_api_mapping(npu_op_name)
        self.assertEqual(name, target_name)

        # inner mapping not found
        npu_op_name = 'Functional.abs.0.input.0'
        mapping_dict.ms_to_pt_mapping = {}
        target_name = 'Functional.abs.0.input.0'
        name = process_df.process_internal_api_mapping(npu_op_name)
        self.assertEqual(name, target_name)

    def test_modify_compare_data_with_user_mapping(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)
        mapping_dict.api_mapping_dict = [{
            'ms_api': 'Functional.conv2d',
            'pt_api': 'Torch.conv2d',
            'ms_args': [0],
            'pt_args': [0]
        }]

        npu_df = pd.DataFrame([
            ['Functional.conv2d.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info', 'Functional.conv2d.0.forward.input.0', 'input', 'Functional.conv2d.0.forward'],
            ['Functional.amax.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info', 'Functional.amax.0.forward.input.0', 'input', 'Functional.amax.0.forward']
        ], columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info', 'compare_key', 'state', 'api_origin_name'])
        bench_df = pd.DataFrame([
            ['Torch.conv2d.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info', 'Torch.conv2d.0.forward.input.0', 'input', 'Functional.conv2d.0.forward'],
            ['Torch.amax.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info', 'Torch.amax.0.forward.input.0', 'input', 'Functional.amax.0.forward']
        ], columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info', 'compare_key', 'state', 'api_origin_name'])

        process_df.modify_compare_data_with_user_mapping(npu_df, bench_df)

    def test_get_api_indices_dict(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)

        op_name_df = pd.DataFrame([
            ['Functional.conv2d.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info', 'Functional.conv2d.0.forward.input.0'],
            ['Functional.amax.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info', 'Functional.amax.0.forward.input.0']
        ], columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info', 'compare_key'])

        api_indices_dict = process_df.get_api_indices_dict(op_name_df)
        expected = {
            'Functional.conv2d': [0],
            'Functional.amax': [1]
        }
        self.assertEqual(api_indices_dict, expected)

    def test_process_cell_mapping(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)

        # not name
        npu_op_name = None
        name = process_df.process_cell_mapping(npu_op_name)
        self.assertEqual(name, CompareConst.N_A)

        # not params_grad
        npu_op_name = 'MintFunctional.embedding.0.input.0'
        name = process_df.process_cell_mapping(npu_op_name)
        self.assertEqual(name, CompareConst.N_A)

        # default replace
        npu_op_name = 'Cell.network_with_loss.module.GPTModel.forward.1.input.0'
        name = process_df.process_cell_mapping(npu_op_name)
        self.assertEqual(name, 'Module.network_with_loss.module.GPTModel.forward.1.input.0')

        # mapping_dict
        npu_op_name = 'Cell.fc1.Dense.forward.0.input.0'
        mapping_dict.cell_mapping_dict = {'fc1.Dense': 'module.name'}
        name = process_df.process_cell_mapping(npu_op_name)
        self.assertEqual(name, 'Module.module.name.forward.0.input.0')

    def test_process_data_mapping(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)

        npu_op_name = 'Functional.flash_attention_score.4.forward.input.0'
        mapping_dict.data_mapping_dict = {'Functional.flash_attention_score.4.forward.input.0': 'NPU.npu_fusion_attention.4.forward.input.0'}
        name = process_df.process_data_mapping(npu_op_name)
        self.assertEqual(name, 'NPU.npu_fusion_attention.4.forward.input.0')

    def test_update_backward_call_digit_at_head(self):
        cmp_df = pd.DataFrame({
            CompareConst.OP_NAME: ["conv.1.backward.fp32"],
            Const.CALL_DIRECTION: ["1.backward"],
            Const.DIRECTION: [Const.BACKWARD],
            Const.BACKWARD_CALL_ORDER: [3],
            Const.OP_NO_NUMBER: ["conv"],
            Const.SUFFIX: [".fp32"],
        })

        result = self.process_df.update_backward_call(cmp_df)

        self.assertEqual(result.loc[0, Const.CALL_DIRECTION], "3.backward")
        self.assertEqual(
            result.loc[0, CompareConst.OP_NAME_UPDATE],
            "conv.3.backward.fp32"
        )

    def test_update_backward_call_digit_at_tail(self):
        cmp_df = pd.DataFrame({
            CompareConst.OP_NAME: ["matmul.backward.2.fp16"],
            Const.CALL_DIRECTION: ["backward.2"],
            Const.DIRECTION: [Const.BACKWARD],
            Const.BACKWARD_CALL_ORDER: [5],
            Const.OP_NO_NUMBER: ["matmul"],
            Const.SUFFIX: [".fp16"],
        })

        result = self.process_df.update_backward_call(cmp_df)

        self.assertEqual(result.loc[0, Const.CALL_DIRECTION], "backward.5")
        self.assertEqual(
            result.loc[0, CompareConst.OP_NAME_UPDATE],
            "matmul.backward.5.fp16"
        )

    def test_update_backward_call_non_backward_should_not_change(self):
        cmp_df = pd.DataFrame({
            CompareConst.OP_NAME: ["add.1.forward.fp32"],
            Const.CALL_DIRECTION: ["1.forward"],
            Const.DIRECTION: [Const.FORWARD],
            Const.BACKWARD_CALL_ORDER: [7],
            Const.OP_NO_NUMBER: ["add"],
            Const.SUFFIX: [".fp32"],
        })

        result = self.process_df.update_backward_call(cmp_df)

        self.assertEqual(result.loc[0, Const.CALL_DIRECTION], "1.forward")
        self.assertEqual(
            result.loc[0, CompareConst.OP_NAME_UPDATE],
            "add.1.forward.fp32"
        )


class TestMatch(unittest.TestCase):

    def test_put_unmatched_in_table(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=False)

        match_result = pd.DataFrame(columns=CompareConst.MATCH_RESULT_COLUMNS)
        npu_op_item = pd.Series(['op', 'float32', [1, 2], 'summary', 'stack_info',
                                 'input', 'op_origin', 'False', 'data_name', 'op', [1, 2]],
                                index=['op_name_x', 'dtype_x', 'shape_x', 'summary_x', 'stack_info_x',
                                       'state_x', 'api_origin_name_x', 'data_name_x', 'requires_grad_x',
                                       'compare_key', 'compare_shape']
                                )
        match_result = match.put_unmatched_in_table(match_result, npu_op_item)
        target_match_result = pd.DataFrame([['op', 'float32', [1, 2], 'summary', 'stack_info',
                                             'input', 'op_origin', 'False', 'data_name', 'op', [1, 2],
                                             'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']],
                                           columns=CompareConst.MATCH_RESULT_COLUMNS)
        self.assertTrue(match_result.equals(target_match_result))

    def test_put_matched_in_table(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=False)

        match_result = pd.DataFrame(columns=CompareConst.MATCH_RESULT_COLUMNS)
        npu_op_item = pd.Series(['op', 'float32', [1, 2], 'summary', 'stack_info',
                                 'input', 'op_origin', 'False', 'data_name', 'op', [1, 2]],
                                index=['op_name_x', 'dtype_x', 'shape_x', 'summary_x', 'stack_info_x',
                                       'state_x', 'api_origin_name_x', 'requires_grad_x', 'data_name_x',
                                       'compare_key', 'compare_shape']
                                )
        bench_op_item = pd.Series(['op', 'float32', [1, 2], 'summary', 'stack_info',
                                   'input', 'op_origin', 'False', 'data_name', 'op', [1, 2]],
                                  index=['op_name_y', 'dtype_y', 'shape_y', 'summary_y', 'stack_info_y',
                                         'state_y', 'api_origin_name_y', 'requires_grad_y', 'data_name_y',
                                         'compare_key', 'compare_shape']
                                  )
        match_result = match.put_matched_in_table(match_result, npu_op_item, bench_op_item)
        target_match_result = pd.DataFrame([['op', 'float32', [1, 2], 'summary', 'stack_info',
                                             'input', 'op_origin', 'False', 'data_name', 'op', [1, 2],
                                             'op', 'float32', [1, 2], 'summary', 'stack_info',
                                             'input', 'op_origin', 'False', 'data_name']],
                                           columns=CompareConst.MATCH_RESULT_COLUMNS)
        self.assertTrue(match_result.equals(target_match_result))

    def test_rename_api(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=False)

        op_name_1 = 'Functional.linear.0.forward.input.0'
        result_1 = match.rename_api(op_name_1)
        self.assertTrue(result_1, 'Functional.linear.input.0')

        op_name_2 = 'Functional.linear.0.backward.input.0'
        result_2 = match.rename_api(op_name_2)
        self.assertTrue(result_2, 'Functional.linear.input.0')

        op_name_3 = 'Functional.linear.0.x.input.0'
        result_3 = match.rename_api(op_name_3)
        self.assertTrue(result_3, 'Functional.linear.0.x.input.0')

    def test_check_op_item(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=False)

        npu_op_item = pd.Series(['op', 'float32', [1, 2], 'summary', 'stack_info', 'data_name', 'Functional.linear.0.forward.input.0', [1, 2]],
                                index=['op_name_x', 'dtype_x', 'shape_x', 'summary_x', 'stack_info_x', 'data_name_x',
                                       'compare_key', 'compare_shape']
                                )
        bench_op_item = pd.Series(['op', 'float32', [1, 2], 'summary', 'stack_info', 'data_name', 'Functional.linear.1.forward.input.0', [1, 2]],
                                  index=['op_name_y', 'dtype_y', 'shape_y', 'summary_y', 'stack_info_y', 'data_name_y',
                                         'compare_key', 'compare_shape']
                                  )
        result = match.check_op_item(npu_op_item, bench_op_item)
        self.assertTrue(result)

    def test_process_fuzzy_match(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=False)

        npu_df = pd.DataFrame([
            ['Functional.conv2d.3.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
             'input', 'Functional.conv2d.3.forward', 'True', 'Functional.conv2d.3.forward.input.0.pt',
             'Functional.conv2d.3.forward.input.0', [1, 2]],
            ['Functional.amax.1.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
             'input', 'Functional.amax.1.forward', 'True', 'Functional.amax.0.forward.input.0.pt',
             'Functional.amax.1.forward.input.0', [1, 2]]
        ], columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info',
                    'state', 'api_origin_name', 'requires_grad', 'data_name',
                    'compare_key', 'compare_shape'])
        bench_df = pd.DataFrame([
            ['Functional.conv2d.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
             'input', 'Functional.conv2d.0.forward', 'True', 'Functional.conv2d.0.forward.input.0.pt',
             'Functional.conv2d.0.forward.input.0', [1, 2]],
            ['Functional.amax.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
             'input', 'Functional.amax.0.forward', 'True', 'Functional.amax.0.forward.input.0.pt',
             'Functional.amax.0.forward.input.0', [1, 2]]
        ], columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info',
                    'state', 'api_origin_name', 'requires_grad', 'data_name', 'compare_key', 'compare_shape'])

        match_result = match.process_fuzzy_match(npu_df, bench_df)
        expected = pd.DataFrame(
            [
                ['Functional.conv2d.3.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
                 'input', 'Functional.conv2d.3.forward', 'True', 'Functional.conv2d.3.forward.input.0.pt',
                 'Functional.conv2d.3.forward.input.0', [1, 2],
                 'Functional.conv2d.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
                 'input', 'Functional.conv2d.0.forward', 'True', 'Functional.conv2d.0.forward.input.0.pt'],
                ['Functional.amax.1.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
                 'input', 'Functional.amax.1.forward', 'True', 'Functional.amax.0.forward.input.0.pt',
                 'Functional.amax.1.forward.input.0', [1, 2],
                 'Functional.amax.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
                 'input', 'Functional.amax.0.forward', 'True', 'Functional.amax.0.forward.input.0.pt']
            ]
        , columns=CompareConst.MATCH_RESULT_COLUMNS)

        self.assertTrue(match_result.equals(expected))

    def test_match_op_both_last_element(self):
        config_dict = {
            'stack_mode': False,
            'auto_analyze': True,
            'fuzzy_match': False,
            'dump_mode': Const.SUMMARY,
        }
        mode_config = ModeConfig(**config_dict)
        mapping_config = MappingConfig()

        match = Match(mode_config, mapping_config, cross_frame=False)
        a, b = match.match_op([npu_op_item_fuzzy], [bench_op_item_fuzzy])
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_match_op_only_npu_last_element(self):
        config_dict = {
            'stack_mode': False,
            'auto_analyze': True,
            'fuzzy_match': False,
            'dump_mode': Const.SUMMARY,
        }
        mode_config = ModeConfig(**config_dict)
        mapping_config = MappingConfig()

        match = Match(mode_config, mapping_config, cross_frame=False)
        a, b = match.match_op([npu_op_item_fuzzy], [bench_op_item_fuzzy, 1])
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_match_op_only_bench_last_element(self):
        config_dict = {
            'stack_mode': False,
            'auto_analyze': True,
            'fuzzy_match': False,
            'dump_mode': Const.SUMMARY,
        }
        mode_config = ModeConfig(**config_dict)
        mapping_config = MappingConfig()

        match = Match(mode_config, mapping_config, cross_frame=False)
        a, b = match.match_op([npu_op_item_fuzzy, npu_op_item_data_fuzzy_2], [bench_op_item_fuzzy])
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_gen_dtype_condition(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=True)

        # data mapping
        mapping_config.data_mapping = True
        match_result = pd.DataFrame([1, 2, 3])
        result = match.gen_dtype_condition(match_result)
        expected = pd.Series([True, True, True])
        self.assertTrue(result.equals(expected))

        # normal
        mapping_config.data_mapping = None
        match_result = pd.DataFrame([['Float16', 'Float32'], ['torch.float32', 'torch.bfloat16']], columns=['dtype_x', 'dtype_y'])
        result = match.gen_dtype_condition(match_result)
        expected = pd.Series([True, True])
        self.assertTrue(result.equals(expected))

    def test_process_cross_frame_dtype(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=True)

        dtype_o = pd.Series(['Int8', 'Float16', 'torch.bool', 'Complex64', 'unknown'])
        dtype = match.process_cross_frame_dtype(dtype_o)
        self.assertTrue(dtype.equals(pd.Series(['int', 'float', 'bool', 'complex', 'unknown'])))


class TestCreateTable(unittest.TestCase):

    def test_process_data_name(self):
        mode_config = ModeConfig()
        create_table = CreateTable(mode_config)

        data = {
            'data_name_x': ['A', 'B', 'C'],
            'data_name_y': ['X', 'Y', 'Z']
        }
        result_o = pd.DataFrame(data)
        result = create_table.process_data_name(result_o)
        target_data = {
            'data_name_x': [['A', 'X'], ['B', 'Y'], ['C', 'Z']],
            'data_name_y': ['X', 'Y', 'Z']
        }
        target_result = pd.DataFrame(target_data)
        self.assertTrue(result.equals(target_result))

    def test_set_summary(self):
        mode_config = ModeConfig()
        create_table = CreateTable(mode_config)

        # all nan
        result = create_table.set_summary(['nan', 'NaN', 'nAn'])
        expected = [CompareConst.NAN, CompareConst.NAN, CompareConst.NAN]
        self.assertEqual(result, expected)

        # mixed values
        result = create_table.set_summary([1, 'nan', 2.0, 'NaN'])
        expected = [1, CompareConst.NAN, 2.0, CompareConst.NAN]
        self.assertEqual(result, expected)

        # NA case
        result = create_table.set_summary(CompareConst.N_A)
        expected = [CompareConst.N_A, CompareConst.N_A, CompareConst.N_A, CompareConst.N_A]
        self.assertEqual(result, expected)

        # empty input
        result = create_table.set_summary([])
        expected = []
        self.assertEqual(result, expected)


class TestCalcStatsDiff(unittest.TestCase):

    def setUp(self):
        """
        每个 test 都会创建一个新的实例
        """
        self.mode_config = ModeConfig()  # 如果构造需要参数，这里补
        self.calc = CalcStatsDiff(self.mode_config)

    # ===============================
    # is_same_value
    # ===============================
    def test_is_same_value(self):
        a = pd.Series(["1", "2", "nan", "INF"])
        b = pd.Series(["1", "3", "NaN", "INF"])

        result = self.calc.is_same_value(a, b)
        expected = pd.Series([True, False, False, True])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # is_number
    # ===============================
    def test_is_number(self):
        s = pd.Series(["1", "2.5", "abc", np.nan, "inf", True, False, 'false', 'True'])

        result = self.calc.is_number(s)
        expected = pd.Series([True, True, False, False, True, False, False, False, False])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # is_nan
    # ===============================
    def test_is_nan(self):
        s = pd.Series(["nan", "NaN", np.nan, "1", "none"])

        result = self.calc.is_nan(s)
        expected = pd.Series([True, True, True, False, False])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # is_inf
    # ===============================
    def test_is_inf(self):
        s = pd.Series(["inf", "INF", np.inf, "-inf", 1])

        result = self.calc.is_inf(s)
        expected = pd.Series([True, True, True, False, False])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # is_neg_inf
    # ===============================
    def test_is_neg_inf(self):
        s = pd.Series(["-inf", "-INF", -np.inf, "inf", 1])

        result = self.calc.is_neg_inf(s)
        expected = pd.Series([True, True, True, False, False])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # is_device
    # ===============================
    def test_is_device(self):
        s = pd.Series(["npu:0", "CPU", "cuda:1", "gpu", None])

        result = self.calc.is_device(s)
        expected = pd.Series([True, True, True, False, False])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # is_na
    # ===============================
    def test_is_na(self):
        s = pd.Series([CompareConst.N_A, "N/A", "na", 1])

        result = self.calc.is_na(s)
        expected = pd.Series([True, True, False, False])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # rule_num_num
    # ===============================
    def test_rule_num_num(self):
        npu = pd.Series([10.0, 5.0, 3.0])
        bench = pd.Series([5.0, 0.0, 3.0])

        diff, rel = self.calc.rule_num_num(npu, bench)

        expected_diff = pd.Series([5.0, 5.0, 0.0])
        expected_rel = pd.Series(["100.0%", CompareConst.INF, "0.0%"])

        pd.testing.assert_series_equal(diff, expected_diff)
        pd.testing.assert_series_equal(rel, expected_rel)

    # ===============================
    # static_diff / DEFAULT_RULE
    # ===============================
    def test_static_diff(self):
        diff, rel = self.calc.static_diff("A", "B")
        self.assertEqual(diff, "A")
        self.assertEqual(rel, "B")

    def test_default_rule(self):
        diff, rel = self.calc.DEFAULT_RULE
        self.assertEqual(diff, CompareConst.N_A)
        self.assertEqual(rel, CompareConst.N_A)

    # ===============================
    # get_number
    # ===============================
    def test_get_number(self):
        mode_config = ModeConfig()
        calc_stats_diff = CalcStatsDiff(mode_config)

        series = pd.Series([1, '2', 3.5, 'text', None])
        result = calc_stats_diff.get_number(series)
        expected = pd.Series([1, 2, 3.5, float('nan'), float('nan')])
        self.assertTrue(result.equals(expected))


class TestCalcStatsDiffBuildRules(unittest.TestCase):

    def setUp(self):
        self.calc = CalcStatsDiff(ModeConfig())

    def test_rules_initialized(self):
        self.assertIsInstance(self.calc.rules, dict)
        self.assertTrue(len(self.calc.rules) > 0)

    def test_num_num_rule(self):
        rule = self.calc.rules.get((ValType.NUM, ValType.NUM))
        self.assertTrue(callable(rule))
        self.assertEqual(rule, self.calc.rule_num_num)

    def test_nan_rules(self):
        nan_rule = self.calc.static_diff(CompareConst.NAN)

        for t in (ValType.NUM, ValType.INF, ValType.NEG_INF, ValType.NAN):
            self.assertEqual(self.calc.rules[(ValType.NAN, t)], nan_rule)
            self.assertEqual(self.calc.rules[(t, ValType.NAN)], nan_rule)

    def test_inf_rules(self):
        pos_inf = self.calc.static_diff(CompareConst.INF)
        neg_inf = self.calc.static_diff(CompareConst.NEG_INF)

        self.assertEqual(self.calc.rules[(ValType.INF, ValType.NUM)], pos_inf)
        self.assertEqual(self.calc.rules[(ValType.NEG_INF, ValType.NUM)], neg_inf)
        self.assertEqual(self.calc.rules[(ValType.NUM, ValType.INF)], neg_inf)
        self.assertEqual(self.calc.rules[(ValType.NUM, ValType.NEG_INF)], pos_inf)

    def test_device_device_rule(self):
        rule = self.calc.rules[(ValType.DEVICE, ValType.DEVICE)]
        self.assertEqual(rule, self.calc.static_diff(CompareConst.N_A))


class TestCalcStatsDiffClassify(unittest.TestCase):

    def setUp(self):
        self.calc = CalcStatsDiff(ModeConfig())

    def test_classify_all_types(self):
        s = pd.Series([
            "1.23",        # NUM
            "nan",         # NAN
            np.nan,        # NAN
            "inf",         # INF
            "-inf",        # NEG_INF
            "npu:0",       # DEVICE
            CompareConst.N_A,  # NA
            "abc",         # OTHER
            True,          # OTHER
            False,         # OTHER
            "false",       # OTHER
            "True"         # OTHER
        ])

        result = self.calc.classify(s)

        expected = pd.Series([
            ValType.NUM,
            ValType.NAN,
            ValType.NAN,
            ValType.INF,
            ValType.NEG_INF,
            ValType.DEVICE,
            ValType.NA,
            ValType.OTHER,
            ValType.OTHER,
            ValType.OTHER,
            ValType.OTHER,
            ValType.OTHER,
        ])

        pd.testing.assert_series_equal(result, expected)


class TestCalcStatsDiffCalcSummaryDiff(unittest.TestCase):

    def setUp(self):
        self.calc = CalcStatsDiff(ModeConfig())

    def test_calc_summary_diff_num_num(self):
        df = pd.DataFrame({
            "NPU mean":   [10.0, 5.0, 3.0],
            "Bench mean": [5.0, 5.0, 3.0],
        })

        self.calc.calc_summary_diff(df, "mean")

        self.assertIn("Mean diff", df.columns)
        self.assertIn("MeanRelativeErr", df.columns)

        # equal
        self.assertEqual(df.loc[1, "Mean diff"], '0')
        self.assertEqual(df.loc[2, "Mean diff"], '0')

        # unequal num × num
        self.assertEqual(df.loc[0, "Mean diff"], '5.0')
        self.assertEqual(df.loc[0, "MeanRelativeErr"], "100.0%")

    def test_calc_summary_diff_special_types(self):
        df = pd.DataFrame({
            "NPU mean":   ["nan", "inf", CompareConst.N_A],
            "Bench mean": [1.0,   1.0,   2.0],
        })

        self.calc.calc_summary_diff(df, "mean")

        self.assertEqual(df.loc[0, "Mean diff"], CompareConst.NAN)
        self.assertEqual(df.loc[1, "Mean diff"], CompareConst.INF)
        self.assertEqual(df.loc[2, "Mean diff"], CompareConst.N_A)

    def test_calc_summary_diff_device(self):
        df = pd.DataFrame({
            "NPU mean":   ["npu:0", "abc"],
            "Bench mean": ["cpu",   1.0],
        })

        self.calc.calc_summary_diff(df, "mean")

        self.assertEqual(df.loc[0, "Mean diff"], CompareConst.N_A)
        self.assertEqual(df.loc[1, "Mean diff"], CompareConst.DIFF_FLAG)


if __name__ == '__main__':
    unittest.main()
