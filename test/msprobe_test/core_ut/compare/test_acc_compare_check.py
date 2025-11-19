# coding=utf-8
import unittest
from unittest.mock import patch

from msprobe.core.compare.check import check_dump_json_str, check_json_key_value, valid_key_value, \
    check_stack_json_str, check_configuration_param
from msprobe.core.common.utils import CompareException
from msprobe.core.common.log import logger
from msprobe.core.compare.acc_compare import ComparisonConfig


# test_check_struct_match
npu_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
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


# test_check_type_shape_match
npu_struct = [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]), ('torch.float32', [16])]
bench_struct = [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]), ('torch.float32', [16])]


# test_check_dump_json_str
op_data = {'input_args': [
    {
        'type': '@torch.Tensor',
        'dtype': 'torch.float32',
        'shape': [2, 2],
        'Max': 0.5,
        'Min': -0.7,
        'Mean': 0.2,
        'Norm': 0.9,
        'requires_grad': True,
        'data_name': 'Function.linear.0.forward.output.0.pt'
    }
]}


# test_check_json_key_value
input_output = {
    'type': '@torch.Tensor',
    'dtype': 'torch.float32',
    'shape': [2, 2],
    'Max': 0.5,
    'Min': -0.7,
    'Mean': 0.2,
    'Norm': 0.9,
    'requires_grad': True,
    'data_name': 'Function.linear.0.forward.output.0.pt'}


op_name = 'Functional.conv2d.0.backward.input.0'


class TestUtilsMethods(unittest.TestCase):
    def test_check_dump_json_str(self):
        with self.assertRaises(CompareException) as context:
            check_dump_json_str(op_data, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_json_key_value(self):
        with self.assertRaises(CompareException) as context:
            check_json_key_value(input_output, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_json_key_value_max_depth(self):
        result = check_json_key_value(input_output, op_name, depth=401)
        self.assertEqual(result, None)

    def test_valid_key_value_type_shape(self):
        key = 'shape'
        value = 'abc'
        with self.assertRaises(CompareException) as context:
            valid_key_value(key, value, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_OBJECT_TYPE_ERROR)

    def test_valid_key_value_type_requires_grad(self):
        key = 'requires_grad'
        value = 'abc'
        with self.assertRaises(CompareException) as context:
            valid_key_value(key, value, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_OBJECT_TYPE_ERROR)

    def test_check_stack_json_str_type_stack_info(self):
        stack_info = 'File'
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_OBJECT_TYPE_ERROR)

    def test_check_stack_json_str_2(self):
        stack_info = ['=File', 'File']
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_stack_json_str_3(self):
        stack_info = ['+File', 'File']
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_stack_json_str_4(self):
        stack_info = ['-File', 'File']
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_stack_json_str_5(self):
        stack_info = ['@File', 'File']
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)
