# coding=utf-8
import os
import copy
import shutil
import tempfile
import unittest
from unittest.mock import patch, DEFAULT
from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import *
from msprobe.core.common.file_utils import get_json_contents, create_directory, save_json, write_csv
from msprobe.core.common.exceptions import FileCheckException
from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check_utils import UtDataInfo, exec_api

base_dir = os.path.dirname(os.path.realpath(__file__))
forward_file = os.path.join(base_dir, "forward.json")
forward_content = get_json_contents(forward_file)
for api_full_name, api_info_dict in forward_content.items():
    api_full_name = api_full_name
    api_info_dict = api_info_dict


class Args:
    def __init__(self, config_path=None, api_info_path=None, out_path=None, result_csv_path=None):
        self.config_path = config_path
        self.api_info_path = api_info_path
        self.out_path = out_path
        self.result_csv_path = result_csv_path


class TestFileCheck(unittest.TestCase):
    def setUp(self):
        src_path = 'temp_path'
        create_directory(src_path)
        dst_path = 'soft_link'
        os.symlink(src_path, dst_path)
        self.hard_path = os.path.abspath(src_path)
        self.soft_path = os.path.abspath(dst_path)
        json_path = os.path.join(self.hard_path, 'test.json')
        json_data = {'key': 'value'}
        save_json(json_path, json_data)
        self.hard_json_path = json_path
        soft_json_path = 'soft.json'
        os.symlink(json_path, soft_json_path)
        self.soft_json_path = os.path.abspath(soft_json_path)
        csv_path = os.path.join(self.hard_path, 'test.csv')
        csv_data = [['1', '2', '3']]
        write_csv(csv_data, csv_path)
        soft_csv_path = 'soft.csv'
        os.symlink(csv_path, soft_csv_path)
        self.csv_path = os.path.abspath(soft_csv_path)
        self.empty_path = "empty_path"

    def tearDown(self):
        os.unlink(self.soft_json_path)
        os.unlink(self.csv_path)
        os.unlink(self.soft_path)
        for file in os.listdir(self.hard_path):
            os.remove(os.path.join(self.hard_path, file))
        os.rmdir(self.hard_path)

    def test_config_path_soft_link_check(self):
        args = Args(config_path=self.soft_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path)

        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_api_info_path_soft_link_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.soft_json_path, out_path=self.hard_path)

        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_out_path_soft_link_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.soft_path)

        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_result_csv_path_soft_link_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path,
                    result_csv_path=self.csv_path)

        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_config_path_empty_check(self):
        args = Args(config_path=self.empty_path, api_info_path=self.hard_json_path, out_path=self.hard_path)

        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_api_info_path_empty_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.empty_path, out_path=self.hard_path)

        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_out_path_empty_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.empty_path)
        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_result_csv_path_empty_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path,
                    result_csv_path=self.empty_path)
        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_config_path_invalid_check(self):
        args = Args(config_path=123, api_info_path=self.hard_json_path, out_path=self.hard_path)
        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_api_info_path_invalid_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path="123", out_path=self.hard_path)
        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_out_path_invalid_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=123)
        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_result_csv_path_invalid_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path,
                    result_csv_path=123)
        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)


class TestRunUtMethods(unittest.TestCase):
    def test_exec_api(self):
        api_info = copy.deepcopy(api_info_dict)

        [api_type, api_name, _, _] = api_full_name.split(".")
        args, kwargs, need_grad = get_api_info(api_info, api_name, None)
        cpu_params = generate_cpu_params(args, kwargs, True, '')
        cpu_args, cpu_kwargs = cpu_params.cpu_args, cpu_params.cpu_kwargs
        cpu_exec_params = ExecParams(api_type, api_name, Const.CPU_LOWERCASE, cpu_args, cpu_kwargs, False, None)
        out = exec_api(cpu_exec_params)
        self.assertEqual(out[0].dtype, torch.float32)
        self.assertTrue(out[0].requires_grad)
        self.assertEqual(out[0].shape, torch.Size([2048, 2, 1, 128]))

    def test_generate_device_params(self):
        mock_tensor = torch.rand([2, 2560, 24, 24], dtype=torch.float32, requires_grad=True)

        with patch.multiple('torch.Tensor',
                            to=DEFAULT,
                            clone=DEFAULT,
                            detach=DEFAULT,
                            requires_grad_=DEFAULT,
                            type_as=DEFAULT,
                            retain_grad=DEFAULT) as mocks:
            mocks['clone'].return_value = mock_tensor
            mocks['detach'].return_value = mock_tensor
            mocks['requires_grad_'].return_value = mock_tensor
            mocks['type_as'].return_value = mock_tensor
            mocks['retain_grad'].return_value = None
            mocks['to'].return_value = mock_tensor

            device_args, device_kwargs, _ = generate_device_params([mock_tensor], {'inplace': False}, True, '')
            self.assertEqual(len(device_args), 1)
            self.assertEqual(device_args[0].dtype, torch.float32)
            self.assertTrue(device_args[0].requires_grad)
            self.assertEqual(device_args[0].shape, torch.Size([2, 2560, 24, 24]))
            self.assertEqual(device_kwargs, {'inplace': False})

    def test_generate_cpu_params(self):
        api_info = copy.deepcopy(api_info_dict)
        [api_type, api_name, _, _] = api_full_name.split(".")
        args, kwargs, need_grad = get_api_info(api_info, api_name, None)
        cpu_params = generate_cpu_params(args, kwargs, True, '')
        cpu_args, cpu_kwargs = cpu_params.cpu_args, cpu_params.cpu_kwargs
        self.assertEqual(len(cpu_args), 2)
        self.assertEqual(cpu_args[0].dtype, torch.float32)
        self.assertTrue(cpu_args[0].requires_grad)
        self.assertEqual(cpu_args[0].shape, torch.Size([2048, 2, 1, 256]))
        self.assertEqual(cpu_kwargs, {'dim': -1})

    def test_UtDataInfo(self):
        data_info = UtDataInfo(None, None, None, None, None, None, None)
        self.assertIsNone(data_info.bench_grad)
        self.assertIsNone(data_info.device_grad)
        self.assertIsNone(data_info.device_output)
        self.assertIsNone(data_info.bench_output)
        self.assertIsNone(data_info.grad_in)
        self.assertIsNone(data_info.in_fwd_data_list)

    def test_blacklist_and_whitelist_filter(self):
        api_name = "test_api"
        black_list = ["test_api"]
        white_list = []
        result = blacklist_and_whitelist_filter(api_name, black_list, white_list)
        self.assertTrue(result)

        api_name = "test_api"
        black_list = []
        white_list = ["another_api"]
        result = blacklist_and_whitelist_filter(api_name, black_list, white_list)
        self.assertTrue(result)

        api_name = "test_api"
        black_list = ["test_api"]
        white_list = ["test_api"]
        result = blacklist_and_whitelist_filter(api_name, black_list, white_list)
        self.assertTrue(result)

        api_name = "test_api"
        black_list = []
        white_list = ["test_api"]
        result = blacklist_and_whitelist_filter(api_name, black_list, white_list)
        self.assertFalse(result)

    def test_supported_api(self):
        api_name = "torch.matmul"
        result = is_unsupported_api(api_name)
        self.assertFalse(result)

        api_name = "Distributed.all_reduce"
        result = is_unsupported_api(api_name)
        self.assertTrue(result)

    def test_no_backward(self):
        grad_index = None
        out = (1, 2, 3)
        result = need_to_backward(grad_index, out)
        self.assertFalse(result)

        grad_index = 0
        out = 42
        result = need_to_backward(grad_index, out)
        self.assertTrue(result)

    def test_check_need_grad_given_out_kwarg_then_return_false(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import check_need_grad

        api_info_dict = {"input_kwargs": {"out": True}}
        result = check_need_grad(api_info_dict)
        self.assertFalse(result)

    def test_check_need_grad_given_no_out_kwarg_then_return_true(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import check_need_grad

        api_info_dict = {"input_kwargs": {}}
        result = check_need_grad(api_info_dict)
        self.assertTrue(result)

    def test_preprocess_forward_content_given_duplicate_apis_then_filter(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import preprocess_forward_content

        forward_content = {
            "torch.add_1": {"input_args": [{"value": 1}], "input_kwargs": {}},
            "torch.add_2": {"input_args": [{"value": 1}], "input_kwargs": {}},
            "torch.sub": {"input_args": [{"value": 2}], "input_kwargs": {}}
        }

        result = preprocess_forward_content(forward_content)

        self.assertEqual(len(result), 2)  # One duplicate should be removed

    def test_initialize_save_error_data_given_valid_path_then_return_path(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name

        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import initialize_save_error_data

        error_data_path = os.path.join(self.test_dir, "error_data")
        result = initialize_save_error_data(error_data_path)

        self.assertTrue(os.path.exists(result))
        self.assertIn("ut_error_data", result)
        self.temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
