import os
import glob
import unittest
import logging
from unittest.mock import patch, mock_open, MagicMock
import json
import signal
from msprobe.core.common.file_utils import create_directory, save_json, write_csv
from msprobe.core.common.exceptions import FileCheckException
from msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check import split_json_file, signal_handler, run_parallel_ut, \
    prepare_config, main, ParallelUTConfig


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
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_api_info_path_soft_link_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.soft_json_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_out_path_soft_link_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.soft_path)
        
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)
    
    def test_result_csv_path_soft_link_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path, 
                    result_csv_path=self.csv_path)
        
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)
    
    def test_config_path_empty_check(self):
        args = Args(config_path=self.empty_path, api_info_path=self.hard_json_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_api_info_path_empty_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.empty_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_out_path_empty_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.empty_path)
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_result_csv_path_empty_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path, 
                    result_csv_path=self.empty_path)
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_config_path_invalid_check(self):
        args = Args(config_path=123, api_info_path=self.hard_json_path, out_path=self.hard_path)
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_api_info_path_invalid_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path="123", out_path=self.hard_path)
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_out_path_invalid_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=123)
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_result_csv_path_invalid_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path, 
                    result_csv_path=123)
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)


class TestMultiRunUT(unittest.TestCase):

    def setUp(self):
        self.test_json_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dump.json")
        self.test_data = {'dump_data_dir': '/test', 'data': {'key1': 'TRUE', 'key2': 'TRUE', 'key3': 'TRUE'}}
        self.test_json_content = json.dumps(self.test_data)
        self.forward_split_files_content = [
            {'key1': 'TRUE', 'key2': 'TRUE'},
            {'key3': 'TRUE', 'key4': 'TRUE'}
        ]

    @patch('os.remove')
    @patch('os.path.realpath', side_effect=lambda x: x)
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.check_link')
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.check_file_suffix')
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.FileChecker')
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.split_json_file',
           return_value=(['forward_split1.json', 'forward_split2.json'], 2))
    def test_prepare_config(self, mock_split_json_file, mock_FileChecker, mock_check_file_suffix, mock_check_link,
                            mock_realpath, mock_remove):
        mock_FileChecker_instance = MagicMock()
        mock_FileChecker_instance.common_check.return_value = './'
        mock_FileChecker.return_value = mock_FileChecker_instance
        args = MagicMock()
        args.api_info = 'forward.json'
        args.out_path = './'
        args.num_splits = 2
        args.save_error_data = True
        args.jit_compile = False
        args.device_id = [0, 1]
        args.result_csv_path = None
        args.config_path = None

        config = prepare_config(args)

        self.assertEqual(config.num_splits, 2)
        self.assertTrue(config.save_error_data_flag)
        self.assertFalse(config.jit_compile_flag)
        self.assertEqual(config.device_id, [0, 1])
        self.assertEqual(config.total_items, 2)


    @patch('argparse.ArgumentParser.parse_args')
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.prepare_config')
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.run_parallel_ut')
    def test_main(self, mock_run_parallel_ut, mock_prepare_config, mock_parse_args):
        main()
        mock_parse_args.assert_called()
        mock_prepare_config.assert_called()
        mock_run_parallel_ut.assert_called()

    def tearDown(self):
        current_directory = os.getcwd()
        pattern = os.path.join(current_directory, 'accuracy_checking_*')
        files = glob.glob(pattern)

        for file in files:
            try:
                os.remove(file)
                logging.info(f"Deleted file: {file}")
            except Exception as e:
                logging.error(f"Failed to delete file {file}: {e}")

