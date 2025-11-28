# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import os, csv, subprocess, io
from unittest.mock import patch, MagicMock, mock_open, call
from msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare import NetCompare


class TestNetCompare_init(unittest.TestCase):
    @patch("msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.check_file_size_valid", return_value=None)
    @patch.object(NetCompare, "_check_msaccucmp_file", return_value="/fake/path/msaccucmp.py")
    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.check_file_size_valid")
    def test_init_basic(self, mock_check_size, mock_check_file, mock_check_file_size_valid):
        class Args:
            cann_path = "/fake/cann"
            quant_fusion_rule_file = None
            output_path = "/out/path"
            dump = False

        args = Args()
        obj = NetCompare("/npu/path", "/cpu/path", "/output/json", args, "/golden/json")
        self.assertEqual(obj.msaccucmp_command_file_path, "/fake/path/msaccucmp.py")
        # check python_version取值正常
        self.assertIn("python", obj.python_version.lower())


class TestNetCompare_execute_command_line(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    @patch("subprocess.Popen")
    def test_execute_command_line(self, mock_popen, mock_logger):
        proc_mock = MagicMock()
        mock_popen.return_value = proc_mock

        cmd = ["echo", "test"]
        proc = NetCompare.execute_command_line(cmd)
        mock_logger.info.assert_called_once()
        mock_popen.assert_called_once_with(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(proc, proc_mock)


class TestNetCompare_check_msaccucmp_file(unittest.TestCase):
    @patch("os.path.exists")
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_check_msaccucmp_file_found(self, mock_logger, mock_exists):
        mock_exists.side_effect = lambda x: "msaccucmp.py" in x
        path = "/some/path"
        file_path = NetCompare._check_msaccucmp_file(path)
        self.assertIn("msaccucmp.py", file_path)
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    @patch("os.path.exists", return_value=False)
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_check_msaccucmp_file_not_found(self, mock_logger, mock_exists):
        with self.assertRaises(Exception):  # AccuracyCompareException
            NetCompare._check_msaccucmp_file("/some/path")
        self.assertTrue(mock_logger.error.called)


class TestNetCompare_check_pyc_to_python_version(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_check_pyc_to_python_version_correct(self, mock_logger):
        # should not raise
        NetCompare._check_pyc_to_python_version("file.py", "3.7.5")

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_check_pyc_to_python_version_incorrect(self, mock_logger):
        with self.assertRaises(Exception):  # AccuracyCompareException
            NetCompare._check_pyc_to_python_version("file.pyc", "3.8.0")
        mock_logger.error.assert_called_once()


class TestNetCompare_catch_compare_result(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_num(self, mock_logger):
        line = b"[INFO] 0.123 0.456"
        res, header = NetCompare._catch_compare_result(line, True)
        self.assertTrue(res)
        self.assertFalse(header)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_nan(self, mock_logger):
        line = b"[INFO] NaN something"
        res, header = NetCompare._catch_compare_result(line, True)
        self.assertTrue(res)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_header(self, mock_logger):
        line = b"[INFO] Cosine Something"
        res, header = NetCompare._catch_compare_result(line, True)
        self.assertFalse(res)
        self.assertTrue(header)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_exception(self, mock_logger):
        with self.assertRaises(Exception):  # AccuracyCompareException
            NetCompare._catch_compare_result(b"\xff", True)  # invalid decode


class TestNetCompare_accuracy_network_compare(unittest.TestCase):
    @patch.object(NetCompare, "_check_msaccucmp_file", return_value="/fake/path/msaccucmp.py")
    @patch.object(NetCompare, "execute_msaccucmp_command", return_value=(0, [], []))
    @patch.object(NetCompare, "_check_pyc_to_python_version")
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_accuracy_network_compare_success(self, mock_logger, mock_check_py_ver, mock_exec, mock_check_file):
        class Args:
            cann_path = "/fake/cann"
            quant_fusion_rule_file = None
            output_path = "/out/path"
            dump = False
        args = Args()
        obj = NetCompare("/npu/path", "/cpu/path", "/output/json", args)
        obj.accuracy_network_compare()
        mock_exec.assert_called_once()
        mock_logger.info.assert_any_call("Finish compare the files in directory /npu/path with those in directory /cpu/path.")

    @patch.object(NetCompare, "execute_msaccucmp_command", return_value=(1, [], []))
    @patch.object(NetCompare, "_check_pyc_to_python_version")
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    @patch.object(NetCompare, "_check_msaccucmp_file", return_value="/fake/path/msaccucmp.py")
    def test_accuracy_network_compare_fail(self, mock_check_file, mock_logger, mock_check_py_ver, mock_exec):
        class Args:
            cann_path = "/fake/cann"
            quant_fusion_rule_file = None
            output_path = "/out/path"
            dump = False
        args = Args()
        obj = NetCompare("/npu/path", "/cpu/path", "/output/json", args)
        with self.assertRaises(Exception):  # AccuracyCompareException
            obj.accuracy_network_compare()


class TestCheckPycToPythonVersion(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_valid_python_version_with_pyc(self, mock_logger):
        # 不应抛异常
        NetCompare._check_pyc_to_python_version("file.py", "3.7.5")
        NetCompare._check_pyc_to_python_version("file.pyc", "3.7.5")

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_invalid_python_version_with_pyc(self, mock_logger):
        with self.assertRaises(Exception):
            NetCompare._check_pyc_to_python_version("file.pyc", "3.8.0")
        mock_logger.error.assert_called_once()


class TestCatchCompareResult(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_valid_number(self, mock_logger):
        line = b'[INFO]123.456 other stuff'
        result, header = NetCompare._catch_compare_result(line, True)
        self.assertTrue(result)
        self.assertFalse(header)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_valid_nan(self, mock_logger):
        line = b'[INFO]NaN some other'
        result, header = NetCompare._catch_compare_result(line, True)
        self.assertTrue(result)
        self.assertFalse(header)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_header(self, mock_logger):
        line = b'[INFO]Cosine Error Distance'
        result, header = NetCompare._catch_compare_result(line, True)
        self.assertFalse(result)
        self.assertTrue(header)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_no_info(self, mock_logger):
        line = b'Some random log line'
        result, header = NetCompare._catch_compare_result(line, True)
        self.assertFalse(result)
        self.assertFalse(header)

    def test_catch_compare_result_exception(self):
        with self.assertRaises(Exception):
            NetCompare._catch_compare_result(123, True)


class TestExecuteCommandLine(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    @patch('subprocess.Popen')
    def test_execute_command_line(self, mock_popen, mock_logger):
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        cmd = ['echo', 'hello']
        ret = NetCompare.execute_command_line(cmd)
        mock_logger.info.assert_called()
        mock_popen.assert_called_with(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(ret, mock_process)


class TestExecuteMsaccucmpCommand(unittest.TestCase):
    @patch.object(NetCompare, "execute_command_line")
    @patch.object(NetCompare, "_catch_compare_result")
    @patch.object(NetCompare, "_check_msaccucmp_file", return_value="/fake/path/msaccucmp.py")
    def test_execute_msaccucmp_command(self, mock_check_file, mock_catch, mock_exec_cmd):
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, None, 0]
        mock_process.stdout.readline.side_effect = [b"[INFO]1.23 4.56\n", b"[INFO]Cosine\n", b""]
        mock_process.returncode = 0
        mock_exec_cmd.return_value = mock_process
        mock_catch.side_effect = [
            (["1.23", "4.56"], []),
            ([], ["Cosine"])
        ]
        class Args:
            cann_path = "/fake/cann"
            quant_fusion_rule_file = None
            out_path = "/out/path"
            advisor = False
            max_cmp_size = None
            dump = False
        args = Args()
        obj = NetCompare("/npu/path", "/cpu/path", "/output/json", args)
        status, result, header = obj.execute_msaccucmp_command(["fake", "cmd"], catch=True)
        self.assertEqual(status, 0)
        self.assertEqual(result, ["1.23", "4.56"])
        self.assertEqual(header, ["Cosine"])
        mock_exec_cmd.assert_called_once_with(["fake", "cmd"])
        self.assertEqual(mock_catch.call_count, 2)


class TestProcessResultOneLine(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.csv.writer')
    @patch('msprobe.core.common.log.logger')
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.ms_open', new_callable=mock_open, read_data="Header,NPU_Dump,GroundTruth\nline1\nline2\n")
    def test_process_result_one_line(self, mock_file, mock_logger, mock_csv_writer):
        mock_fp_write = MagicMock()
        mock_fp_read = io.StringIO("NPU Dump,NPUDump,GroundTruth\nNode_Output,abc,*\nother,line,*\n")
        instance = NetCompare.__new__(NetCompare)
        # 模拟 _check_msaccucmp_compare_support_advisor 返回 False
        instance._check_msaccucmp_compare_support_advisor = MagicMock(return_value=False)
        # 调用
        instance._process_result_one_line(mock_fp_write, mock_fp_read, 'npu_file.npy', 'golden_file.npy', ['1','2'])
        self.assertTrue(mock_csv_writer.return_value.writerow.called)


class TestProcessResultToCsv(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.csv.writer')
    def test_process_result_to_csv_with_header(self, mock_writer):
        mock_fp_write = MagicMock()
        mock_fp_write.readlines.return_value = ['line1', 'line2', 'line3']
        mock_fp_write.seek = MagicMock()
        csv_info = MagicMock()
        csv_info.header = ['Cosine', 'Error']
        csv_info.npu_file_name = 'npu.npy'
        csv_info.golden_file_name = 'golden.npy'
        csv_info.result = ['0.1', '0.2']
        instance = NetCompare.__new__(NetCompare)
        instance._process_result_to_csv(mock_fp_write, csv_info)
        self.assertTrue(mock_writer.return_value.writerow.called)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.csv.writer')
    def test_process_result_to_csv_no_header(self, mock_writer):
        mock_fp_write = MagicMock()
        mock_fp_write.readlines.return_value = ['line1', 'line2', 'line3']
        mock_fp_write.seek = MagicMock()
        csv_info = MagicMock()
        csv_info.header = []
        csv_info.npu_file_name = 'npu.npy'
        csv_info.golden_file_name = 'golden.npy'
        csv_info.result = ['0.1', '0.2']
        instance = NetCompare.__new__(NetCompare)
        instance._process_result_to_csv(mock_fp_write, csv_info)
        self.assertTrue(mock_writer.return_value.writerow.called)
