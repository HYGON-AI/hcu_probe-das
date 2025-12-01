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
from unittest.mock import patch, MagicMock

from msprobe.infer.offline.compare.msquickcmp.dump.dump_process import (
    dump_process,
    dump_data,
    check_and_dump
)


class TestDumpProcess(unittest.TestCase):
    @patch(
        "msprobe.infer.offline.compare.msquickcmp.dump.dump_process.convert_npy_to_bin",
        return_value="converted_input"
    )
    @patch("msprobe.infer.offline.compare.msquickcmp.dump.dump_process.check_and_dump")
    @patch("os.path.realpath", side_effect=lambda x: f"/abs/{x}")
    def test_dump_process(self, mock_realpath, mock_check_and_dump, mock_convert):
        args = MagicMock()
        args.golden_path = "model"
        args.cann_path = "cann"
        args.input_data = "input"
        dump_process(args)
        self.assertEqual(args.golden_path, "/abs/model")
        self.assertEqual(args.input_data, "converted_input")
        mock_check_and_dump.assert_called_once()


class TestCheckAndDump(unittest.TestCase):
    @patch("msprobe.infer.offline.compare.msquickcmp.dump.dump_process.dump_data")
    @patch(
        "msprobe.infer.offline.compare.msquickcmp.dump.dump_process.utils.parse_dym_shape_range",
        return_value=["shape"]
    )
    @patch("msprobe.infer.offline.compare.msquickcmp.dump.dump_process.create_directory")
    @patch("msprobe.infer.offline.compare.msquickcmp.dump.dump_process.check_file_or_directory_path")
    @patch("msprobe.infer.offline.compare.msquickcmp.dump.dump_process.utils.check_device_param_valid")
    @patch("os.path.realpath", side_effect=lambda p: f"/abs/{p}")
    @patch("time.strftime", return_value="20250610123456")
    def test_check_and_dump_with_dym_shapes_1(
            self,
            mock_time,
            mock_realpath,
            mock_check_device,
            mock_check_file,
            mock_create_dir,
            mock_parse_shape,
            mock_dump
    ):
        args1 = MagicMock(
            golden_path="model",
            output_path="out",
            rank="0",
            dym_shape_range="test_shape_range"
        )
        check_and_dump(args1)
        mock_time.assert_called_once()
        mock_realpath.assert_called_once()
        mock_check_device.assert_called_once()
        mock_check_file.assert_called_once()
        mock_create_dir.assert_called_once()
        mock_parse_shape.assert_called_once()
        mock_dump.assert_called_once()

    @patch("msprobe.infer.offline.compare.msquickcmp.dump.dump_process.dump_data")
    @patch(
        "msprobe.infer.offline.compare.msquickcmp.dump.dump_process.utils.parse_dym_shape_range",
        return_value=["shape1", "shape2"]
    )
    @patch("msprobe.infer.offline.compare.msquickcmp.dump.dump_process.create_directory")
    @patch("msprobe.infer.offline.compare.msquickcmp.dump.dump_process.check_file_or_directory_path")
    @patch("msprobe.infer.offline.compare.msquickcmp.dump.dump_process.utils.check_device_param_valid")
    @patch("os.path.realpath", side_effect=lambda p: f"/abs/{p}")
    @patch("time.strftime", return_value="20250610123456")
    def test_check_and_dump_with_dym_shapes_2(
            self,
            mock_time,
            mock_realpath,
            mock_check_device,
            mock_check_file,
            mock_create_dir,
            mock_parse_shape,
            mock_dump
    ):
        args2 = MagicMock(
            golden_path="model",
            output_path="out",
            rank="0",
            dym_shape_range="test_shape_range"
        )
        check_and_dump(args2)
        mock_time.assert_called_once()
        mock_realpath.assert_called_once()
        mock_check_device.assert_called_once()
        mock_check_file.assert_called_once()
        mock_create_dir.assert_called_once()
        mock_parse_shape.assert_called_once()
        self.assertTrue(mock_dump.call_count == 2)


class TestDumpData(unittest.TestCase):

    def setUp(self):
        """测试前准备"""
        self.mock_args = MagicMock()
        self.mock_args.golden_path = "/fake/path/model.onnx"
        self.mock_args.input_shape = None
        self.mock_args.output_path = "/fake/output"
        self.mock_args.cann_path = "/fake/cann/path"
        self.mock_args.target_path = None

        self.input_shape = "input:1,224,224,3"
        self.original_out_path = "/fake/original/output"

    @patch('os.path.join')
    def test_dump_data_onnx_with_input_shape(self, mock_join):
        with patch('msprobe.infer.offline.compare.msquickcmp.onnx_model.onnx_dump_data.OnnxDumpData') as mock_dump:
            mock_dumper_instance = MagicMock()
            mock_dump.return_value = mock_dumper_instance

            dump_data(self.mock_args, self.input_shape, self.original_out_path)

            self.assertEqual(self.mock_args.input_shape, self.input_shape)
            mock_join.assert_called()
            mock_dumper_instance.generate_inputs_data.assert_called_once_with(
                npu_dump_data_path=None,
                use_aipp=False
            )
            mock_dumper_instance.generate_dump_data.assert_called_once()

    @patch('os.path.join')
    def test_dump_data_onnx_without_input_shape(self, mock_join):
        with patch('msprobe.infer.offline.compare.msquickcmp.onnx_model.onnx_dump_data.OnnxDumpData') as mock_dump:
            mock_dumper_instance = MagicMock()
            mock_dump.return_value = mock_dumper_instance

            dump_data(self.mock_args, None, self.original_out_path)

            self.assertIsNone(self.mock_args.input_shape)
            mock_join.assert_not_called()
            mock_dumper_instance.generate_dump_data.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.dump.dump_process.atc_utils.convert_model_to_json')
    @patch('os.path.join')
    def test_dump_data_om_with_aipp(self, mock_join, mock_convert):
        self.mock_args.golden_path = "/fake/path/model.om"
        mock_convert.return_value = "/fake/output/model.json"

        with patch('msprobe.infer.offline.compare.msquickcmp.dump.dump_process.OmParser') as mock_om_parser:
            mock_parser_instance = MagicMock()
            mock_parser_instance.get_aipp_config_content.return_value = {"aipp_config": "some_config"}
            mock_om_parser.return_value = mock_parser_instance

            with patch('msprobe.infer.offline.compare.msquickcmp.dump.dump_process.NpuDumpData') as mock_npu_dump:
                mock_dumper_instance = MagicMock()
                mock_npu_dump.return_value = mock_dumper_instance

                dump_data(self.mock_args, self.input_shape, self.original_out_path)

                mock_convert.assert_called_once_with(
                    self.mock_args.cann_path,
                    self.mock_args.golden_path,
                    self.mock_args.output_path
                )
                self.assertEqual(self.mock_args.target_path, self.mock_args.golden_path)
                mock_dumper_instance.generate_inputs_data.assert_called_once_with(use_aipp=True)
                mock_dumper_instance.generate_dump_data.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.dump.dump_process.atc_utils.convert_model_to_json')
    @patch('os.path.join')
    def test_dump_data_om_without_aipp(self, mock_join, mock_convert):
        self.mock_args.golden_path = "/fake/path/model.om"
        mock_convert.return_value = "/fake/output/model.json"

        with patch('msprobe.infer.offline.compare.msquickcmp.dump.dump_process.OmParser') as mock_om_parser:
            mock_parser_instance = MagicMock()
            mock_parser_instance.get_aipp_config_content.return_value = None

            mock_om_parser.return_value = mock_parser_instance

            with patch('msprobe.infer.offline.compare.msquickcmp.dump.dump_process.NpuDumpData') as mock_npu_dump:
                mock_dumper_instance = MagicMock()
                mock_npu_dump.return_value = mock_dumper_instance

                dump_data(self.mock_args, self.input_shape, self.original_out_path)

                mock_dumper_instance.generate_inputs_data.assert_called_once_with(use_aipp=False)

    @patch('msprobe.infer.offline.compare.msquickcmp.dump.dump_process.logger')
    def test_dump_data_unsupported_model_type(self, mock_logger):
        self.mock_args.golden_path = "/fake/path/model.pb"

        from msprobe.infer.offline.compare.msquickcmp.dump.dump_process import AccuracyCompareException
        with self.assertRaises(AccuracyCompareException):
            dump_data(self.mock_args, self.input_shape, self.original_out_path)
