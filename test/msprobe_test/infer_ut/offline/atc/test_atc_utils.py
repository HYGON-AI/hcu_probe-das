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
import stat
import msprobe.infer.offline.compare.msquickcmp.common.utils as utils
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException
from msprobe.infer.offline.compare.msquickcmp.atc.atc_utils import convert_model_to_json, get_atc_path


class TestConvertModelToJson(unittest.TestCase):
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.get_model_name_and_extension")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.logger")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.get_atc_path")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.execute_command")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.check_file_size_valid")
    def test_invalid_model_extension(self, mock_check_size, mock_execute, mock_get_atc, mock_os_path, mock_logger,
                                     mock_get_model):
        mock_get_model.return_value = ("model_name", ".invalid")
        mock_os_path.isdir.return_value = True
        with self.assertRaises(AccuracyCompareException) as context:
            convert_model_to_json("/cann_path", "model.invalid", "/out_path")
        self.assertEqual(context.exception.error_info, utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)
        mock_logger.error.assert_called_once_with(
            'The offline model file not ends with .om or .txt, Please check model.invalid.'
        )

    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.get_model_name_and_extension")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path")
    def test_cann_path_not_directory(self, mock_os_path, mock_get_model):
        mock_get_model.return_value = ("model_name", ".om")
        mock_os_path.isdir.return_value = False
        mock_os_path.realpath.return_value = "/invalid_cann_path"
        with self.assertRaises(AccuracyCompareException) as context:
            convert_model_to_json("/invalid_cann_path", "model.om", "/out_path")
        self.assertEqual(context.exception.error_info, utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.logger")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.get_atc_path")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path.join")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path.exists")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path.realpath")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path.isdir")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.get_model_name_and_extension")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.check_file_or_directory_path")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.stat")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.check_file_size_valid")
    def test_output_json_exists(
        self,
        mock_check_file_size_valid,
        mock_os_stat,
        mock_check_path,
        mock_get_model_name_ext,
        mock_isdir,
        mock_realpath,
        mock_exists,
        mock_join,
        mock_get_atc_path,
        mock_logger,
    ):
        mock_get_model_name_ext.return_value = ("model_name", ".om")
        mock_realpath.return_value = "/cann_path"
        mock_isdir.return_value = True
        mock_join.return_value = "/out_path/model/model_name.json"
        mock_exists.return_value = True
        mock_get_atc_path.return_value = "/cann_path/compiler/bin/atc"
        mock_os_stat.return_value = MagicMock(st_size=123456)

        result = convert_model_to_json("/cann_path", "model.om", "/out_path")

        self.assertEqual(result, "/out_path/model/model_name.json")
        mock_logger.info.assert_any_call("The {} file is exists.".format("/out_path/model/model_name.json"))
        mock_check_path.assert_called_once_with("/cann_path/compiler/bin/atc")
        mock_check_file_size_valid.assert_called_once_with("/out_path/model/model_name.json", 
                                                           unittest.mock.ANY)

    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.get_model_name_and_extension")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.get_atc_path")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.execute_command")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.logger")
    def test_successful_om_conversion(self, mock_logger, mock_execute, mock_get_atc, mock_os_path, mock_get_model):
        mock_get_model.return_value = ("model_name", ".om")
        mock_os_path.isdir.return_value = True
        mock_os_path.join.return_value = "/out_path/model/model_name.json"
        mock_os_path.exists.return_value = False
        mock_os_path.realpath.return_value = "/cann_path"
        mock_get_atc.return_value = "/cann_path/compiler/bin/atc"

        with patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.check_file_or_directory_path"), \
            patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.check_file_size_valid"), \
            patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.stat", return_value=MagicMock(st_size=123456)):
            
            result = convert_model_to_json("/cann_path", "model.om", "/out_path")

            self.assertEqual(result, "/out_path/model/model_name.json")
            mock_logger.info.assert_any_call("Start to converting the model to json")
            expected_cmd = [
                "/cann_path/compiler/bin/atc",
                "--mode=1",
                "--om=model.om",
                "--json=/out_path/model/model_name.json"
            ]
            mock_logger.info.assert_any_call(f"ATC command line {' '.join(expected_cmd)}")
            mock_execute.assert_called_once_with(expected_cmd)
            mock_logger.info.assert_any_call("Complete model conversion to json /out_path/model/model_name.json.")

    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.stat")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.check_file_size_valid")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.check_file_or_directory_path")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.execute_command")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.get_atc_path")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.logger")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.get_model_name_and_extension")
    def test_txt_model_conversion(
        self,
        mock_get_model,
        mock_logger,
        mock_os_path,
        mock_get_atc,
        mock_execute,
        mock_check_path,
        mock_check_size,
        mock_os_stat
    ):
        mock_get_model.return_value = ("txt_model", ".txt")
        mock_os_path.isdir.return_value = True
        mock_os_path.realpath.return_value = "/cann_path"
        mock_os_path.join.return_value = "/out_path/model/txt_model.json"
        mock_os_path.exists.return_value = False
        mock_get_atc.return_value = "/cann_path/compiler/bin/atc"
        mock_os_stat.return_value = MagicMock(st_size=123456)

        result = convert_model_to_json("/cann_path", "model.txt", "/out_path")

        expected_cmd = [
            "/cann_path/compiler/bin/atc",
            "--mode=5",
            "--om=model.txt",
            "--json=/out_path/model/txt_model.json"
        ]
        self.assertEqual(result, "/out_path/model/txt_model.json")
        mock_execute.assert_called_once_with(expected_cmd)


class TestGetAtcPath(unittest.TestCase):
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path.exists")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.access")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.stat")
    def test_valid_atc_path(self, mock_stat, mock_access, mock_exists):
        mock_exists.side_effect = lambda x: x == "/cann_path/compiler/bin/atc"
        mock_access.return_value = True
        mock_stat.return_value.st_mode = stat.S_IRUSR | stat.S_IXUSR
        
        result = get_atc_path("/cann_path")
        self.assertEqual(result, "/cann_path/compiler/bin/atc")

    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path.exists")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.access")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.stat")
    def test_fallback_to_old_path(self, mock_stat, mock_access, mock_exists):
        mock_exists.side_effect = lambda x: x == "/cann_path/atc/bin/atc"
        mock_access.return_value = True
        mock_stat.return_value.st_mode = stat.S_IRUSR | stat.S_IXUSR
        
        result = get_atc_path("/cann_path")
        self.assertEqual(result, "/cann_path/atc/bin/atc")

    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path.exists")
    def test_no_atc_found(self, mock_exists):
        mock_exists.return_value = False
        
        with self.assertRaises(AccuracyCompareException) as context:
            get_atc_path("/cann_path")
        self.assertEqual(context.exception.error_info, utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path.exists")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.access")
    def test_atc_not_executable(self, mock_access, mock_exists):
        mock_exists.return_value = True
        mock_access.return_value = False
        
        with self.assertRaises(AccuracyCompareException) as context:
            get_atc_path("/cann_path")
        self.assertEqual(context.exception.error_info, utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.path.exists")
    @patch("msprobe.infer.offline.compare.msquickcmp.atc.atc_utils.os.stat")
    def test_insecure_permissions(self, mock_stat, mock_exists):
        mock_exists.return_value = True
        mock_stat.return_value.st_mode = stat.S_IWGRP | stat.S_IWOTH
        
        with self.assertRaises(AccuracyCompareException) as context:
            get_atc_path("/cann_path")
        self.assertEqual(context.exception.error_info, utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)
