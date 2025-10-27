#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
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
"""
from unittest import TestCase
from unittest.mock import patch

from python.msprobe.core.common.log import logger
from python.msprobe.core.common.const import Const
from python.msprobe.core.common.exceptions import MsprobeException
from python.msprobe.core.common_config import CommonConfig, BaseConfig


class TestCommonConfig(TestCase):
    @patch.object(logger, "error_log_with_exp")
    def test_common_config(self, mock_error_log_with_exp):
        json_config = dict()

        common_config = CommonConfig(json_config)
        self.assertIsNone(common_config.task)
        self.assertIsNone(common_config.dump_path)
        self.assertEqual(common_config.rank, [])
        self.assertEqual(common_config.step, [])
        self.assertIsNone(common_config.level)
        self.assertFalse(common_config.enable_dataloader)

        json_config.update({"task": "md5"})
        CommonConfig(json_config)
        self.assertEqual(mock_error_log_with_exp.call_args[0][0],
                         "task is invalid, it should be one of {}".format(Const.TASK_LIST))
        self.assertEqual(str(mock_error_log_with_exp.call_args[0][1]),
                         MsprobeException.err_strs.get(MsprobeException.INVALID_PARAM_ERROR))

        json_config.update({"task": Const.TENSOR})
        json_config.update({"rank": [0]})
        json_config.update({"step": [0]})
        json_config.update({"level": "L3"})
        CommonConfig(json_config)
        self.assertEqual(mock_error_log_with_exp.call_args[0][0],
                         "level is invalid, it should be one of {}".format(Const.LEVEL_LIST))
        self.assertEqual(str(mock_error_log_with_exp.call_args[0][1]),
                         MsprobeException.err_strs.get(MsprobeException.INVALID_PARAM_ERROR))

        json_config.update({"task": Const.TENSOR})
        json_config.update({"rank": [0]})
        json_config.update({"step": [0]})
        json_config.update({"level": "L0"})
        json_config.update({"enable_dataloader": "ENABLE"})
        CommonConfig(json_config)
        self.assertEqual(mock_error_log_with_exp.call_args[0][0],
                         "enable_dataloader is invalid, it should be a boolean")
        self.assertEqual(str(mock_error_log_with_exp.call_args[0][1]),
                         MsprobeException.err_strs.get(MsprobeException.INVALID_PARAM_ERROR))


class TestBaseConfig(TestCase):
    @patch.object(logger, "error_log_with_exp")
    def test_base_config(self, mock_error_log_with_exp):
        json_config = dict()

        base_config = BaseConfig(json_config)
        base_config.check_config()
        self.assertIsNone(base_config.scope)
        self.assertIsNone(base_config.list)
        self.assertIsNone(base_config.file_format)
        self.assertIsNone(base_config.summary_mode)
        self.assertIsNone(base_config.overflow_nums)
        self.assertIsNone(base_config.check_mode)

        json_config.update({"scope": "Tensor_Add"})
        base_config = BaseConfig(json_config)
        base_config.check_config()
        self.assertEqual(mock_error_log_with_exp.call_args[0][0],
                         "scope is invalid, it should be a list[str]")
        self.assertEqual(str(mock_error_log_with_exp.call_args[0][1]),
                         MsprobeException.err_strs.get(MsprobeException.INVALID_PARAM_ERROR))

        json_config.update({"scope": ["Tensor_Add"]})
        json_config.update({"list": "Tensor_Add"})
        base_config = BaseConfig(json_config)
        base_config.check_config()
        self.assertEqual(mock_error_log_with_exp.call_args[0][0],
                         "list is invalid, it should be a list[str]")
        self.assertEqual(str(mock_error_log_with_exp.call_args[0][1]),
                         MsprobeException.err_strs.get(MsprobeException.INVALID_PARAM_ERROR))

    @patch.object(logger, "error_log_with_exp")
    def test_check_data_mode(self, mock_error_log_with_exp):
        self.config = BaseConfig({})

        self.config._check_data_mode()
        mock_error_log_with_exp.assert_not_called()

        self.config.data_mode = ["all"]
        self.config._check_data_mode()
        mock_error_log_with_exp.assert_not_called()

        self.config.data_mode = "all"
        self.config._check_data_mode()
        self.assertEqual(
            mock_error_log_with_exp.call_args_list[0][0][0],
            "data_mode is invalid, it should be a list[str]"
        )

        mock_error_log_with_exp.reset_mock()
        self.config.data_mode = ["all", "forward"]
        self.config._check_data_mode()
        self.assertEqual(
            mock_error_log_with_exp.call_args_list[0][0][0],
            "'all' cannot be combined with other options in data_mode."
        )

        mock_error_log_with_exp.reset_mock()
        self.config.data_mode = ["test", "input", "output", "forward", "backward"]
        self.config._check_data_mode()
        self.assertEqual(
            mock_error_log_with_exp.call_args_list[0][0][0],
            f"The number of elements in the data_made cannot exceed {len(Const.DUMP_DATA_MODE_LIST) - 1}."
        )

        mock_error_log_with_exp.reset_mock()
        self.config.data_mode = [123, 'test_case_1']
        self.config._check_data_mode()
        self.assertEqual(
            mock_error_log_with_exp.call_args_list[0][0][0],
            "data_mode is invalid, it should be a list[str]"
        )

        mock_error_log_with_exp.reset_mock()
        self.config.data_mode = ['forward', 'test_case_1']
        self.config._check_data_mode()
        self.assertEqual(
            mock_error_log_with_exp.call_args_list[0][0][0],
            f"The element 'test_case_1' of data_mode {self.config.data_mode} is not in {Const.DUMP_DATA_MODE_LIST}."
        )
