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
import json
import os
import shutil
from unittest import TestCase
from unittest.mock import patch, MagicMock, mock_open

from msprobe.core.common.exceptions import ParseJsonException
from msprobe.pytorch.common.parse_json import parse_data_name_with_pattern
from msprobe.pytorch.common.parse_json import parse_json_info_forward_backward


class TestParseJson(TestCase):

    def setUp(self):
        self.base_test_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.input_dir = os.path.join(self.base_test_dir, 'resources')
        self.output_path = os.path.abspath(os.path.join(self.base_test_dir, 'test_output'))

        os.makedirs(self.output_path, mode=0o700, exist_ok=True)
        self.has_error = False

    def tearDown(self) -> None:
        shutil.rmtree(self.output_path, ignore_errors=True)

    def test_parse_data_name_with_pattern_exception(self):
        invalid_data_name = "matmul.unforward"
        pattern = "forward"
        json_path = "test/test.json"

        with self.assertRaises(ParseJsonException):
            parse_data_name_with_pattern(invalid_data_name, pattern, json_path)

    def test_parse_json_info_forward_backward_no_dump_data_exception(self):
        no_dump_data_json = os.path.join(self.input_dir, 'common', 'test_no_dump_data.json')
        with self.assertRaises(ParseJsonException):
            parse_json_info_forward_backward(no_dump_data_json)

    def test_parse_json_info_forward_backward(self):
        backward_json = os.path.join(self.input_dir, 'common', 'test_backward.json')
        golden_result = ({}, {"matmul": 1}, "./")
        result = parse_json_info_forward_backward(backward_json)
        self.assertEqual(result, golden_result)

    def test_parse_json_info_forward_backward_no_fb_data_exception(self):
        no_fb_data_json = os.path.join(self.input_dir, 'common', 'test_no_fb_data.json')
        with self.assertRaises(ParseJsonException):
            parse_json_info_forward_backward(no_fb_data_json)
