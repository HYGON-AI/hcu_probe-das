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
from msprobe.core.compare.layer_mapping.data_scope_parser import DumpDataItem
from msprobe.core.common.utils import CompareException


class TestDataScopeParser(TestCase):

    def test_check_stack_valid_invalid_stack_type(self):
        stack_info_string = "conv1.Conv2d.forward.input"
        with self.assertRaises(CompareException) as context:
            DumpDataItem.check_stack_valid(stack_info_string)
        self.assertEqual(context.exception.code, CompareException.INVALID_DATA_ERROR)

    def test_check_stack_valid_invalid_stack_info(self):
        stack_info_list = ["conv1.Conv2d.forward.input", 1]
        with self.assertRaises(CompareException) as context:
            DumpDataItem.check_stack_valid(stack_info_list)
        self.assertEqual(context.exception.code, CompareException.INVALID_DATA_ERROR)
