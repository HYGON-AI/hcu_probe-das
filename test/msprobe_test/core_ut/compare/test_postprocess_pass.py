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
from dataclasses import dataclass
from unittest import TestCase
from msprobe.core.compare.layer_mapping.postprocess_pass import extract_next_item_last_number, \
    replace_next_item_index, renumber_index_pass


@dataclass
class DataItem:
    """Class for keeping track of an item in inventory"""
    type_name: str
    full_scope: str
    layer_scope: str


class TestPostProcessPass(TestCase):

    def test_check_path_type_None(self):
        input_data = "conv1.Conv2d.forward.input"
        prefix = "Conv2d"
        none_result = extract_next_item_last_number(input_data, prefix)
        self.assertEqual(none_result, None)

    def test_check_path_type_find_result(self):
        input_data = "conv1.Conv2d.forward.input.conv1"
        prefix = "conv1"
        result_2 = extract_next_item_last_number(input_data, prefix)
        self.assertEqual(result_2, 2)

    def test_replace_next_item_index(self):
        input_data = "conv1.Conv2d.forward.input.conv1"
        prefix = "conv1"
        replace_result = replace_next_item_index(input_data, prefix, 1)
        self.assertEqual(replace_result, "conv1.1.forward.input.conv1")

    def test_replace_next_item_index_with_inf(self):
        input_data = "conv1.Conv2d.forward.input.conv1"
        prefix = "conv1"
        inf_value = float("inf")
        replace_result = replace_next_item_index(input_data, prefix, inf_value)
        self.assertEqual(replace_result, input_data)

    def test_renumber_index_pass(self):
        a = DataItem("ParallelTransformer", "fake_data.layers.10", "fake_data.layers")
        b = DataItem("ParallelTransformer", "fake_data.layers.12", "fake_data.layers")
        c = DataItem("FakeLayer", "fake_data.layers.10.a.b.c", "fake_data.layers.a.b")
        data_items = [a, b, c]
        renumber_index_pass(data_items, "ParallelTransformer")
        self.assertEqual(a.full_scope, "fake_data.layers.0")
        self.assertEqual(b.full_scope, "fake_data.layers.2")
        self.assertEqual(c.full_scope, "fake_data.layers.0.a.b.c")
