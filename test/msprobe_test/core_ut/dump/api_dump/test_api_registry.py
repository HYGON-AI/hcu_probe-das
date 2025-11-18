# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
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

from unittest import TestCase
from unittest.mock import patch

import torch

from msprobe.core.common.const import Const
from msprobe.core.dump.api_dump.api_registry import _get_attr, ApiWrapper


class TestFunctions(TestCase):
    def test__get_attr(self):
        module = torch

        attr_name = 'linalg.norm'
        target_value = torch.linalg.norm
        actual_value = _get_attr(module, attr_name)
        self.assertEqual(target_value, actual_value)

        attr_name = 'norm'
        target_value = torch.norm
        actual_value = _get_attr(module, attr_name)
        self.assertEqual(target_value, actual_value)


class TestApiWrapper(TestCase):
    api_types = {
        Const.PT_FRAMEWORK: {
            Const.PT_API_TYPE_TORCH: ((torch,), torch),
        }
    }
    supported_api_list_path = (Const.SUPPORT_API_FILE_NAME,)
    yaml_value = {'torch': ['linalg.norm', 'norm']}
    api_names = {Const.PT_FRAMEWORK: {'torch': {'linalg.norm', 'norm'}}}

    def test___init__(self):
        with patch('msprobe.core.dump.api_dump.api_registry.load_yaml', return_value=self.yaml_value):
            api_wrapper = ApiWrapper(self.api_types, self.supported_api_list_path)
            self.assertEqual(api_wrapper.api_types, self.api_types)
            self.assertEqual(api_wrapper.api_list_paths, self.supported_api_list_path)
            self.assertEqual(api_wrapper.api_names, self.api_names)
            self.assertEqual(api_wrapper.wrapped_api_functions, {})

            api_wrapper = ApiWrapper(self.api_types, Const.SUPPORT_API_FILE_NAME)
            self.assertEqual(api_wrapper.api_list_paths, list(self.supported_api_list_path))

            with self.assertRaises(Exception) as context:
                api_wrapper = ApiWrapper(self.api_types, (Const.SUPPORT_API_FILE_NAME, Const.SUPPORT_API_FILE_NAME))
            self.assertEqual(str(context.exception),
                             "The number of api_list_paths must be equal to the number of frameworks in 'api_types', "
                             "when api_list_paths is a list or tuple.")

    def test__get_api_names(self):
        target_value = self.api_names
        with patch('msprobe.core.dump.api_dump.api_registry.load_yaml', return_value=self.yaml_value):
            api_wrapper = ApiWrapper(self.api_types, self.supported_api_list_path)
            actual_value = api_wrapper._get_api_names()
        self.assertEqual(target_value, actual_value)
