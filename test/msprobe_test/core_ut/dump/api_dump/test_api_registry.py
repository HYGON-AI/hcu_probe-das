# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

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
