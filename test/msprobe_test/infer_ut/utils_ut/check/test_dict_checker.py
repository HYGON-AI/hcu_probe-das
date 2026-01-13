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

import pytest

from msprobe.infer.utils.check.dict_checker import DictChecker
from msprobe.infer.utils.check.number_checker import NumberChecker


@pytest.mark.parametrize("test_dict, value", [
    ({"a": 1}, True),
    ({}, True),
    ("a", False),
])
def test_is_dict(test_dict, value):
    dict_checker = DictChecker(test_dict)
    assert dict_checker.is_dict().passed is value


@pytest.mark.parametrize("test_dict, value", [
    ({"a": 1}, True),
    ({}, False),
])
def test_is_dict_not_empty(test_dict, value):
    dict_checker = DictChecker(test_dict)
    assert dict_checker.is_dict_not_empty().passed is value


@pytest.mark.parametrize("test_dict, test_key, value", [
    ({"a": {"b": 1}}, "b", True),
    ({"a": {"b": 1}}, "c", False),
])
def test_is_key_exists_in_dict(test_dict, test_key, value):
    dict_checker = DictChecker(test_dict)
    assert dict_checker.is_key_exists_in_dict(test_key).passed is value


@pytest.mark.parametrize("test_dict, value_rules, value", [
    ({"a": 1}, {"a": NumberChecker().is_int()}, True),
    ({"a": 1, "b": 1}, {"b": NumberChecker().is_float()}, False),
])
def test_is_values_valid(test_dict, value_rules, value):
    dict_checker = DictChecker(test_dict)
    check_result = dict_checker.is_values_valid(**value_rules)
    assert check_result.passed is value


@pytest.mark.parametrize("test_dict, exp_type, value", [
    ({"a": 1, "b": 1}, str, True),
    ({"a": 2, 100: 1}, int, False),
])
def test_is_key_type_valid(test_dict, exp_type, value):
    dict_checker = DictChecker(test_dict)
    assert dict_checker.is_key_type_valid(exp_type).passed is value
