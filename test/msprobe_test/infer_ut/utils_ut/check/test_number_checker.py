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
from msprobe.infer.utils.check.number_checker import NumberChecker


@pytest.mark.parametrize("param,value", [
    (10, "pass"),
    (10.989, "pass"),
    ("223", "not number")
])
def test_is_number(param, value):
    num_checker = NumberChecker(param)
    assert str(num_checker.is_number()) == value


@pytest.mark.parametrize("param,value", [
    (10, "pass"),
    (10.1, "not integer")
])
def test_is_int(param, value):
    positive_checker = NumberChecker(param)
    assert str(positive_checker.is_int()) == value


@pytest.mark.parametrize("param,value", [
    (2.7, "pass"),
    (1, "not float")
])
def test_is_float(param, value):
    negative_checker = NumberChecker(param)
    assert str(negative_checker.is_float()) == value


@pytest.mark.parametrize("param,value", [
    (0, "pass"),
    (100, "not zero")
])
def test_is_zero(param, value):
    zero_checker = NumberChecker(param)
    assert str(zero_checker.is_zero()) == value


@pytest.mark.parametrize("param1,param2,value", [
    (2, 10, "pass"),
    (4, 15, "not divisible by 15")
])
def test_is_divisible_by(param1, param2, value):
    divisible_checker = NumberChecker(param1)
    assert str(divisible_checker.is_divisible_by(param2)) == value


@pytest.mark.parametrize("param1,param2,param3,value", [
    (3, 2, 4, True),
    (4, 1, 2, False)
])
def test_in_range(param1, param2, param3, value):
    within_range_checker = NumberChecker(param1)
    assert within_range_checker.in_range(param2, param3).passed is value


@pytest.mark.parametrize("test_num, min_value, value", [
    (3, 2, True),
    (3, 4, False)
])
def test_greater_equal(test_num, min_value, value):
    min_checker = NumberChecker(test_num)
    assert min_checker.greater_equal(min_value).passed is value


@pytest.mark.parametrize("test_num, max_value, value", [
    (3, 4, True),
    (3, 2, False)
])
def test_less_equal(test_num, max_value, value):
    max_checker = NumberChecker(test_num)
    assert max_checker.less_equal(max_value).passed is value


@pytest.mark.parametrize("test_num, max_value, value", [
    (3, 4, True),
    (3, 2, False)
])
def test_less_than(test_num, max_value, value):
    less_than_checker = NumberChecker(test_num)
    assert less_than_checker.less_than(max_value).passed is value


@pytest.mark.parametrize("test_num, min_value, value", [
    (3, 2, True),
    (3, 4, False)
])
def test_greater_than(test_num, min_value, value):
    more_than_checker = NumberChecker(test_num)
    assert more_than_checker.greater_than(min_value).passed is value
