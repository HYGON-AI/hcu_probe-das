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
from msprobe.infer.utils.check.checker import Checker


@pytest.mark.parametrize("value", [(10), (10.989), ("223")])
def test_checker_given_value_when_not_eq_then_pass(value):
    assert not Checker(value).not_eq(value)


@pytest.mark.parametrize("value", [(10), (10.989), ("223")])
def test_checker_given_not_none_value_when_none_rule_then_pass(value):
    assert Checker(value).not_none()
    assert not Checker(value).is_none()


def test_checker_given_none_value_when_none_rule_then_pass():
    assert not Checker(None).not_none()
    assert Checker(None).is_none()


@pytest.mark.parametrize("value, vtype", [(10, int), (10.989, float), ("223", str), ("223", (int, str))])
def test_checker_given_value_types_when_type_rule_then_pass(value, vtype):
    assert Checker(value).is_type(vtype)


@pytest.mark.parametrize("value, vtype", [(10, float), (10.989, int), ("223", dict)])
def test_checker_given_value_err_types_when_type_rule_then_not_pass(value, vtype):
    assert not Checker(value).is_type(vtype)


@pytest.mark.parametrize("value", [(None), (10.989), (0)])
def test_checker_given_mutil_rule_when_any_rule_then_pass(value):
    assert Checker(value).any(
        Checker().is_none(),
        Checker().is_type(float),
        Checker().eq(0),
    )


def test_checker_given_mutil_rule_when_any_rule_then_not_pass():
    assert not Checker("88").any(
        Checker().is_none(),
        Checker().is_type(float),
        Checker().eq(0),
    )


@pytest.mark.parametrize("value, vtype", [(10, int), (10.989, float), ("223", str), ("223", (int, str))])
def test_checker_given_values_when_rule_then_pass(value, vtype):
    assert Checker().is_type(vtype).check(value)


def test_checker_given_rules_when_rule_then_pass():
    rule = Checker().is_type(int).eq(10)
    assert rule.check(10)
    assert not rule.check(11)
    assert not rule.check(10.0)


@pytest.mark.parametrize("value, res", [(10, False), (11, False), (12, False), (9.0, False), (9, True)])
def test_checker_given_same_rules_when_rule_then_pass(value, res):
    rule = Checker().is_type(int).not_eq(10).not_eq(11).not_eq(12)
    assert bool(rule.check(value)) == res


@pytest.mark.parametrize("value, res", [(5.0, False), (5, False), (8, False), (8.0, True)])
def test_checker_default_when_any_then_pass(value, res):
    rule = Checker().is_type(int).not_eq(5).as_default().is_type(float)
    assert bool(rule.check(value)) == res


@pytest.mark.parametrize("value, res", [(5, True), (8, False)])
def test_checker_default_when_any_then_pass(value, res):
    rule = Checker().is_type(int).not_eq(5).as_default().not_eq(8)
    assert bool(rule.check(value)) == res
