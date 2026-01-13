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
from msprobe.infer.utils.check import Rule, NumberChecker


@pytest.mark.parametrize(
    "vrange, value, res",
    [
        ((1.2, 3.4), 2.0, True),
        ((1.2, 3.4), "2", True),
        ((1.2, 3.4), "5.2", False),
        ((1.2, 3.4), 5.0, False),
        ((1.2, 3.4), "ERW", False),
    ],
)
def test_rule_given_to_float_when_any_then_pass(vrange, value, res):
    assert bool(Rule.to_float().in_range(*vrange).check(value)) == res


@pytest.mark.parametrize(
    "vrange, value, res",
    [
        ((1.2, 3.4), "2.0", False),
        ((1.2, 3.4), 2.0, True),
        ((1.2, 3.4), "2", True),
        ((1.2, 3.4), "5.2", False),
        ((1.2, 3.4), 5, False),
        ((1.2, 3.4), "ERW", False),
    ],
)
def test_rule_given_to_int_when_any_then_pass(vrange, value, res):
    assert bool(Rule.to_int().in_range(*vrange).check(value)) == res


def test_rule_given_num_when_any_then_pass():
    assert Rule.num().check(2.3)
    assert not Rule.num().check("2.3")


def test_rule_given_str_when_any_then_pass():
    assert not Rule.str().check(2.3)
    assert Rule.str().check("2.3")


def test_rule_given_dict_when_any_then_pass():
    assert not Rule.dict().check("2.3")
    assert Rule.dict().is_key_exists_in_dict("ss").check({"ss": 2.3})


def test_rule_given_obj_when_any_then_pass():
    assert not Rule.obj(float).check("2.3")
    assert Rule.obj(str).check("2.3")
    assert Rule.obj(NumberChecker).is_attr_valid("instance", Rule.num()).check(NumberChecker(2.3))


def test_rule_given_obj_when_mutil_errs_then_pass():
    rule = Rule.obj(NumberChecker).is_attrs_valid(instance=Rule.str(), converter=Rule.obj(str))
    is_pass = rule.check(NumberChecker(2.3))
    assert not is_pass
    assert 2 == len(str(is_pass).split("\n"))


def test_rule_given_any_when_any_then_pass():
    rule = Rule.any(Rule.none(), Rule.num().is_int().in_range(3, 5))
    assert rule.check(None)
    assert rule.check(4)
    assert rule.check(3)
    assert rule.check(5)
    assert not rule.check("234")
    assert not rule.check(234)
    assert not rule.check(234.0)
