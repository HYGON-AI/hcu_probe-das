# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
