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

from msprobe.infer.utils.check import Rule, ObjectChecker


class TmpObj:
    def __init__(self, param_a, param_b) -> None:
        self.p_a = param_a
        self.p_b = param_b


def test_obj_checker_given_attr_rule_when_any_pass():
    obj_rule = ObjectChecker().is_attrs_valid(p_a=Rule.num(), p_b=Rule.str())
    assert obj_rule.check(TmpObj(1, "1"))


def test_obj_checker_given_attr_rule_when_any_failed():
    obj_rule = ObjectChecker().is_attrs_valid(p_a=Rule.num(), p_b=Rule.str())
    assert not obj_rule.check(TmpObj("1", 1))
    assert len(str(obj_rule.check(TmpObj("1", 1))).split("\n")) == 2


def test_obj_checker_given_attr_rule_when_both_failed():
    obj_rule = ObjectChecker().is_attrs_valid(p_a=Rule.num(), p_b=Rule.str())
    result = obj_rule.check(TmpObj("not_num", 123))
    assert not bool(result), "Check should fail because both 'p_a' and 'p_b' are invalid"
    err_msgs = str(result).split("\n")  
    assert len(err_msgs) == 2, "There should be two error messages"
    assert "p_a" in err_msgs[0], "'p_a' should be mentioned in the first error message"
    assert "p_b" in err_msgs[1], "'p_b' should be mentioned in the second error message"


def test_obj_checker_none_instance():
    obj_rule = ObjectChecker().is_attrs_valid(p_a=Rule.num(), p_b=Rule.str())
    result = obj_rule.check(None)
    assert not bool(result), "Check should fail because instance is None"
    assert "None" in str(result), "Error message should mention that instance is None"


def test_obj_checker_missing_attribute():
    obj_rule = ObjectChecker().is_attrs_valid(p_a=Rule.num(), p_b=Rule.str())
    class ObjWithoutPB:
        def __init__(self):
            self.p_a = 1
    result = obj_rule.check(ObjWithoutPB())
    assert not bool(result), "Check should fail because 'p_b' attribute is missing"
    err_msgs = str(result).split("\n")
    assert len(err_msgs) == 1, "There should be one error message"
    assert "p_b" in err_msgs[0], "'p_b' should be mentioned in the error message"