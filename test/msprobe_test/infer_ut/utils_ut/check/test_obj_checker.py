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