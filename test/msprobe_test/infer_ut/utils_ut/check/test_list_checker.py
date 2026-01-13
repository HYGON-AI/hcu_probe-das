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

from msprobe.infer.utils.check.list_checker import ListChecker
from msprobe.infer.utils.check.number_checker import NumberChecker


@pytest.mark.parametrize("param, value", [
    ([1, [2, 3]], True),
    ([], True),
    (1, False),
])
def test_is_list(param, value):
    list_checker = ListChecker(param)
    assert list_checker.is_list().passed is value


@pytest.mark.parametrize("param, value", [
    ([1, 2, 3], True),
    ([], False),
])
def test_is_list_not_empty(param, value):
    list_checker = ListChecker(param)
    assert list_checker.is_list_not_empty().passed is value


@pytest.mark.parametrize("param, check_rule, value", [
    ([1, 2, 3], NumberChecker().is_int(), True),
    (["a", 2, "c"], NumberChecker().is_int(), False),
])
def test_is_element_valid(param, check_rule, value):
    list_checker = ListChecker(param)
    assert list_checker.is_element_valid(check_rule).passed is value


@pytest.mark.parametrize("param, min_length, max_length, value", [
    ([1, 2, 3], 3, None, True),
    ([1, 2, 3], None, 3, True),
    ([1, 2, 3], 1, 4, True),
    ([1, 2, 3], 2, 2, False),
])
def test_is_length_valid(param, min_length, max_length, value):
    list_checker = ListChecker(param)
    assert list_checker.is_length_valid(min_length, max_length).passed is value
