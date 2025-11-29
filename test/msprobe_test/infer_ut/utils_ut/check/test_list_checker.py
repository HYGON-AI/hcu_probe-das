# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
