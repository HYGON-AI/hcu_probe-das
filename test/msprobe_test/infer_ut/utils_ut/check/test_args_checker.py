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
from argparse import ArgumentTypeError

from msprobe.infer.utils.check import Rule, ArgsChecker


def test_args_checker_given_type_checker_when_any_pass():
    args_checker = ArgsChecker(Rule.to_int())
    assert args_checker("12") == 12
    with pytest.raises(ArgumentTypeError):
        args_checker("15.2")
