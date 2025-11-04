# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

from typing import Any
from argparse import ArgumentTypeError
from msprobe.infer.utils.check.checker import Checker


class ArgsChecker:
    """
    parser.add_argument(..., type=ArgsChecker(Rule.to_int()), ...)
    """

    def __init__(self, rule: Checker) -> None:
        self.rule = rule

    def __call__(self, value) -> Any:
        try:
            self.rule.check(value, will_raise=True)
        except ValueError as err:
            raise ArgumentTypeError(str(err)) from err
        return self.rule.get_value()
