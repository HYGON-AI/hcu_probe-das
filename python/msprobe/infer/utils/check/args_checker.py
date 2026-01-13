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
