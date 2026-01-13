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

from typing import Union
from msprobe.infer.utils.check.checker import Checker, CheckResult, rule


class DictChecker(Checker):

    @rule()
    def is_dict(self) -> Union["DictChecker", CheckResult]:
        return isinstance(self.instance, dict), "Input object is not a dict."

    @rule()
    def is_dict_not_empty(self) -> Union["DictChecker", CheckResult]:
        is_pass = self.is_dict()
        if not is_pass:
            return is_pass
        return bool(self.instance), "Dict is empty."

    @rule()
    def is_key_exists_in_dict(self, key) -> Union["DictChecker", CheckResult]:
        is_pass = self.is_dict()
        if not is_pass:
            return is_pass
        res = self._find_key_in_dict(self.instance, key)
        return res, f"Key='{key}' is not in the dict."

    @rule()
    def is_values_valid(self, default_value=None, **value_rules: Checker) -> Union["DictChecker", CheckResult]:
        is_pass = self.is_dict().passed
        if not is_pass:
            return is_pass, "Input object is not a dict."
        err_msgs = []
        for key, value_rule in value_rules.items():
            value = self.instance.get(key, default_value)
            rule_is_pass = value_rule.check(value)
            if not rule_is_pass:
                is_pass = False
                err_msgs.append(f"{key} is invalid. {str(rule_is_pass)}")

        return is_pass, "\n".join(err_msgs)

    @rule()
    def is_key_type_valid(self, exp_type) -> Union["DictChecker", CheckResult]:
        is_pass = self.is_dict().passed
        if not is_pass:
            return is_pass
        err_msg = []
        for key in self.instance.keys():
            if not isinstance(key, exp_type):
                is_pass = False
                err_msg.append(f'The type of {key} is not {exp_type.__name__}')
        return is_pass, "\n".join(err_msg)

    def _find_key_in_dict(self, my_dict, my_key):
        if not isinstance(my_dict, dict):
            return False
        if my_key in my_dict:
            return True
        for dict_val in my_dict.values():
            if isinstance(dict_val, dict):
                if self._find_key_in_dict(dict_val, my_key):
                    return True
        return False
