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
