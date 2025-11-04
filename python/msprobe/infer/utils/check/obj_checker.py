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


class ObjectChecker(Checker):
    @rule()
    def is_attr_valid(self, key, check_rule: Checker, default_value=None) -> Union["ObjectChecker", CheckResult]:
        value = getattr(self.instance, key, default_value)
        return check_rule.check(value)
    
    @rule()
    def is_attrs_valid(self, default_value=None, **attr_rules: Checker) -> Union["ObjectChecker", CheckResult]:
        is_pass = self.not_none()
        if not is_pass:
            return is_pass
        
        err_msgs = []
        for attr_name, attr_rule in attr_rules.items():
            attr_value = getattr(self.instance, attr_name, default_value)
            rule_is_pass = attr_rule.check(attr_value)
            if not rule_is_pass:
                is_pass = False 
                err_msgs.append(f"{attr_name} is invalid. {str(rule_is_pass)}")
        
        return is_pass, "\n".join(err_msgs)