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