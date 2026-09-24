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

import inspect
from typing import Callable
from functools import wraps


class FuncWrapper:
    def __init__(self, check_rules: dict) -> None:
        self.check_rules = check_rules
        self.args_names = []
        self.args_name_set = set()
        self.var_keyword_rule = None

    def is_function_param_valid(self, to_raise, *args, **kwargs) -> bool:
        if kwargs is not None:
            for args_name, arg_value in kwargs.items():
                check_rule = self.check_rules.get(args_name)
                if check_rule is None and args_name not in self.args_name_set and self.var_keyword_rule is not None:
                    check_rule = self.var_keyword_rule
                if check_rule is None:
                    continue

                is_pass = check_rule.check(arg_value, will_raise=to_raise)
                if not is_pass:
                    return bool(is_pass), f"{args_name} is invalid. {str(is_pass)}"

        if args is not None and len(self.args_names) > 0:
            for index, arg_value in enumerate(args):

                if len(self.args_names) > index:
                    args_name = self.args_names[index]
                else:
                    args_name = self.args_names[-1]

                check_rule = self.check_rules.get(args_name)
                if check_rule is None:
                    continue

                is_pass = check_rule.check(arg_value, will_raise=to_raise)
                if not is_pass:
                    return bool(is_pass), f"{args_name} is invalid. {str(is_pass)}"
        return True

    def parse_function(self, func):
        parameters = inspect.signature(func).parameters

        for param in parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                self.var_keyword_rule = self.check_rules.get(param.name)
            else:
                self.args_names.append(param.name)
                if param.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]:
                    self.args_name_set.add(param.name)

    def create_wrapper(self, ret_value, to_raise, logger=None) -> Callable:
        def decorator(func) -> Callable:
            self.parse_function(func)

            @wraps(func)
            def wrapper(*args, **kwargs):
                is_pass = self.is_function_param_valid(to_raise, *args, **kwargs)
                if isinstance(is_pass, bool) and is_pass:
                    return func(*args, **kwargs)
                else:
                    if logger is not None and hasattr(logger, "error"):
                        logger.error(str(is_pass))
                    return ret_value

            return wrapper

        return decorator

    def to_return(self, ret_value, logger=None) -> Callable:
        return self.create_wrapper(ret_value, False, logger)

    def to_raise(self) -> Callable:
        return self.create_wrapper(None, True, None)


def validate_params(**check_rules) -> FuncWrapper:
    return FuncWrapper(check_rules)