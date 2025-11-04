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