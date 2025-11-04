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

from functools import wraps
from typing import Callable, Union, List
from enum import Enum


class EnumInstance(Enum):
    NO_INSTANCE = 0


class CheckResult:
    def __init__(self, passed=True, msg="") -> None:
        self.passed = bool(passed)
        self.msg = msg

    def __bool__(self):
        return self.passed

    def __str__(self) -> str:
        return self.get_msg()

    def __repr__(self) -> str:
        return f"{self.passed}:{self.msg}"

    def get_msg(self):
        return "pass" if self.passed else self.msg


class WaitingRule:
    def __init__(self, func, args, kwargs, err_msg=None) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.err_msg = f"{self.name} check failed." if err_msg is None else err_msg

    @property
    def name(self):
        return self.func.__name__

    def check(self, instence: "CheckerInstence"):
        passed = self.func(instence, *self.args, **self.kwargs)
        if isinstance(passed, CheckResult):
            return passed
        elif isinstance(passed, tuple) and len(passed) == 2:
            return CheckResult(passed[0], passed[1])
        else:
            return CheckResult(bool(passed), self.err_msg)


class Recorder:
    def __init__(self) -> None:
        self.record_set = set()

    def add_record(self, *names) -> None:
        self.record_set.update(names)

    def in_records(self, name) -> bool:
        return name in self.record_set

    def union(self, recorder: "Recorder"):
        self.record_set.update(recorder.record_set)


class RuleRunner:
    def __init__(self) -> None:
        self.waiting_rules: List[WaitingRule] = []
        self.default_rules: List[WaitingRule] = []
        self.running_default = False
        self.instence = None
        self.recorder = Recorder()

    def add_rule(self, waiting_rule):
        self.waiting_rules.append(waiting_rule)

    def set_recorder(self, recorder: Recorder):
        self.recorder = recorder

    def get_recorder(self) -> Recorder:
        return self.recorder

    def run_rule(self, waiting_rule: WaitingRule):
        if self.running_default and self.recorder.in_records(waiting_rule.name):
            return CheckResult()
        self.recorder.add_record(waiting_rule.name)
        return waiting_rule.check(self.instence)

    def implement_check(self, instence: "CheckerInstence"):
        self.instence = instence
        self.running_default = False

        for waiting_rule in self.waiting_rules:
            passed = self.run_rule(waiting_rule)
            if not passed:
                return passed

        self.running_default = True
        for waiting_rule in self.default_rules:
            passed = self.run_rule(waiting_rule)
            if not passed:
                return passed
        return CheckResult()

    def is_running(self):
        return self.instence is not None

    def as_default(self):
        self.default_rules = self.waiting_rules
        self.waiting_rules = []
        return self


class CheckerInstence:
    def __init__(self, instance=EnumInstance.NO_INSTANCE, converter=None):
        self.instance = instance
        self.converter = converter

    def has_instance(self):
        return self.instance != EnumInstance.NO_INSTANCE

    def get_instance(self):
        return self.instance

    def get_value(self):
        # for args type
        return self.instance

    def convert_instance(self):
        if self.converter is None:
            return CheckResult()

        self.instance, passed, err_msg = self.converter(self.instance)
        return CheckResult(passed, err_msg)


def rule(err_msg=None) -> Callable:
    if not isinstance(err_msg, (type(None), str)):
        raise TypeError("err_msg must be a string")

    def make_rule_wrapper(func: Callable) -> Callable:
        if not callable(func):
            raise ValueError("The @rule decorator can only be applied to functions")

        @wraps(func)
        def wrapper(checker: "Checker", *args, **kwargs) -> Union["Checker", CheckResult]:
            waiting_rule = WaitingRule(func, args, kwargs, err_msg)
            if checker.is_running():
                return checker.run_rule(waiting_rule)
            else:
                checker.add_rule(waiting_rule)

            if checker.has_instance():
                return checker.check(checker.get_instance())

            return checker

        return wrapper

    return make_rule_wrapper


class Checker(RuleRunner, CheckerInstence):
    def __init__(self, instance=EnumInstance.NO_INSTANCE, converter=None) -> None:
        RuleRunner.__init__(self)
        CheckerInstence.__init__(self, instance, converter)

    def check(self, value, will_raise=False) -> CheckResult:
        self.instance = value
        passed = self.convert_instance() and self.implement_check(self)
        if not passed and will_raise:
            raise ValueError(passed)
        return passed

    @rule(err_msg="value is None")
    def not_none(self) -> Union["Checker", CheckResult]:
        return self.instance is not None

    @rule(err_msg="value is not None")
    def is_none(self) -> Union["Checker", CheckResult]:
        return self.instance is None

    @rule()
    def not_eq(self, cmp_value) -> Union["Checker", CheckResult]:
        return self.instance != cmp_value, f"equal {cmp_value}"

    @rule()
    def eq(self, cmp_value) -> Union["Checker", CheckResult]:
        return self.instance == cmp_value, f"not equal {cmp_value}?"

    @rule()
    def is_type(self, *types) -> Union["Checker", CheckResult]:
        return self.not_none() and isinstance(self.instance, types), f"is not {types}"

    @rule()
    def any(self, *rules: "Checker") -> Union["Checker", CheckResult]:
        passed = CheckResult()
        for running_rule in rules:
            if self.running_default:
                running_rule.set_recorder(self.recorder)
            passed = running_rule.check(self.instance)
            if passed:
                if not self.running_default:
                    self.recorder.union(running_rule.get_recorder())
                return passed

        return passed

    @rule()
    def anti(self, anti_rule: "Checker") -> Union["Checker", CheckResult]:
        anti_rule.set_recorder(self.recorder)
        passed = anti_rule.check(self.instance)

        return not passed, f"not {str(passed)}"
