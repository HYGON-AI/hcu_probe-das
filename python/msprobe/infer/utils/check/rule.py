# Copyright (c) 2023-2025 Huawei Technologies Co., Ltd.
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

from msprobe.infer.utils.check import NumberChecker, ObjectChecker, StringChecker, DictChecker, PathChecker, ListChecker
from msprobe.infer.utils.check.checker import Checker
from msprobe.infer.utils.constants import INPUT_FILE_MAX_SIZE


class NumConverter:
    def __init__(self, convert_type=float):
        self.convert_type = convert_type

    def __call__(self, value) -> Any:
        return self.convert(value)

    def convert(self, value: str):
        try:
            if self.convert_type == int:
                return int(value), True, ""
            elif self.convert_type == float:
                return float(value), True, ""
            else:
                return value, True, ""
        except ValueError as er:
            return value, False, str(er)


class Rule:
    @staticmethod
    def none() -> Checker:
        return Checker().is_none()

    @staticmethod
    def num() -> NumberChecker:
        return NumberChecker().is_number()

    @staticmethod
    def str() -> StringChecker:
        return StringChecker().is_str()

    @staticmethod
    def dict() -> DictChecker:
        return DictChecker().is_dict()

    @staticmethod
    def obj(obj_type) -> ObjectChecker:
        return ObjectChecker().is_type(obj_type)

    @staticmethod
    def path() -> PathChecker:
        return PathChecker()

    @staticmethod
    def list() -> ListChecker:
        return ListChecker().is_list()

    @staticmethod
    def config_file() -> PathChecker:
        return (
            PathChecker()
            .exists()
            .is_file()
            .is_readable()
            .is_not_writable_to_others()
            .is_safe_parent_dir()
            .max_size(10 * 1000 * 1000)
            .as_default()
        )

    @staticmethod
    def input_file() -> PathChecker:
        return (
            PathChecker()
            .exists()
            .forbidden_softlink()
            .is_file()
            .is_readable()
            .is_owner()
            .is_not_writable_to_others()
            .is_safe_parent_dir()
            .max_size(INPUT_FILE_MAX_SIZE)
            .as_default()
        )

    @staticmethod
    def input_dir() -> PathChecker:
        return PathChecker().exists().is_dir().is_readable().is_uid_matched().is_not_writable_to_others().as_default()

    @staticmethod
    def output_dir() -> PathChecker:
        return (
            Rule.path()
            .any(Rule.anti(PathChecker().exists()), PathChecker().is_dir().is_writeable().is_not_writable_to_others())
            .as_default()
        )

    @staticmethod
    def any(*rules: Checker) -> Checker:
        return Checker().any(*rules)

    @staticmethod
    def anti(rule: Checker) -> Checker:
        return Checker().anti(rule)

    @staticmethod
    def to_int() -> NumberChecker:
        return NumberChecker(converter=NumConverter(int))

    @staticmethod
    def to_float() -> NumberChecker:
        return NumberChecker(converter=NumConverter(float))
