# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
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

from abc import ABC, abstractmethod
import re

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import ScopeException


class ScopeFactory:
    def __init__(self, config):
        self.task = config.task
        self.level = config.level
        self.scope = config.scope
        self.api_list = config.list

    def build_scope(self):
        if not self.scope and not self.api_list:
            return None
        if self.scope is None:
            self.scope = []
        if self.api_list is None:
            self.api_list = []
        if self.task == Const.FREE_BENCHMARK:
            return ListScope(self.scope, self.api_list)
        return self._build_range_scope()

    def _build_range_scope(self):
        api_range_scope = APIRangeScope(self.scope, self.api_list, self.level)
        module_range_scope = ModuleRangeScope(self.scope, self.api_list, self.level)
        mix_range_scope = MixRangeScope(self.scope, self.api_list, self.level)

        if self.level == Const.LEVEL_MIX:
            return mix_range_scope

        if not self.scope:
            return api_range_scope
        if api_range_scope.is_valid and module_range_scope.is_valid:
            raise ScopeException(ScopeException.InvalidScope, f"scope={self.scope}.")
        elif api_range_scope.is_valid:
            return api_range_scope
        elif module_range_scope.is_valid:
            return module_range_scope
        else:
            raise ScopeException(ScopeException.InvalidScope, f"scope={self.scope}")


class BaseScope(ABC):
    Module_Type_Module = "Module"
    Module_Type_API = "api"
    module_type = ["Module", "Cell"]

    def __init__(self, scope, api_list, level=None):
        self.level = level
        scope, api_list = self.rectify_args(scope, api_list)
        self.scope = scope
        self.api_list = api_list

    def rectify_args(self, scope, api_list):
        if not isinstance(api_list, list):
            raise ScopeException(ScopeException.InvalidApiStr,
                                 f"api_list参数须配置为列表，实际类型为{type(api_list)}.")
        for api in api_list:
            if not isinstance(api, str):
                raise ScopeException(ScopeException.InvalidApiStr,
                                     f"api_list中的元素须配置为字符串，实际类型为{type(api)}.")
        if isinstance(scope, str):
            scope = [scope]
            return scope, api_list
        if not isinstance(scope, list):
            raise ScopeException(ScopeException.InvalidScope,
                                 f"scope参数须配置为字符串或列表，实际类型为{type(scope)}.")
        for s in scope:
            if not isinstance(s, str):
                raise ScopeException(ScopeException.InvalidScope,
                                     f"scope列表元素要求类型为字符串，实际类型为{type(s)}.")
        return scope, api_list

    @abstractmethod
    def check(self, name):
        pass

    def check_api_list(self, api_name):
        if not self.api_list:
            return True
        for api_str in self.api_list:
            if api_str in api_name:
                return True
        return False


class ListScope(BaseScope):
    def rectify_args(self, scope, api_list):
        if scope and api_list:
            raise ScopeException(ScopeException.ArgConflict,
                                 f"scope和api_list不可以同时配置，实际配置为scope={scope}, api_list={api_list}.")
        return super().rectify_args(scope, api_list)

    def check(self, name):
        if not self.scope or name in self.scope:
            return self.check_api_list(name)
        return False


class RangeScope(BaseScope, ABC):

    def __init__(self, *args):
        super().__init__(*args)
        self.in_scope = False
        self.in_list = False
        self.start_name_set = set()
        self.is_valid = self.check_scope_is_valid()

    def check_name_pattern(self, name):
        options_pattern = "|".join(re.escape(option) for option in Const.DUMP_PREFIX)
        api_pattern = rf"^({options_pattern})\..*\.\d+\.(forward|backward)$"
        module_pattern = r"^(Cell|Module)\..*\.(forward|backward)\.\d+$"

        if self.level == Const.LEVEL_L1:
            if not re.match(api_pattern, name):
                raise ScopeException(ScopeException.InvalidScope,
                                     f"scope参数格式错误，要求格式为api完整命名，实际为{name}.")

        if self.level == Const.LEVEL_L0:
            if not re.match(module_pattern, name):
                raise ScopeException(ScopeException.InvalidScope,
                                     f"scope参数格式错误，要求格式为模块完整命名，实际为{name}.")

        if self.level == Const.LEVEL_MIX:
            if not re.match(api_pattern, name) and not re.match(module_pattern, name):
                raise ScopeException(ScopeException.InvalidScope,
                                     f"scope参数格式错误，要求格式为api或模块完整命名，实际为{name}.")

    def rectify_args(self, scope, api_list):
        scope, api_list = super().rectify_args(scope, api_list)
        if scope and len(scope) != 2:
            raise ScopeException(ScopeException.InvalidScope,
                                 f"scope参数指定区间断点，须传入长度为2的列表，实际长度为{len(scope)}.")
        for name in scope:
            self.check_name_pattern(name)
        return scope, api_list

    @abstractmethod
    def check_scope_is_valid(self):
        pass

    def begin_module(self, module_name):
        pass

    def end_module(self, module_name):
        pass


class APIRangeScope(RangeScope):
    def check_scope_is_valid(self):
        if not self.scope:
            return True
        scope_start_type = self.scope[0].split(Const.SEP)[0]
        if scope_start_type in BaseScope.module_type:
            return False
        scope_stop_type = self.scope[1].split(Const.SEP)[0]
        if scope_stop_type in BaseScope.module_type:
            return False
        return True

    def check(self, name):
        if self.scope and name == self.scope[0]:
            self.in_scope = True

        if not self.scope or self.in_scope:
            result = self.check_api_list(name)
        else:
            result = False

        if self.scope and name == self.scope[1]:
            self.in_scope = False
        return result


class ModuleRangeScope(RangeScope):
    """
        模块与api不同的是，模块内部还有子结构需要dump，
        需要用pre_hook和full_backward_hook来精确控制module的开始和结束，
        在这些hook触发时调用begin_module和end_module做区间控制
    """

    def check_scope_is_valid(self):
        if not self.scope:
            return True
        scope_start_type = self.scope[0].split(Const.SEP)[0]
        scope_stop_type = self.scope[1].split(Const.SEP)[0]
        if scope_start_type in BaseScope.module_type and \
                scope_stop_type in BaseScope.module_type:
            return True
        return False

    def begin_module(self, module_name):
        if not self.scope:
            return
        if module_name == self.scope[0]:
            self.in_scope = True

    def end_module(self, module_name):
        if not self.scope:
            return
        if module_name == self.scope[1]:
            self.in_scope = False

    def check(self, name):
        if not self.scope or self.in_scope:
            return self.check_api_list(name)
        return False


class MixRangeScope(RangeScope):
    def check_scope_is_valid(self):
        return True if self.scope else False

    def begin_module(self, module_name):
        if self.scope and module_name == self.scope[0]:
            self.in_scope = True
        for name in self.api_list:
            if name in module_name:
                self.in_list = True
                self.start_name_set.add(module_name)  # 记录每一个开启in_list的module_name

    def end_module(self, module_name):
        if self.scope and module_name == self.scope[1]:
            self.in_scope = False
        self.start_name_set.discard(module_name)  # 从集合中删除每一个module_name
        if not self.start_name_set:  # 如果集合为空，说明当前module_name是最后一个开启in_list的module_name
            self.in_list = False  # 关闭in_list

    def check_api_list(self, api_name):
        if not self.api_list:
            return True

        for name in self.api_list:
            if name in api_name:
                return True
        return False

    def check(self, name):
        """
        dump时调用的接口，根据scope和api_list判断是否需要dump
        """
        result = False
        if self.scope and name == self.scope[0]:
            self.in_scope = True

        if not self.scope or self.in_scope:
            if self.in_list:
                result = True
            else:
                result = self.check_api_list(name)

        if self.scope and name == self.scope[1]:
            self.in_scope = False
        return result
