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
import os.path
from dataclasses import dataclass, field
from typing import Set

from msprobe.core.common.file_utils import load_yaml
from msprobe.core.overflow_check.api_info import APIInfo
from msprobe.core.overflow_check.utils import has_nan_inf

cur_path = os.path.dirname(os.path.realpath(__file__))


class IgnoreFilter:
    def __init__(self, rule_path=os.path.join(cur_path, './ignore_rules.yaml')):
        self.rules = dict()
        self._load_rules(rule_path)

    def has_api_rule(self, api_name: str) -> bool:
        return api_name in self.rules.keys()

    def apply_filter(self, api_info: APIInfo) -> bool:
        """
            应用过滤规则，返回是否需要被过滤
        Args:
            api_info: API调用信息
        Returns:
            是否为误检，是否需要过滤
        """
        torch_api = api_info.torch_api_name
        if not self.has_api_rule(torch_api):
            return False
        rule = self.rules.get(torch_api)
        if not rule.match(api_info):
            return False
        return True

    def _load_rules(self, rule_file_path):
        if self.rules and len(self.rules):
            return
        data = load_yaml(rule_file_path)
        self.rules = dict()
        for rule_item in data.get('ignore_nan_inf', []):
            rule = Rule(
                api_name=rule_item.get('api_name', ''),
                desc=rule_item.get('description', ''),
                input_ignore=rule_item.get('input_ignore', []),
                output_ignore=rule_item.get('output_ignore', [])
            )
            if not rule.verify_field():
                continue
            if self.has_api_rule(rule.api_name):
                continue
            self.rules[rule.api_name] = rule


class Rule:

    def __init__(self, api_name, desc='', input_ignore=None, output_ignore=None):
        self.api_name = api_name
        self.desc = desc
        self.input_ignore = IgnoreItem()
        self.output_ignore = IgnoreItem()
        self._init_ignore(input_ignore, output_ignore)

    def __repr__(self):
        return (f'Rule(api_name={self.api_name}, desc={self.desc}, input_ignore={self.input_ignore}, output_ignore='
                f'{self.output_ignore})')

    def verify_field(self):
        if self.api_name == '':
            return False
        # 若无输入输出规则长度，则为无效规则
        if not (len(self.input_ignore.index) + len(self.input_ignore.name) + len(self.output_ignore.index)):
            return False
        return True

    def match(self, api_info: APIInfo) -> bool:
        """
            匹配API信息是否符合规则
        Returns:
            bool: True if the api_info matches this rule, False otherwise
        """
        # 首先检查API名称是否匹配
        api_name = api_info.torch_api_name
        if api_name != self.api_name:
            return False

        # 检查输入参数中的NaN/Inf
        if self.input_ignore.index and len(api_info.input_args):
            for idx, arg in enumerate(api_info.input_args):
                if has_nan_inf(arg) and not self.input_ignore.has_index(idx):
                    return False

        # 检查输入kwargs中的NaN/Inf
        if self.input_ignore.name and len(api_info.input_kwargs):
            for name, value in api_info.input_kwargs.items():
                if has_nan_inf(value) and not self.input_ignore.has_name(name):
                    return False

        # 检查输出中的NaN/Inf
        if self.output_ignore.index and len(api_info.output_data):
            for idx, out in enumerate(api_info.output_data):
                if has_nan_inf(out) and not self.output_ignore.has_index(idx):
                    return False

        return True

    def _init_ignore(self, input_ignore=None, output_ignore=None):
        """初始化忽略项"""
        if input_ignore is None:
            input_ignore = []
        if output_ignore is None:
            output_ignore = []

        # 处理输入忽略规则
        for item in input_ignore:
            if 'index' in item:
                self.input_ignore.add_index(item['index'])
            if 'name' in item:
                self.input_ignore.add_name(item['name'])

        # 处理输出忽略规则
        for item in output_ignore:
            if 'index' in item:
                self.output_ignore.add_index(item['index'])


@dataclass
class IgnoreItem:
    """存储需要忽略的索引和名称"""
    index: Set[int] = field(default_factory=set)
    name: Set[str] = field(default_factory=set)

    def add_index(self, idx: int):
        self.index.add(idx)

    def add_name(self, name: str):
        self.name.add(name)

    def has_index(self, idx: int) -> bool:
        return idx in self.index

    def has_name(self, name: str) -> bool:
        return name in self.name
