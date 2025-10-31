# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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

import re
from abc import ABC, abstractmethod

from msprobe.core.config_check.utils.utils import config_checking_print
from msprobe.core.common.file_utils import FileOpen, load_yaml
from msprobe.core.common.const import Const, FileCheckConst


class Parser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> dict:
        pass

    def run(self, file_path: str) -> dict:
        """
            统一对外调用接口
        :param file_path: 需解析的文件路径
        :return:
        """
        try:
            result = self.parse(file_path)
        except Exception as exc:
            config_checking_print(f"{self.__class__} parsing error, skip file path: {file_path}, error: {exc}")
            result = {}
        return result


class ShellParser(Parser):
    def parse(self, file_path: str) -> dict:
        """
        Extracts arguments from bash script used to run a model training.
        """
        hyperparameters = {}
        script_content_list = []
        with FileOpen(file_path, 'r') as file:
            for line in file:
                stripped_line = line.lstrip()
                if not stripped_line.startswith('#'):
                    line = line.split('#')[0].rstrip() + '\n'
                    if line.strip():
                        script_content_list.append(line)
        script_content = ''.join(script_content_list)

        command_line = re.search(r'msrun\s[^|]*|torchrun\s[^|]*|python\d? -m torch.distributed.launch\s[^|]*',
                                 script_content,
                                 re.DOTALL)
        if command_line:
            command_line = command_line.group()

            blocks = re.findall(r'([a-zA-Z0-9_]{1,20}_ARGS)="(.*?)"', script_content, re.DOTALL)
            block_contents = {}
            for block_name, block_content in blocks:
                block_content = block_content.replace('\n', ' ')
                block_contents[block_name] = block_content
                command_line = command_line.replace(f"${block_name}", block_content)

            matches = re.findall(r'--([\w-]+)(?:\s+([^\s\\]+))?', command_line)
            for match in matches:
                key, value = match
                args_key = re.match(r'\$\{?(\w+)}?', value)
                if args_key:
                    env_vars = re.findall(rf'{args_key.group(1)}=\s*(.+)', script_content)
                    if env_vars:
                        value = env_vars[-1]
                hyperparameters[key] = value if value else True

        return hyperparameters


class YamlParser(Parser):
    hyperparameters = {}

    def parse(self, file_path: str) -> dict:
        ori_hyper = load_yaml(file_path)
        self.recursive_parse_parameters(ori_hyper, "")
        return self.hyperparameters

    def recursive_parse_parameters(self, parameters, prefix):
        if isinstance(parameters, dict):
            for key, value in parameters.items():
                new_prefix = prefix + Const.SEP + key if prefix else key
                self.recursive_parse_parameters(value, new_prefix)
        elif isinstance(parameters, list):
            if all(isinstance(x, (int, float, str, bool, list))for x in parameters):
                self.hyperparameters.update({prefix: parameters})
            else:
                for idx, value in enumerate(parameters):
                    new_prefix = prefix + Const.SEP + str(idx) if prefix else str(idx)
                    self.recursive_parse_parameters(value, new_prefix)
        elif isinstance(parameters, (int, float, str, bool)):
            self.hyperparameters.update({prefix: parameters})


class ParserFactory:
    __ParserDict = {
        FileCheckConst.SHELL_SUFFIX: ShellParser(),
        FileCheckConst.YAML_SUFFIX: YamlParser()
    }

    def get_parser(self, file_type: str) -> Parser:
        parser = self.__ParserDict.get(file_type, None)
        if not parser:
            raise ValueError(f'Invalid parser type: {file_type}')
        return parser
