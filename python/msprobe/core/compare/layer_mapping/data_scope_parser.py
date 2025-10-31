# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple

import yaml
from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import save_yaml
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException, add_time_with_yaml
from msprobe.core.compare.layer_mapping.postprocess_pass import postprocess_pass


@dataclass
class DumpDataItem:
    framework: str
    data_name: Optional[str] = None
    api_type: Optional[str] = None
    api_name: Optional[str] = None
    type_name: Optional[str] = None
    full_scope: str = ""
    layer_scope: str = ""
    stack_scope: str = ""
    frame_stack_scope: str = ""
    user_stack_scope: str = ""
    construct_scope: str = ""
    scope_direction: Optional[str] = None
    scope_id: Optional[int] = None
    state: str = ""

    # 类变量使用 ClassVar
    layernames: ClassVar[set] = {Const.CELL, Const.MODULE}
    framework2stack_sign: ClassVar[Dict[str, Tuple[str, str]]] = {
        Const.MS_FRAMEWORK: ("Template", "construct"),
        Const.PT_FRAMEWORK: ("Template", r"in (for|back)ward,")
    }

    @staticmethod
    def check_stack_valid(stack_info):
        if stack_info is not None:
            if not isinstance(stack_info, list):
                logger.error(f"stack is invalid, it should be a list[str], but got {stack_info}")
                raise CompareException(CompareException.INVALID_DATA_ERROR)
            for stack in stack_info:
                if not isinstance(stack, str):
                    logger.error(f"stack is invalid, it should be a list[str], but got {stack_info}")
                    raise CompareException(CompareException.INVALID_DATA_ERROR)

    def set(self, data_name: str, construct_info: str, stack_info: str) -> None:
        self.set_name(data_name)
        self.set_layer_scope(construct_info)
        self.set_stack_scope(stack_info)
        self.set_full_scope()

    def set_name(self, data_name: str) -> None:
        self.data_name = data_name
        data_name_list = data_name.split(Const.SEP)
        if not data_name_list or len(data_name_list) < abs(Const.LAYER_NAME_INDEX):
            logger.error(
                f"The dump data does not comply with the format specification and "
                f"must contain no less than four fields. "
                f"The current data is {data_name}"
            )
            raise CompareException(CompareException.INVALID_DATA_ERROR)

        if data_name_list[Const.LAST_INDEX] == Const.PARAMS_GRAD:
            self.api_type = Const.PARAMS_GRAD
            self.api_name = data_name_list[Const.PARAMS_GRAD_NAME_INDEX]
            self.type_name = data_name_list[Const.PARAMS_GRAD_TYPE_NAME_INDEX]
            self.state = Const.PARAMS_GRAD
            return

        self.api_type = data_name_list[Const.API_TYPE_INDEX]
        self.type_name = data_name_list[Const.TYPE_NAME_INDEX]
        if self.api_type in self.layernames:
            self.api_name = data_name_list[Const.LAYER_NAME_INDEX]
            self.state = data_name_list[Const.SCOPE_DIRECTION_INDEX]
        else:
            self.api_name = self.type_name
            self.state = data_name_list[Const.LAST_INDEX]

    def set_layer_scope(self, construct_info: str) -> None:
        self.construct_scope = construct_info
        if self.api_type in self.layernames:
            # remove api name
            data_list = self.data_name.split(Const.SEP)
            data_list = data_list[:Const.LAYER_NAME_INDEX] + data_list[Const.TYPE_NAME_INDEX:]
        elif self.api_type == Const.PARAMS_GRAD:
            data_list = self.data_name.split(Const.SEP)
        elif construct_info:
            data_list = construct_info.split(Const.SEP)
        else:
            data_list = []

        if data_list:
            self.layer_scope = Const.SEP.join(data_list[:Const.TYPE_NAME_INDEX])
        else:
            self.layer_scope = Const.TOP_LAYER
        if construct_info and Const.SEP in construct_info:
            construct_list = construct_info.split(Const.SEP)
            if len(construct_list) < abs(Const.LAYER_NAME_INDEX):
                logger.error(
                    f"The construct data does not comply with the format specification and "
                    f"must contain no less than four fields. "
                    f"The current data is {construct_info}"
                )
                raise CompareException(CompareException.INVALID_DATA_ERROR)
            self.scope_id = construct_list[Const.SCOPE_ID_INDEX]
            self.scope_direction = construct_list[Const.SCOPE_DIRECTION_INDEX]

    def set_stack_scope(self, stack_info: str) -> None:
        # Cell/Module has no stack info
        if self.api_type in self.layernames:
            return

        if self.api_type in Const.DATA_TYPE_SKIP_LIST or not stack_info:
            return

        start_sign, end_sign = self.framework2stack_sign.get(self.framework)
        self.check_stack_valid(stack_info)
        start_pos, end_pos = find_regard_scope(stack_info, start_sign, end_sign)
        # 获取指定范围的代码
        regard_scope = stack_info[start_pos + 1:end_pos]
        frame_func_stack_list, user_func_stack_list = find_stack_func_list(regard_scope)
        self.frame_stack_scope = Const.SEP.join(frame_func_stack_list)
        self.user_stack_scope = Const.SEP.join(user_func_stack_list)

    def set_full_scope(self, use_user_func_scope=False, use_frame_func_scope=True) -> None:
        scope_list = [self.layer_scope]
        if use_user_func_scope and self.user_stack_scope:
            scope_list.append(self.user_stack_scope)
        if use_frame_func_scope and self.frame_stack_scope:
            scope_list.append(self.frame_stack_scope)
        scope_list.append(self.api_name)
        self.full_scope = Const.SEP.join(scope_list)


def find_regard_scope(lines, start_sign, end_sign):
    # 找出 start_pos 和 end_pos
    start_pos = -1
    end_pos = len(lines)
    for idx, ii in enumerate(lines):
        if re.search(start_sign, ii):
            start_pos = idx
        elif start_pos >= 0 and re.search(end_sign, ii):
            end_pos = idx
            break
    return start_pos, end_pos


def find_stack_func_list(lines, record_user=True):
    res_list = []
    user_stack = []
    frame_stack = None
    no_entrance = True
    for line in lines:
        ele_list = line.split(Const.COMMA)
        file_ele = ele_list[Const.STACK_FILE_INDEX]
        # if framework func line and no framework entrance found yet
        if any(ii in file_ele for ii in Const.FRAME_FILE_LIST) and no_entrance:
            frame_stack = line  # Update the last target index
        else:
            if record_user:
                user_stack.append(line)
            no_entrance = False

    # Check if the last string in the list contains target str
    if frame_stack and no_entrance:
        no_entrance = False

    # 过滤和处理 regard_scope
    frame_func = get_stack_in_lines([frame_stack])
    user_func = get_stack_in_lines(user_stack)
    return (frame_func, user_func)


def get_stack_in_lines(simplified: List[str]):
    res_list = []
    if not simplified:
        return res_list
    for line in simplified:
        if not line:
            continue

        ele_list = line.split(Const.COMMA)
        file_ele = ele_list[Const.STACK_FILE_INDEX]
        if any(ii in file_ele for ii in Const.FILE_SKIP_LIST):
            continue

        func_ele = ele_list[Const.STACK_FUNC_INDEX]
        if any(ii in func_ele for ii in Const.FUNC_SKIP_LIST):
            continue

        in_func_name = func_ele.split()[Const.STACK_FUNC_ELE_INDEX]

        res_list.append(in_func_name)

    reversed_list = res_list[::-1]
    return reversed_list


def dumpdata_representer(dumper, data):
    d = deepcopy(data.__dict__)
    d.pop("data_name")
    return dumper.represent_dict(d)


def get_dump_data_items(dump, stack, construct, framework, output_path=None):
    if not stack or not construct:
        return []
    name2item = {}
    data_items = []

    dump_data = dump.get("data", {})
    for data_name in dump_data:
        code_info = stack.get(data_name, None)
        parent_info = construct.get(data_name, None)
        data_item = DumpDataItem(framework)
        data_item.set(data_name, parent_info, code_info)
        name2item[data_name] = data_item
        data_items.append(data_item)

    postprocess_pass(data_items, name2item)

    if output_path:
        yaml.add_representer(DumpDataItem, dumpdata_representer)
        file_name = add_time_with_yaml(f"{framework}_data")
        file_path = os.path.join(os.path.realpath(output_path), file_name)
        save_yaml(file_path, name2item)
    return data_items
