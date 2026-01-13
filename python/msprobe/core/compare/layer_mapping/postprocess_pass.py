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

import re
import math

from msprobe.core.common.const import Const


def postprocess_pass(data_items, name2item):
    backward_pass(data_items, name2item)
    renumber_index_pass(data_items, "ParallelTransformer", "layers")


def backward_pass(data_items, name2item):
    # 处理反向数据，反向无栈信息，沿用正向数据栈信息
    for data_item in data_items:
        data_name_list = data_item.data_name.split(Const.SEP)
        if not data_name_list:
            continue
        if Const.BACKWARD in data_name_list[Const.SCOPE_DIRECTION_INDEX:]:
            data_name_list[Const.SCOPE_DIRECTION_INDEX:] = [
                s.replace(Const.BACKWARD, Const.FORWARD)
                for s in data_name_list[Const.SCOPE_DIRECTION_INDEX:]
            ]
            forward_name = Const.SEP.join(data_name_list)
            forward_item = name2item.get(forward_name, None)
            if not forward_item:
                continue
            data_item.stack_scope = forward_item.stack_scope
            data_item.full_scope = forward_item.full_scope
            data_item.layer_scope = forward_item.layer_scope


def extract_next_item_last_number(data, prefix, default_result=None):
    result = default_result
    match = re.search(rf"^{re.escape(prefix)}\.(\S+?)(?:\.|$)", data)
    if match:
        next_item = match.group(1)
        numbers = re.findall(r"\d+", next_item)
        if numbers:
            result = int(numbers[-1])
    return result


def replace_next_item_index(full_scope, prefix, index):
    if math.isinf(index):
        return full_scope
    prefix_pattern = rf"^{re.escape(prefix)}\."
    result = full_scope
    match = re.search(rf"{prefix_pattern}(\S+?)(?:\.|$)", full_scope)
    if match:
        next_item = match.group(1)
        pattern = rf"{prefix_pattern}{re.escape(next_item)}"
        result = re.sub(pattern, f"{prefix}.{index}", full_scope, count=1)
    return result


def renumber_index_pass(data_items, type_name, suffix=None):
    """
    该函数为解决并行切分场景中编号不一致的比对问题。例如在MindSpore中ParallelTransformer层的PP切分场景，
    MindSpore中的layers的成员编号是全局的，而在PyTorch中编号为局部的。
    为适配此种场景，对指定层的索引进行重新编号，以确保在后续处理阶段序号对齐。
    """
    prefix_dict = {}  # 保存类型为type_name的前缀和最小编号的映射
    for data_item in data_items:
        if data_item.type_name == type_name:
            prefix = f"{data_item.full_scope}.{suffix}" if suffix else data_item.layer_scope
            prefix_dict[prefix] = math.inf

    # 计算前缀对应的最小编号
    for prefix in prefix_dict:
        for data_item in data_items:
            res = extract_next_item_last_number(data_item.full_scope, prefix, math.inf)
            prefix_dict[prefix] = min(prefix_dict[prefix], res)

    # 重新编号
    for prefix, min_index in prefix_dict.items():
        for data_item in data_items:
            full_scope = data_item.full_scope
            abs_index = extract_next_item_last_number(data_item.full_scope, prefix, math.inf)
            rel_index = abs_index - min_index
            full_scope = replace_next_item_index(full_scope, prefix, rel_index)
            data_item.full_scope = full_scope
