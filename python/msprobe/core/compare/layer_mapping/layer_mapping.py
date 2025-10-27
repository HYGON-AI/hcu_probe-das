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
from collections import defaultdict

from python.msprobe.core.common.const import CompareConst, Const
from python.msprobe.core.common.file_utils import load_json, load_yaml, save_yaml
from python.msprobe.core.common.utils import add_time_with_yaml, detect_framework_by_dump_json, \
    get_stack_construct_by_dump_json_path, CompareException
from python.msprobe.core.compare.layer_mapping.data_scope_parser import get_dump_data_items
from python.msprobe.core.compare.utils import read_op, reorder_op_name_list
from python.msprobe.core.common.decorator import recursion_depth_decorator
from python.msprobe.core.common.log import logger


class LayerTrie:
    def __init__(self, type_name, framework=None):
        self.type_name = type_name
        self.data_items = defaultdict(list)
        self.children = {}
        self.framework = framework

    def __repr__(self):
        data_nums = [{k: len(v)} for k, v in self.data_items.items()]
        return f"Layer(type_name={self.type_name}, data_number={data_nums})"

    def get(self, name):
        return self.children.get(name)

    def insert(self, data_item):
        parts = data_item.full_scope.split(Const.SEP)
        node = self
        scope_name_list = parts[Const.RIGHT_MOVE_INDEX:]

        for name in scope_name_list:
            if name not in node.children:
                node.children[name] = LayerTrie(name, data_item.framework)
            node = node.children[name]
        node.data_items[data_item.state].append(data_item)
        node.type_name = data_item.type_name

    def query_data(self, scope, state, index, default_value=None):
        parts = scope.split(Const.SEP)
        node = self
        scope_name_list = parts[1:]

        for name in scope_name_list:
            if name not in node.children:
                return default_value
            node = node.children[name]
        if index >= len(node.data_items[state]):
            return default_value
        if node.data_items[state]:
            return node.data_items[state][index]
        else:
            logger.error(f"node.data_items of state:{state} is empty, please check.")
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR)

    def save_to_yaml(self, output_path):
        result = {f"{self.type_name} @ {self}": self.convert_to_dict(self)}
        file_name = add_time_with_yaml(f"{self.framework}_tree")
        file_path = os.path.join(os.path.realpath(output_path), file_name)
        save_yaml(file_path, result)

    @recursion_depth_decorator("LayerMapping: LayerTrie.convert_to_dict", max_depth=100)
    def convert_to_dict(self, node):
        result = {}
        result["data_item"] = {st: [dt.data_name for dt in dts] for st, dts in node.data_items.items()}
        for child_key, child_node in node.children.items():
            key = f"{child_key} @ {child_node}"
            result[key] = self.convert_to_dict(child_node)
        return result


def convert_scope(layer_trie, data_item, mapping=None):
    if not mapping:
        mapping = {}
    new_scope = Const.TOP_LAYER
    scope_list = data_item.full_scope.split(Const.SEP)
    cur_node = layer_trie

    idx = 0
    while idx < len(scope_list) - 1:
        child_name = scope_list[idx + 1]
        type_name = cur_node.type_name
        prefix_mapping = mapping.get(type_name, {})
        mapping_list = prefix_mapping.get(child_name, [])
        mapping_list.append((child_name, child_name, 1))
        step = 1
        for origin, target, level in mapping_list:
            if Const.SEP.join(scope_list[idx + 1: idx + level + 1]) == origin:
                new_scope = new_scope + Const.SEP + target
                step = level
                break
        for _ in range(step):
            child_node = cur_node.get(scope_list[idx + 1])
            cur_node = child_node
            idx += 1
    index = -1
    state = data_item.state
    for idx, child in enumerate(cur_node.data_items[state]):
        if data_item.data_name == child.data_name:
            index = idx
    return new_scope, state, index


def get_data_items_and_tree(dump_json_path, output_path):
    framework = detect_framework_by_dump_json(dump_json_path)
    stack, construct = get_stack_construct_by_dump_json_path(dump_json_path)
    dump = load_json(dump_json_path)
    dump_data_items = get_dump_data_items(dump, stack, construct, framework, output_path)
    root = LayerTrie(Const.TOP_LAYER, framework)
    for data_item in dump_data_items:
        root.insert(data_item)
    if output_path:
        root.save_to_yaml(output_path)
    return dump_data_items, root


def convert_data_item(npu_tree, bench_tree, npu_data_item, mapping):
    new_scope, state, index = convert_scope(npu_tree, npu_data_item, mapping)
    bench_data_item = bench_tree.query_data(new_scope, state, index)
    return bench_data_item


def update_keys_in_place(d):
    """
    This function is used to compare and maintain compatibility between the old and new versions.
    In the old version, 'Cell' was used as the top layer name, while the new version uses 'TopLayer'.
    """
    cell_value = d.pop(Const.CELL, None)

    if cell_value is not None:
        d[Const.TOP_LAYER] = cell_value


def preprocess_layer_mapping(mapping):
    """
    before:
        {'A': {'a.b.c': 'new_c',
               'a.demo': 'new_demo',
               'z': 'new_z',
               'd.e': 'e'}}
    after:
        {'A': {'a': [('a.b.c', 'new_c', 3), ('a.demo', 'new_demo', 2)],
               'z': [('z', 'new_z', 1)],
               'd': [('d.e', 'e', 2)]}}
    """
    update_keys_in_place(mapping)
    final_mapping = {}

    for type_name, name_map in mapping.items():
        final_mapping[type_name] = {}

        for key, value in name_map.items():
            key_list = key.split('.')
            prefix = key_list[0]  # 取前缀
            value_list = value.split('(')
            value = value_list[0]  # 取前缀
            key_len = len(key_list)
            if prefix not in final_mapping[type_name]:
                final_mapping[type_name][prefix] = []
            final_mapping[type_name][prefix].append((key, value, key_len))

        # 前缀映射列表按规则长度排序
        for prefix in final_mapping[type_name]:
            final_mapping[type_name][prefix].sort(key=lambda x: -x[-1])

    return final_mapping


def convert_data_items(npu_tree, bench_tree, npu_data_items, mapping):
    mapping = preprocess_layer_mapping(mapping)
    api_mapping = {}
    for npu_data_item in npu_data_items:
        bench_data_item = convert_data_item(npu_tree, bench_tree, npu_data_item, mapping)
        bench_name = bench_data_item.data_name if bench_data_item else CompareConst.N_A
        npu_name = npu_data_item.data_name
        api_mapping[npu_name] = bench_name
    return api_mapping


def generate_api_mapping_by_layer_mapping(npu_json_path, bench_json_path, layer_mapping_path=None, output_path=None):
    npu_data_items, npu_root = get_data_items_and_tree(npu_json_path, output_path)
    _, bench_root = get_data_items_and_tree(bench_json_path, output_path)
    if isinstance(layer_mapping_path, str):
        mapping = load_yaml(layer_mapping_path)
    else:
        mapping = {}
    api_mapping = convert_data_items(npu_root, bench_root, npu_data_items, mapping)
    if output_path:
        file_name = add_time_with_yaml("api_mapping")
        file_path = os.path.join(os.path.realpath(output_path), file_name)
        save_yaml(file_path, api_mapping)
    return api_mapping


def generate_data_mapping(npu_json_path, bench_json_path, api_mapping, output_path=None):
    def read_full_op_names(data, op_name):
        op_parsed_list = read_op(data.get(op_name, {}), op_name)
        full_op_names = [op_parsed.get('full_op_name') for op_parsed in op_parsed_list]
        states = [op_parsed.get(Const.STATE) for op_parsed in op_parsed_list]
        return full_op_names, states

    def generate_op_data_mapping(npu_op_name, npu_full_op_names, bench_op_name, bench_full_op_names):
        suffix_to_full_op_name = {}
        op_data_mapping = {}
        for bench_full_op_name in bench_full_op_names:
            suffix = bench_full_op_name[len(bench_op_name):]
            suffix_to_full_op_name[suffix] = bench_full_op_name

        for npu_full_op_name in npu_full_op_names:
            suffix = npu_full_op_name[len(npu_op_name):]
            op_data_mapping[npu_full_op_name] = suffix_to_full_op_name.get(suffix, CompareConst.N_A)
        return op_data_mapping

    npu_data = load_json(npu_json_path).get("data", {})
    bench_data = load_json(bench_json_path).get("data", {})
    data_mapping = {}
    for npu_op_name, bench_op_name in api_mapping.items():
        if not npu_op_name:
            continue
        npu_full_op_names, npu_states = read_full_op_names(npu_data, npu_op_name)
        bench_full_op_names, bench_states = read_full_op_names(bench_data, bench_op_name)
        npu_full_op_names_reorder, _ = reorder_op_name_list(npu_full_op_names, npu_states)
        bench_full_op_names_reorder, _ = reorder_op_name_list(bench_full_op_names, bench_states)
        mapping = generate_op_data_mapping(npu_op_name, npu_full_op_names_reorder,
                                           bench_op_name, bench_full_op_names_reorder)
        data_mapping.update(mapping)
    if output_path:
        file_name = add_time_with_yaml("data_mapping")
        file_path = os.path.join(os.path.realpath(output_path), file_name)
        save_yaml(file_path, data_mapping)
    return data_mapping


def generate_data_mapping_by_layer_mapping(input_param, layer_mapping_path=None, output_path=None):
    npu_json_path = input_param.get("npu_json_path")
    bench_json_path = input_param.get("bench_json_path")
    api_mapping = generate_api_mapping_by_layer_mapping(
        npu_json_path, bench_json_path, layer_mapping_path)
    data_mapping = generate_data_mapping(
        npu_json_path, bench_json_path, api_mapping, output_path)
    return data_mapping
