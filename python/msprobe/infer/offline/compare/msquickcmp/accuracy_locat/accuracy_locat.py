# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
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

from collections import deque
from collections import OrderedDict
import re
import os

import numpy as np

from msprobe.infer.utils.security_check import ms_makedirs
from msprobe.infer.utils.util import load_file_to_read_common_check


def calculate_flow(graph, startnode, endnode):
    """
    Function:
        simplifying the graph by using flow calculation to a linear node list
    Return:
        a node list which is linear
    """
    #误差限
    eps = 1e-10
    lin = 0
    for output_name in startnode.outputs:
        for next_node in graph.get_next_nodes(output_name):
            if next_node is not None:
                lin += 1
    if lin < 512:
        lin *= 512

    flow = {}
    incnt = {}
    for node in graph.nodes:
        flow[node.name] = float(0)
        incnt[node.name] = len(node.inputs)
    flow[startnode.name] = float(lin)
    satisfied_nodes = []
    visited = set()
    queue = deque([(startnode, flow.get(startnode.name))])
    visited.add(startnode)
    while queue:
        current_node, current_flow = queue.popleft()
        if abs(current_flow - lin) < eps:
            satisfied_nodes.append(current_node)
        outdegree = 0
        for output_name in current_node.outputs:
            for next_node in graph.get_next_nodes(output_name):
                if next_node is not None:
                    outdegree += 1

        if outdegree != 0:
            flow_increment = float(current_flow) / float(outdegree)
        for output_name in current_node.outputs:
            for next_node in graph.get_next_nodes(output_name):
                if next_node is not None:
                    flow[next_node.name] += flow_increment
                    incnt[next_node.name] -= 1
                if next_node is not None and check_node_valid(incnt, graph, next_node):
                    queue.append([next_node, flow.get(next_node.name)])
                    visited.add(next_node)
    return satisfied_nodes


def find_npy_files_with_prefix(workdir, prefix):
    """
    Function:
        according given prefix list, find all the satisfied files
    Return:
        a matching file path list
    """
    pattern = r'^{}.*\.npy'.format(re.escape(prefix))
    regex = re.compile(pattern)
    matched_files = []
    for root, _, files in os.walk(workdir):
        for file in files:
            if regex.match(file):
                matched_files.append(os.path.join(root, file))
    return matched_files


def create_bin_file(out_path, matched_files):
    """
    Function:
        convert all the matched_files in npy format
        to bin format
    Return:
        bin file path list
    """
    bin_files_list = []
    bin_file_path = './tmp'
    bin_file_path = os.path.join(out_path, bin_file_path)
    bin_file_path = os.path.realpath(bin_file_path)
    if not os.path.exists(bin_file_path):
        ms_makedirs(bin_file_path)
    for i, npy_file in enumerate(matched_files):
        npy_file = load_file_to_read_common_check(npy_file)
        data = np.load(npy_file)
        bin_file_name = f'{i}.bin'
        bin_file = os.path.join(bin_file_path, bin_file_name)
        bin_files_list.append(bin_file)
        data.tofile(bin_file)
    bin_files_name_list = ','.join(bin_files_list)
    return bin_files_name_list


def input_completion(og, inputs_list):
    """
    Function:
        find all the inputs needed according to inputs_list
        generate a need list
    Return:
        return a need file name list
    """
    input_need_list = []
    index = 0
    for node_input in inputs_list:
        input_node = og.get_prev_node(node_input[0])
        if input_node is None:
            continue
        for i, pre_input in enumerate(input_node.inputs):
            if pre_input == node_input[0]:
                index = i
                break
        input_need_list.append(f"{input_node.name}.{index}.")
    input_need_list = list(OrderedDict.fromkeys(input_need_list))
    return input_need_list


def check_node_valid(incnt, graph, node):
    """
    Function:
        check node is the current input node in graph
        using incnt to present the incount of node
    Return:
        true if node is the current input node of graph
        false otherwise
    """
    if incnt.get(node.name) == 0:
        return True
    else:
        emp_cnt = 0
        for node_input in node.inputs:
            input_node = graph.get_prev_node(node_input)
            if input_node is None:
                emp_cnt += 1
        if emp_cnt == incnt.get(node.name):
            return True
    return False


def check_input_node(og, node):
    """
    Function:
        check node is an input node in model og
    Return:
        true if check node is an input node in model og
        false otherwise
    """
    input_cnt = 0
    for node_input in node.inputs:
        input_node = og.get_prev_node(node_input)
        if input_node is None:
            input_cnt += 1
    if input_cnt == len(node.inputs):
        return True
    return False


def check_res(res, endnode):
    """
    check result rows
    check error is relative to endnode
    """
    for row in res:
        gdt_name_list = row["GroundTruth"].split(",")
        for ground_truth_name in gdt_name_list:
            if ground_truth_name == endnode.name:
                return True
    return False