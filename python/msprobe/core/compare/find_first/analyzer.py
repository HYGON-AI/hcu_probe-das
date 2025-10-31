# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

import time
from collections import defaultdict
import os
from itertools import dropwhile, chain

from msprobe.core.common.file_utils import save_json, make_dir
from msprobe.core.common.log import logger
from msprobe.core.common.const import Const
from msprobe.core.compare.utils import compare_distributed_inner
from msprobe.core.compare.find_first.utils import (RankPath, FileCache, is_communication_op, is_ignore_op,
                                                   DiffAnalyseConst, analyze_diff_in_group)
from msprobe.core.compare.find_first.graph import DataNode, CommunicationNode


class DiffAnalyzer:
    def __init__(self, npu_path, bench_path, output_path):
        self._bench_path = bench_path
        self._npu_path = npu_path
        self._output_path = output_path
        self._paths = {}
        self._diff_nodes = []  # 记录所有异常节点
        self._cache = FileCache()
        self._first_comm_nodes = {}  # 记录各rank下首个通信节点的node_id
        self._after_comm_diffs = {}  # 记录各rank下发生在通信节点之后的异常计算节点
        self._rank_comm_nodes_dict = {}  # 记录各rank的通信节点

    def analyze(self):
        self._pre_process()
        for analyze_func in [self._pre_analyze, self._analyze, self._post_analyze]:
            analyze_func()
            if self._diff_nodes:
                self._gen_analyze_info()
                return
        logger.info('Cannot find any diff node, no need to generate analyze file.')

    def _pre_process(self):
        logger.info("Start comparing data ......")
        compare_distributed_inner(self._npu_path, self._bench_path, self._output_path, first_diff_analyze=True)
        self._resolve_input_path(self._output_path)
        logger.info("Pre Process completed.")

    """
    这里需要生成stack，但是直接用dict中自带就行，在op_items.NPU_Stack_Info中
    """
    def _resolve_input_path(self, result_input_path):
        contents = os.listdir(result_input_path)
        rank_paths = {}
        
        for path in contents:
            # 检查文件名是否符合compare_result_rank{rank_id}_{timestamp}.json格式
            if not path.startswith('compare_result_rank'):
                continue
            if not path.endswith('.json'):
                continue
                
            # 从文件名中提取rank_id
            try:
                path_ele_list = path.split('_')
                if len(path_ele_list) <= 2:
                    continue
                rank_part = path_ele_list[2]
                if not rank_part.startswith('rank'):
                    continue
                rank_str = rank_part.strip('rank')  # 去掉'rank'前缀
                rank = int(rank_str) if rank_str else 0
            except (IndexError, ValueError):
                continue
                
            # 构建完整的json文件路径
            dump_path = os.path.join(result_input_path, path)
            rank_paths[rank] = RankPath(rank, dump_path)
            
        # 按照rank id排序后添加到self._paths中
        for rank in sorted(rank_paths.keys()):
            self._paths[rank] = rank_paths[rank]

    def _pre_analyze(self):
        logger.info('Start searching diff node before communication.')
        for path in self._paths.values():
            dump_data = self._cache.load_json(path.dump_path)
            if not dump_data:
                logger.warning(f'Rank {path.rank} has no dump data!')
                continue
            for op_name, op_data in dump_data.items():
                if is_ignore_op(op_name):
                    continue
                if is_communication_op(op_name):
                    self._first_comm_nodes[path.rank] = op_name
                    break
                data_node = DataNode(op_name, path.rank, op_data)
                if data_node.is_diff:
                    self._diff_nodes.append(data_node)
                    break

    def _analyze(self):
        logger.info('Start searching diff node during communication.')
        self._rank_comm_nodes_dict = {rank: self._analyze_comm_nodes(rank) for rank in self._paths}
        self._connect_comm_nodes()
        self._pruning()
        self._search_first_diff()

    def _post_analyze(self):
        logger.info('Start searching diff node after communication.')
        for nodes in self._after_comm_diffs.values():
            if nodes:
                self._diff_nodes.append(nodes[0])

    def _connect_comm_nodes(self):
        searched_ranks = set()
        for rank, nodes in list(self._rank_comm_nodes_dict.items())[:-1]:
            searched_ranks.add(rank)
            seen_nodes = set()
            last_node = None
            for cur_node in nodes.values():
                is_overflow = last_node and hasattr(last_node, 'layer') and hasattr(cur_node, 'layer') and \
                last_node.layer >= cur_node.layer
                if is_overflow:
                    cur_node.layer = last_node.layer + 1
                conn_info = cur_node.find_connected_nodes()
                if not conn_info.get('ranks'):
                    conn_info['ranks'] = self._rank_comm_nodes_dict.keys()
                last_node = cur_node
                if not self._find_connection(conn_info, cur_node, searched_ranks, seen_nodes):
                    logger.debug(f'Cannot find connected communication node for "{cur_node.node_id}".')

    def _find_connection(self, conn_info, cur_node, searched_ranks, seen_nodes):
        def connect(search_node):
            seen_nodes.add(search_node.node_id)
            if search_node.type == DiffAnalyseConst.DST:
                cur_node.add_dst(search_node)
            elif search_node.type == DiffAnalyseConst.SRC:
                search_node.layer = cur_node.layer
                search_node.add_dst(cur_node)
            else:
                cur_node.add_link(search_node)

        found = cur_node.connected
        for connected_rank in conn_info['ranks']:
            if connected_rank in searched_ranks:
                continue
            tar_id_prefix = f'{connected_rank}.{conn_info["api"]}'
            for search_id, search_node in self._rank_comm_nodes_dict[connected_rank].items():
                if search_id in seen_nodes:
                    continue
                if not (search_id.startswith(tar_id_prefix) and search_node.type == conn_info.get('type')):
                    continue
                search_conn_ranks = search_node.find_connected_nodes().get('ranks')
                if ((not search_conn_ranks and search_node.api not in DiffAnalyseConst.DIRECTED_API) or
                    cur_node.rank in search_conn_ranks):  # 有些无向通信算子没有填ProcessGroup，默认连接所有rank
                    connect(search_node)
                    found = True
                    break
        return found

    def _analyze_comm_nodes(self, rank):
        path = self._paths[rank]
        data = self._cache.load_json(path.dump_path)
        communication_nodes = {}
        if rank not in self._first_comm_nodes:  # 此rank没有通信节点
            return communication_nodes
        last_node_id = None  # 记录上一个通信节点的node_id
        compute_ops = []  # 记录两个通信节点之间的计算节点
        sub_layer = 0  # 记录两个通信算子之间异常计算节点的调用序数
        for op_name in dropwhile(lambda k: k != self._first_comm_nodes[rank], data):
            node_id = f'{rank}.{op_name}'
            op_data = data[op_name]
            if is_communication_op(op_name):
                comm_node = CommunicationNode(node_id, rank, DataNode(op_name, rank, op_data, sub_layer=sub_layer),
                                              compute_ops=compute_ops)
                if last_node_id:
                    communication_nodes.get(last_node_id).add_next(comm_node)
                communication_nodes[node_id] = comm_node
                last_node_id = node_id
                compute_ops = []
                sub_layer = 0
            elif not is_ignore_op(op_name):
                data_node = DataNode(op_name, rank, op_data, sub_layer=sub_layer)
                if data_node.is_diff:
                    compute_ops.append(data_node)
                sub_layer += 1
        if compute_ops:
            self._after_comm_diffs[rank] = compute_ops
        return communication_nodes

    def _pruning(self):
        deleted_node_id = []
        for nodes in self._rank_comm_nodes_dict.values():
            for node_id in list(nodes.keys()):
                node = nodes[node_id]
                if node.is_diff or node.compute_ops:
                    continue
                deleted_node_id.append(node_id)
                node.delete()
                del nodes[node_id]
        logger.debug(f'After pruning, following nodes are removed: [{", ".join(deleted_node_id)}]')

    def _search_first_diff(self):
        nodes_queues = []
        for comm_nodes in self._rank_comm_nodes_dict.values():
            nodes_queues.append(sorted(list(comm_nodes.values()), key=lambda x: x.layer))
        seen_nodes = set()

        def get_next_node(node_list):
            while node_list:
                next_node = node_list.pop(0)
                if next_node.node_id not in seen_nodes:
                    return next_node
            return None

        def find_all_members(ori_node):
            ids = get_relative_ids(ori_node)
            id_queue = list(chain(*[get_relative_ids(self._get_node_by_id(n_id)).difference(ids) for n_id in ids]))
            while id_queue:
                new_id = id_queue.pop(0)
                ids.add(new_id)
                id_queue.extend(get_relative_ids(self._get_node_by_id(new_id)).difference(ids))
            return ids

        def get_relative_ids(ori_node):
            if not ori_node:
                return set()
            return ({ori_node.node_id} | ori_node.link_nodes.keys() | ori_node.src_nodes.keys() |
                    ori_node.dst_nodes.keys())

        while any(nodes_queues):
            groups = []
            all_ids_in_groups = set()
            for nodes in nodes_queues:
                node = get_next_node(nodes)
                if not node:
                    continue
                if not groups or node.node_id not in all_ids_in_groups:
                    new_group = find_all_members(node)
                    groups.append(new_group)
                    all_ids_in_groups.update(new_group)
            for group in groups:
                seen_nodes.update(group)
                self._diff_nodes.extend(analyze_diff_in_group([self._get_node_by_id(n_id) for n_id in group]))
            if self._diff_nodes:
                # 找出所有layer和sub_layer最小的节点
                min_layer_sublayer = min((x.layer, x.sub_layer) for x in self._diff_nodes)
                self._diff_nodes = [
                                        node
                                        for node in self._diff_nodes
                                        if (node.layer, node.sub_layer) == min_layer_sublayer
                                   ]
                return

    def _get_node_by_id(self, node_id):
        splits = node_id.split(Const.SEP, 1)
        if len(splits) < 2 or not splits[0].isdigit():
            logger.error(f'invalid node_id {node_id}')
            raise RuntimeError(f'invalid node_id {node_id}')
        rank = int(splits[0])
        return self._rank_comm_nodes_dict.get(rank, {}).get(node_id)

    def _gen_analyze_info(self):
        if not os.path.exists(self._output_path):
            make_dir(self._output_path)
        file_name = f'diff_analyze_{time.time_ns()}.json'
        result_file = os.path.join(self._output_path, file_name)
        result_content = defaultdict(list)
        for node in self._diff_nodes:
            result_content[f'rank_{node.rank}'].append(node.gen_node_info(self._paths[node.rank]))
        save_json(result_file, result_content, 2)
        logger.info(f"The analyze result is saved in: {result_file}")
