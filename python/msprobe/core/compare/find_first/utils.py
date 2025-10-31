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

from collections import OrderedDict
from dataclasses import dataclass
import sys
import time
import psutil

from msprobe.core.common.file_utils import check_file_or_directory_path, load_json
from msprobe.core.common.const import Const


@dataclass
class RankPath:
    rank: int
    dump_path: str

    def __init__(self, rank, dump_path):
        self.rank = rank
        check_file_or_directory_path(dump_path)
        self.dump_path = dump_path


class FileCache:
    """
    lazy load file
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self._max_memory_usage = psutil.virtual_memory().available / 4  # 最大占用当前可用内存空间的1/4
        self._cache = OrderedDict()
        self._access_cnt = {}
        self._access_time = {}
        self._size = {}

    @staticmethod
    def _sizeof(obj):
        seen = set()
        objs = [obj]
        size = 0
        while objs:
            obj = objs.pop()
            obj_id = id(obj)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            size += sys.getsizeof(obj)
            if isinstance(obj, dict):
                objs.extend(obj.keys())
                objs.extend(obj.values())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                objs.extend(obj)
        return size

    def load_json(self, json_path):
        if json_path in self._cache:
            self._access_cnt[json_path] += 1
            self._access_time[json_path] = time.monotonic()
            self._cache.move_to_end(json_path)
            return self._cache[json_path]
        self._cleanup()
        return self._load(json_path)

    def _load(self, json_path):
        data = load_json(json_path)
        self._add_to_cache(json_path, data)
        return data

    def _add_to_cache(self, key, data):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = data
            self._access_cnt[key] = 0
            self._access_time[key] = time.monotonic()
            self._size[key] = self._sizeof(data)

    def _calc_cache_size(self):
        return sys.getsizeof(self._cache) + sum(self._size.values())

    def _cleanup(self):
        while self._calc_cache_size() > self._max_memory_usage and self._cache:
            least_frequent_key = min(self._access_cnt.keys(), key=lambda k: self._access_cnt[k])
            least_recent_key = min(self._access_time.keys(), key=lambda k: self._access_time[k])
            largest_key = max(self._cache.keys(), key=lambda k: self._size[k])
            key_to_rm = min([least_frequent_key, least_recent_key, largest_key],
                            key=lambda k: (self._access_cnt[k], self._access_time[k], -self._size[k]))
            del self._cache[key_to_rm]
            del self._access_cnt[key_to_rm]
            del self._access_time[key_to_rm]
            del self._size[key_to_rm]


def is_communication_op(op_name):
    # 定义通信算子的关键字，覆盖各种通信操作，如all_reduce, send, broadcast等
    # 从wrap文件中读取，先硬编码在文件中
    return (op_name.startswith((Const.DISTRIBUTED, Const.MINT_DIST_API_TYPE_PREFIX, Const.MS_API_TYPE_COM)) and
            any(keyword in op_name for keyword in DiffAnalyseConst.COMMUNICATION_KEYWORDS))


def is_ignore_op(op_name):
    ignore_keywords = [
        'Torch.empty',
        'Torch.fill',
        'Tensor.__setitem__'
    ]
    return any(keyword in op_name for keyword in ignore_keywords)


class DiffAnalyseConst:
    COMMUNICATION_KEYWORDS = {
        'send',  # send 算子
        'recv',  # recv 算子
        'broadcast',  # broadcast 算子
        'all_reduce',  # all_reduce 算子
        'reduce',  # reduce 算子
        'all_gather',  # all_gather 算子
        'gather',  # gather 算子
        'isend',  # isend 算子
        'irecv',  # irecv 算子
        'scatter',  # scatter 算子
        'reduce_scatter',  # reduce_scatter 算子
        '_reduce_scatter_base',  # _reduce_scatter_base 算子
        '_all_gather_base',  # _all_gather_base 算子
        'all_to_all_single',  # all_to_all_single 算子
        'all_to_all',  # all_to_all 算子
        'all_gather_into_tensor',  # all_gather_into_tensor 算子
        'reduce_scatter_tensor',  # reduce_scatter_tensor 算子
        'send_object_list',  # send_object_list 算子
        'recv_object_list'  # recv_object_list 算子
    }
    P2P_API_MAPPING = {'send': 'recv', 'recv': 'send', 'isend': 'irecv', 'irecv': 'isend',
                       'send_object_list': 'recv_object_list', 'recv_object_list': 'send_object_list'}
    SRC = 'src'
    DST = 'dst'
    SRC_GROUP = 'group_src'
    DST_GROUP = 'group_dst'
    LINK = 'link'
    DIRECTED_API = {'send': DST, 'recv': SRC, 'isend': DST, 'irecv': SRC, 'broadcast': SRC, 'scatter': SRC,
                    'gather': DST, 'send_object_list': DST, 'recv_object_list': SRC}
    OPPOSITE_DIR = {SRC: DST, DST: SRC}
    DUMP_FILE = "dump.json"
    CONSTRUCT_FILE = "construct.json"
    STACK_FILE = "stack.json"


def analyze_diff_in_group(nodes_group):
    diff_nodes = []

    def get_compute_ops_from_comm_nodes(comm_nodes):
        for comm_node in comm_nodes:
            for op_node in comm_node.compute_ops:
                op_node.layer = comm_node.layer
                diff_nodes.append(op_node)

    def get_comm_ops(comm_nodes):
        for node in comm_nodes:
            node.data.layer = node.layer
            diff_nodes.append(node.data)

    # 先看src或link中input是否有异常
    src_list = list(filter(lambda node: node.type in [DiffAnalyseConst.SRC, DiffAnalyseConst.LINK], nodes_group))
    input_diff_nodes = list(filter(lambda node: node.is_diff, src_list))
    # 如果有异常回溯计算节点找到异常来源
    # 使用cpu模拟节点进行计算，查看结果是否有问题。需要对所有计算节点录入/映射，暂不实现。
    get_compute_ops_from_comm_nodes(nodes_group)
    # 筛选入参没问题但出参有问题的通信节点
    output_diff_nodes = list(filter(lambda node: node.data.is_diff, nodes_group))
    get_comm_ops(output_diff_nodes)
    return diff_nodes