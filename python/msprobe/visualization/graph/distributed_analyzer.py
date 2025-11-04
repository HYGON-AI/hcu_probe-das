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
from enum import Enum
from msprobe.visualization.utils import GraphConst
from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.log import logger


class CommunicationType(Enum):
    """
    通信类型：发送、接收、发送接收
    """
    SEND = 'send'
    RECEIVE = 'receive'
    SEND_RECEIVE = 'send_receive'


class DistributedType(Enum):
    """
    分布式类型：点对点通信、集体通信
    """
    P2P = 'p2p'
    COLLECTIVE = 'collective'


CANNOT_MATCH = 'cannot match distributed node in rank'


class DistributedAnalyzer:

    def __init__(self, graphs: dict, overflow_check: bool):
        self.graphs = graphs
        self.overflow_check = overflow_check
        self.config = {
            # 当前通信api名称: 匹配目标通信api名称, 获取rank信息的位置参数或关键字参数, 通信类型, 分布式类型
            'send': ['recv', GraphConst.DST, CommunicationType.SEND.value, DistributedType.P2P],
            'isend': ['irecv', GraphConst.DST, CommunicationType.SEND.value, DistributedType.P2P],
            'recv': ['send', GraphConst.SRC, CommunicationType.RECEIVE.value, DistributedType.P2P],
            'irecv': ['isend', GraphConst.SRC, CommunicationType.RECEIVE.value, DistributedType.P2P],
            'broadcast': ['broadcast', '1', CommunicationType.SEND.value, DistributedType.COLLECTIVE],
            'scatter': ['scatter', GraphConst.SRC, CommunicationType.SEND.value, DistributedType.COLLECTIVE],
            'gather': ['gather', GraphConst.DST, CommunicationType.RECEIVE.value, DistributedType.COLLECTIVE],
            'reduce': ['reduce', '1', CommunicationType.RECEIVE.value, DistributedType.COLLECTIVE]
        }
        self.group_node_mapping = {}
        self._make_group_node_mapping()

    @staticmethod
    def _get_opposite_communication_type(action):
        if action == CommunicationType.SEND.value:
            return CommunicationType.RECEIVE.value
        elif action == CommunicationType.RECEIVE.value:
            return CommunicationType.SEND.value
        return action

    @staticmethod
    def _node_output_all_equal(data: dict, target_data: dict):
        keys_to_compare = [Const.DTYPE, Const.SHAPE, Const.MAX, Const.MIN, Const.MEAN, Const.NORM]
        return all(data.get(key) == target_data.get(key) for key in keys_to_compare)

    @staticmethod
    def _get_target_rank(node, rank, parameter):
        """
        点对点通信, 从输出数据参数src或dst, 获取通信目标rank
        一对多通信和多对一通信, 从输出数据参数src或dst或位置参数, 获取发送或接收的rank源头
        :param node: 当前节点
        :param rank: 当前rank
        :param parameter: 输出数据参数
        :return: 目标rank
        """
        target_rank = node.input_data.get(f'{node.id}{GraphConst.INPUT}{parameter}', {}).get('value')
        if target_rank is None:
            logger.debug(f'The parameter {parameter} of node {node.id} does not exist, {CANNOT_MATCH}{rank}')
        return target_rank

    @staticmethod
    def _get_group_info(node, rank):
        """
        获取当前通信节点的group参数中的group_ranks和group_id
        :param node: 当前通信节点
        :param rank: 当前rank
        :return: group_ranks和group_id
        """
        group = node.input_data.get(f'{node.id}{GraphConst.INPUT}group', {})
        if not group:
            logger.debug(f'The kwarg group of node {node.id} does not exist, {CANNOT_MATCH}{rank}')
            return None, None
        group_ranks = group.get('group_ranks')
        if not group_ranks:
            logger.debug(f'The group_ranks of node {node.id} does not exist, {CANNOT_MATCH}{rank}')
            return None, None
        group_id = group.get('group_id')
        if not group_id:
            logger.debug(f'The group_id of node {node.id} does not exist, {CANNOT_MATCH}{rank}')
            return None, None
        return group_ranks, group_id

    def distributed_match(self):
        for rank, graph in self.graphs.items():
            nodes = graph.node_map
            for node_id, node in nodes.items():
                # 不是通信节点或者已经匹配过了
                if not node_id.startswith(Const.DISTRIBUTED) or node.matched_distributed:
                    continue
                api_name, distributed_type = self._get_distributed_name_and_type(node_id)
                if api_name == GraphConst.BATCH_P2P:
                    self._batch_p2p_match(node, rank)
                elif distributed_type == DistributedType.P2P:
                    self._p2p_match(node, rank, api_name)
                else:
                    self._collective_match(node, rank, api_name)

    def _make_group_node_mapping(self):
        """
        建立通信节点的全局唯一标识映射
        key: rank号, value: unique_group_id与node_id之间的映射
        {
            "0": {
                "unique_group_id1": "node_id1",
                "unique_group_id2": "node_id2",
                "node_id1": "unique_group_id1",
                "node_id2": "unique_group_id2"
            },
            "1": {},
            "2": {}
        }
        """
        for rank, graph in self.graphs.items():
            group_count = {}
            group_info = {}
            batch_p2p_count = {}
            nodes = graph.node_map
            for node_id, node in nodes.items():
                if not node_id.startswith(Const.DISTRIBUTED):
                    continue
                api_name, distributed_type = self._get_distributed_name_and_type(node_id)
                if api_name == GraphConst.BATCH_P2P:
                    self._make_batch_p2p_mapping(node, rank, batch_p2p_count)
                    continue
                elif distributed_type == DistributedType.P2P:
                    config_info = self.config.get(api_name)
                    target_rank = self._get_target_rank(node, rank, config_info[1])
                    if target_rank is None:
                        continue
                    # p2p通信节点，api名称+传输目标rank作为group_id
                    group_id = api_name + Const.RANK + str(target_rank)
                else:
                    # 其他通信节点直接获取group_id, 并拼接api名称
                    _, group_id = self._get_group_info(node, rank)
                    if not group_id:
                        continue
                    group_id += api_name
                # 同group_id的调用次数累计
                group_count[group_id] = group_count.get(group_id, 0) + 1
                # group_id+同group_id的调用次数作为唯一的unique_group_id
                unique_group_id = group_id + Const.REPLACEMENT_CHARACTER + str(group_count.get(group_id))
                group_info[unique_group_id] = node_id
                group_info[node_id] = unique_group_id
            if rank not in self.group_node_mapping:
                self.group_node_mapping[rank] = {}
            self.group_node_mapping[rank].update(group_info)

    def _make_batch_p2p_mapping(self, node, rank, batch_p2p_count):
        """
        给batch_isend_irecv接口的每个p2p内容赋予唯一标识
        """
        if rank not in self.group_node_mapping:
            self.group_node_mapping[rank] = {}
        params = []
        for info_dict in node.batch_p2p_info:
            op = info_dict.get(GraphConst.OP)
            target_rank = info_dict.get(GraphConst.PEER)
            if op is None or target_rank is None:
                logger.debug('Cannot get param op or peer.')
                continue
            group_id = op + Const.REPLACEMENT_CHARACTER + Const.RANK + str(target_rank) + \
                       Const.REPLACEMENT_CHARACTER + info_dict.get(GraphConst.GROUP_ID, '')
            batch_p2p_count[group_id] = batch_p2p_count.get(group_id, 0) + 1
            # 例如: isend_rank0_5a4d31ad765260ba50eb190f1f9fd163_1
            unique_group_id = group_id + Const.REPLACEMENT_CHARACTER + str(batch_p2p_count.get(group_id))
            params.append(unique_group_id)
            self.group_node_mapping.get(rank)[unique_group_id] = node.id
        if params:
            self.group_node_mapping.get(rank)[node.id] = params

    def _get_distributed_name_and_type(self, node_id):
        if Const.SEP not in node_id:
            raise ValueError(f'Invalid node id {node_id}.')
        api_name = node_id.split(Const.SEP)[1]
        if api_name in self.config:
            return api_name, self.config.get(api_name)[3]
        return api_name, DistributedType.COLLECTIVE

    def _get_target_node(self, rank, unique_group_id, api_name, target_rank, target_api_name=None):
        """
        获取名称匹配上的目标节点
        :param rank: 当前rank
        :param unique_group_id: 当前节点唯一group id
        :param api_name: 当前节点的api名称, 例如Distributed.isend.0.forward, api名称为isend
        :param target_rank: 与当前节点产生通信的rank
        :param target_api_name: 与当前节点产生通信的节点api名称, 仅p2p通信需要配置
        :return: 目标节点
        """
        target_graph = self.graphs.get(target_rank)
        if not target_graph:
            logger.debug(f'Graph data does not exist, {CANNOT_MATCH}{target_rank}')
            return None
        target_group_mapping = self.group_node_mapping.get(target_rank)
        # p2p通信，想要获取目标节点，需要替换unique_group_id中的rank和api name,
        # 例如isend发送到rank1，对应的irecv接收自rank0, isend_rank1与irecv_rank0对应
        target_unique_group_id = (unique_group_id
                                  .replace(Const.RANK + str(target_rank), Const.RANK + str(rank))
                                  .replace(api_name, target_api_name)) if target_api_name else unique_group_id
        target_node_id = target_group_mapping.get(target_unique_group_id, '')
        target_node = target_graph.node_map.get(target_node_id)
        if not target_node:
            logger.debug(f'Node {target_node_id} does not exist, {CANNOT_MATCH}{target_rank}')
            return None
        return target_node

    def _add_node_matched_distributed(self, node, target_node, api_name, target_rank, reversal_type=False):
        """
        给当前节点添加matched_distributed字段信息
        :param node: 当前节点
        :param target_node: 匹配上的目标节点
        :param api_name: 当前节点的api名称
        :param target_rank: 匹配上的目标rank
        :param reversal_type: 是否需要反转通信类型，例如broadcast在rank0通信类型是发送，但在其他rank通信类型是接收
        """
        communications_type = self.config.get(api_name)[2]
        communications_type = self._get_opposite_communication_type(communications_type) if reversal_type \
            else communications_type
        index = target_node.data.get(GraphConst.OVERFLOW_LEVEL, CompareConst.NAN) if self.overflow_check \
            else target_node.data.get(GraphConst.JSON_INDEX_KEY, CompareConst.NAN)
        matched_distributed = {
            'communications_type': communications_type,
            'nodes_info': {target_rank: [str(index), target_node.id]}
        }
        node.matched_distributed = matched_distributed

    def _p2p_match(self, node, rank, api_name):
        """
        点对点通信匹配

        根据当前点对点通信节点的输出数据中的src或dst参数, 确定目标rank, 并从目标rank中找到对应的点对点通信节点, 校验输出数据是否一致，
        校验通过则在两个匹配节点增加匹配信息
        Args:
            node: 当前点对点通信节点
            rank: 当前节点所属rank
            api_name: 当前节点的api名称
        Returns:
        """
        config_info = self.config.get(api_name)
        target_api_name = config_info[0]
        #
        target_rank = self._get_target_rank(node, rank, config_info[1])
        if target_rank is None:
            return
        unique_group_id = self.group_node_mapping.get(rank, {}).get(node.id, '')
        target_node = self._get_target_node(rank, unique_group_id, api_name, target_rank, target_api_name)
        if not target_node:
            return
        target_config_info = self.config.get(target_api_name)
        source_rank = (target_node.input_data.get(f'{target_node.id}{GraphConst.INPUT}{target_config_info[1]}', {})
                       .get('value'))
        if source_rank is None:
            logger.debug(
                f'The kwarg {target_config_info[1]} of node {target_node.id} does not exist, '
                f'{CANNOT_MATCH}{source_rank}')
            return
        if source_rank != rank:
            # 点对点通信，待匹配目标节点包含的rank信息要与当前rank一致
            logger.debug(
                f'{node.id} of rank{rank} is expected to communicate with {target_node.id} of rank{target_rank}, '
                f'but the data shows that {target_node.id} communicates with rank{source_rank}.'
                f'The rank is inconsistent, cannot match distributed node')
            return

        # 点对点通信，两个匹配节点的输出数据要一致
        if not DistributedAnalyzer._node_output_all_equal(node.output_data.get(node.id + '.output.0'),
                                                          target_node.output_data.get(target_node.id + '.output.0')):
            logger.debug(f'{node.id} output of rank{rank} is different from the {target_node.id} '
                           f'output of rank{target_rank}, cannot match distributed node')
            return

        self._add_node_matched_distributed(node, target_node, api_name, target_rank)
        self._add_node_matched_distributed(target_node, node, target_api_name, rank)

    def _collective_match(self, node, rank, api_name):
        """
        集体通信匹配

        一对多通信和多对一通信, 需要先获取节点输出数据中的src或dst或位置参数, 确定发送源或接收源, 多对多通信不需要
        :param node: 当前集体通信节点
        :param rank: 当前节点所属rank
        :param api_name: 当前节点的api名称
        :return:
        """
        communications_type = CommunicationType.SEND_RECEIVE.value
        config_info = self.config.get(api_name)
        if config_info:
            # 此时为一对多通信或多对一通信
            source_rank = self._get_target_rank(node, rank, config_info[1])
            if source_rank is None or str(source_rank) != str(rank):
                return
            communications_type = config_info[2]
        group_ranks, group_id = self._get_group_info(node, rank)
        if not group_ranks or not group_id:
            return
        unique_group_id = self.group_node_mapping.get(rank, {}).get(node.id, '')
        matched_distributed = {'communications_type': communications_type}
        nodes_info = {}
        for target_rank in group_ranks:
            if str(target_rank) == str(rank):
                continue
            target_node = self._get_target_node(rank, unique_group_id, api_name, target_rank)
            if not target_node:
                continue
            _, target_group_id = self._get_group_info(target_node, target_rank)
            if not target_group_id:
                continue
            if group_id != target_group_id:
                logger.debug(
                    f'{node.id} of rank{rank} is expected to communicate with {target_node.id} of rank{target_rank}'
                    f', but the data shows that the group id of the two nodes are different, '
                    f'cannot match distributed node')
                continue
            # 给当前通信节点添加matched_distributed字段信息
            index = target_node.data.get(GraphConst.OVERFLOW_LEVEL, CompareConst.NAN) if self.overflow_check \
                else target_node.data.get(GraphConst.JSON_INDEX_KEY, CompareConst.NAN)
            nodes_info[target_rank] = [str(index), target_node.id]
            if config_info:
                # 给匹配上的目标节点也添加matched_distributed字段信息
                self._add_node_matched_distributed(target_node, node, api_name, rank, True)
        if nodes_info:
            matched_distributed['nodes_info'] = nodes_info
            node.matched_distributed = matched_distributed

    def _batch_p2p_match(self, node, rank):
        """
        批量点对点匹配

        针对torch.distributed.batch_isend_irecv接口，其入参是一个包含点对点通信信息的集合，需要遍历集合对每个点对点通信信息进行匹配
        :param node: 当前集体通信节点
        :param rank: 当前节点所属rank
        :return:
        """
        unique_group_ids = self.group_node_mapping.get(rank, {}).get(node.id)
        if not unique_group_ids:
            return
        matched_distributed = [] if len(unique_group_ids) > 1 else {}
        for unique_group_id in unique_group_ids:
            try:
                id_info = unique_group_id.split(Const.REPLACEMENT_CHARACTER)
                api_name = id_info[0]
                target_api_name = self.config.get(api_name)[0]
                target_rank = int(id_info[1].replace(Const.RANK, ''))
            except Exception as e:
                logger.debug(f'Failed to parse batch p2p parameter with error info: {e}.')
                continue
            target_node = self._get_target_node(rank, unique_group_id, api_name, target_rank, target_api_name)
            if not target_node:
                continue
            communications_type = self.config.get(api_name)[2]
            index = target_node.data.get(GraphConst.OVERFLOW_LEVEL, CompareConst.NAN) if self.overflow_check \
                else target_node.data.get(GraphConst.JSON_INDEX_KEY, CompareConst.NAN)
            matched_info = {
                'communications_type': communications_type,
                'nodes_info': {target_rank: [str(index), target_node.id]}
            }
            matched_distributed.append(matched_info) if isinstance(matched_distributed, list) \
                else matched_distributed.update(matched_info)
        if matched_distributed:
            node.matched_distributed = matched_distributed
