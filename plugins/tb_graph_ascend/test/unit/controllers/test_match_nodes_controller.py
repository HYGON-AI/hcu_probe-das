# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pytest

from server.app.controllers.match_nodes_controller import MatchNodesController
from server.app.utils.global_state import GraphState
from data.test_case_factory import TestCaseFactory


@pytest.mark.unit
class TestMatchNodesController:
    """测试匹配节点功能"""

    @pytest.mark.parametrize("test_case", TestCaseFactory.get_process_task_add_cases(),
                             ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_process_task_add(self, test_case):
        """测试添加节点功能"""
        graph_data, npu_node_name, bench_node_name, task = test_case['input'].values()
        expected = test_case['expected']
        actual = MatchNodesController.process_task_add(graph_data, npu_node_name, bench_node_name, task)
        assert actual == expected
        
    @pytest.mark.parametrize("test_case", TestCaseFactory.get_process_task_delete_cases(),
                             ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_process_task_delete(self, test_case):
        """测试删除节点功能"""
        if test_case.get('config', None):
            GraphState.set_global_value("config_data", test_case['config'])
        graph_data, npu_node_name, bench_node_name, task = test_case['input'].values()
        expected = test_case['expected']
        actual = MatchNodesController.process_task_delete(graph_data, npu_node_name, bench_node_name, task)
        assert actual == expected

    @pytest.mark.parametrize("test_case", TestCaseFactory.get_process_task_add_child_layer_cases(),
                             ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_process_task_add_child_layer(self, test_case):
        """测试添加子节点层功能"""
        graph_data, npu_node_name, bench_node_name, task = test_case['input'].values()
        excepted = test_case['expected']
        actual = MatchNodesController.process_task_add_child_layer(graph_data, npu_node_name, bench_node_name, task)
        assert actual == excepted

    @pytest.mark.parametrize("test_case", TestCaseFactory.get_process_task_delete_child_layer_cases(),
                             ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_process_task_delete_child_layer(self, test_case):
        """测试删除子节点层功能"""
        if test_case.get('config', None):
            GraphState.set_global_value("config_data", test_case['config'])
        graph_data, npu_node_name, bench_node_name, task = test_case['input'].values()
        excepted = test_case['expected']
        actual = MatchNodesController.process_task_delete_child_layer(graph_data, npu_node_name, bench_node_name, task)
        assert actual == excepted
    
    @pytest.mark.parametrize("test_case", TestCaseFactory.get_process_task_add_child_layer_by_config_cases(),
                             ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_process_task_add_child_layer_by_config(self, test_case):
        """测试根据配置文件添加子节点层功能"""
        graph_data, match_node_links, task = test_case['input'].values()
        excepted = test_case['expected']
        actual = MatchNodesController.process_task_add_child_layer_by_config(graph_data, match_node_links, task)
        assert actual == excepted
   
