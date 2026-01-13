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
   
