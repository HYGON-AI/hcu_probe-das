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
from data.test_case_factory import TestCaseFactory
from server.app.utils.global_state import SINGLE
from server.app.controllers.layout_hierarchy_controller import LayoutHierarchyController


@pytest.mark.unit
class TestLayoutHierarchyController:
    
    @pytest.mark.parametrize("test_case",
        TestCaseFactory.get_change_expand_state_cases(), ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_change_expand_state(self, test_case):
        graph_type = test_case['input']['graph_type']
        if graph_type == SINGLE:
            test_case['input']['graph'] = TestCaseFactory.load_single_graph_test_data()
        else:
            test_case['input']['graph'] = TestCaseFactory.load_compare_graph_test_data().get(graph_type, {})
        node_name, graph_type, graph, micro_step = test_case['input'].values()
        excepted = test_case['expected']
        actual = LayoutHierarchyController.change_expand_state(node_name, graph_type, graph, micro_step)
        assert actual == excepted

    @pytest.mark.parametrize("test_case",
        TestCaseFactory.get_update_hierarchy_data_cases(), ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_update_hierarchy_data(self, test_case):
        graph_type = test_case['input']['graph_type']
        excepted = test_case['expected']
        actual = LayoutHierarchyController.update_hierarchy_data(graph_type)
        assert actual == excepted
  
