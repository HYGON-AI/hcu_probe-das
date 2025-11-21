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
  
