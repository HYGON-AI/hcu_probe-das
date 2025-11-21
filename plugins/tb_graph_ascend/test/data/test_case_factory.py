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
import json
import os


class TestCaseFactory:
    """管理所有测试用例的统一工厂"""
    
    UT_CASE_DIR = os.path.join(os.path.dirname(__file__), 'ut_test_cases')
    ST_CASE_DIR = os.path.join(os.path.dirname(__file__), 'st_test_cases')

    @classmethod
    def get_process_task_add_cases(cls):
        return cls.load_ut_cases('test_match_node_controller/process_task_add_case.json')
    
    @classmethod
    def get_process_task_delete_cases(cls):
        return cls.load_ut_cases('test_match_node_controller/process_task_delete_case.json')
    
    @classmethod
    def get_process_task_add_child_layer_cases(cls):
        return cls.load_ut_cases('test_match_node_controller/process_task_add_child_layer.json')
    
    @classmethod
    def get_process_task_delete_child_layer_cases(cls):
        return cls.load_ut_cases('test_match_node_controller/process_task_delete_child_layer.json')
    
    @classmethod
    def get_process_task_add_child_layer_by_config_cases(cls):
        return cls.load_ut_cases('test_match_node_controller/process_task_add_child_layer_by_config.json')

    @classmethod
    def get_change_expand_state_cases(cls):
        return cls.load_ut_cases('test_layout_hierarchy_controller/change_expand_state_case.json')
    
    @classmethod
    def get_update_hierarchy_data_cases(cls):
        return cls.load_ut_cases('test_layout_hierarchy_controller/update_hierarchy_data_case.json')
    
    @classmethod
    def load_single_graph_test_data(cls):
        return cls.load_ut_cases('test_layout_hierarchy_controller/mock_single_statis_graph.vis')
    
    @classmethod
    def load_compare_graph_test_data(cls):
        return cls.load_ut_cases('test_layout_hierarchy_controller/mock_compare_statis_graph.vis')

    @classmethod
    def load_ut_cases(cls, file_name):
        """从JSON文件加载测试用例"""
        path = os.path.join(cls.UT_CASE_DIR, file_name)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ST
    @classmethod
    def get_load_graph_config_info_cases(cls):
        return cls.load_st_cases('test_load_graph_config_info.json')
    
    @classmethod
    def get_load_graph_all_node_list_cases(cls):
        return cls.load_st_cases('test_load_graph_all_node_list.json')
    
    @classmethod
    def get_change_node_expand_state_cases(cls):
        return cls.load_st_cases('test_change_node_expand_state.json')
    
    @classmethod
    def get_test_add_match_nodes_cases(cls):
        return cls.load_st_cases('test_add_match_nodes.json')
    
    @classmethod
    def get_test_update_hierarchy_data_cases(cls):
        return cls.load_st_cases('test_update_hierarchy_data.json')
    
    @classmethod
    def get_test_delete_match_nodes_cases(cls):
        return cls.load_st_cases('test_delete_match_nodes.json')
    
    @classmethod
    def get_test_get_node_info_cases(cls):
        return cls.load_st_cases('test_get_node_info.json')
    
    @classmethod
    def get_test_add_match_nodes_by_config_cases(cls):
        return cls.load_st_cases('test_add_match_nodes_by_config.json')

    @classmethod
    def get_test_update_colors_cases(cls):
        return cls.load_st_cases('test_update_colors.json')

    @classmethod
    def load_compare_resnet_test_data(cls):
        return cls.load_st_cases('mock_compare_resnet_data.vis')

    @classmethod
    def load_st_cases(cls, file_name):
        """从JSON文件加载测试用例"""
        path = os.path.join(cls.ST_CASE_DIR, file_name)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
