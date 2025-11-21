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
from pathlib import Path
from types import SimpleNamespace

import pytest

from werkzeug.wrappers import Request
from werkzeug.test import EnvironBuilder
from data.test_case_factory import TestCaseFactory
from server.app.utils.global_state import GraphState
from server.app.views.graph_views import GraphView


@pytest.mark.integration
class TestGraphViews:
    
    captured = SimpleNamespace(status=None, headers=None)
    
    mock_vis_tag = 'mock_compare_resnet_data'

    @staticmethod
    def start_response(status, response_headers):
        TestGraphViews.captured.status = status
        TestGraphViews.captured.headers = dict(response_headers)
        return lambda x: None  # 必须返回一个 writer callable

    @staticmethod
    def create_mock_request(path="/meta"):
        builder = EnvironBuilder(path=path)
        return builder.get_environ()

    @pytest.mark.parametrize("test_case",
                            [
                                {"case_id": "1",
                                 "description": "测试index.html",
                                 "input": "/data/plugin/graph_ascend/index.html",
                                 "excepted": "200 OK"
                                },
                                {"case_id": "2",
                                 "description": "测试index.js",
                                 "input": "/data/plugin/graph_ascend/index.js",
                                 "excepted": "200 OK"
                                },
                                {"case_id": "3",
                                 "description": "测试404文件",
                                 "input": "/data/plugin/graph_ascend/index.css",
                                 "excepted": "404 NOT FOUND"
                                },
                                                                
                            ], ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_static_file_route(self, test_case):
        request = TestGraphViews.create_mock_request(test_case['input'])
        excepted = test_case['excepted']
        GraphView.static_file_route(request, TestGraphViews.start_response)
        assert TestGraphViews.captured.status == excepted

    @pytest.mark.parametrize("test_case",
                             [
                                {"case_id": "1",
                                 "description": "test_load_meta_dir",
                                 "excepted": {'data': {'st_test_cases': ['mock_compare_resnet_data']}, 'error': []}
                                }
                              ],
                             ids=lambda c: f"{c['case_id']}: {c['description']}")
    def test_load_meta_dir(self, test_case):
        logdir = Path(__file__).resolve().parent.parent.parent / 'data' / 'st_test_cases'
        GraphState.set_global_value('logdir', str(logdir))
        # 构造请求
        request = TestGraphViews.create_mock_request("/data/plugin/graph_ascend/load_meta_dir")
        response_iter = GraphView.load_meta_dir(request, TestGraphViews.start_response)
        excepted = test_case['excepted']
        # 获取响应内容
        response_body = json.loads(b''.join(response_iter).decode('utf-8'))
        assert response_body == excepted
        assert TestGraphViews.captured.status == "200 OK"
        assert TestGraphViews.captured.headers["Content-Type"] == "application/json"
        
    @pytest.mark.parametrize("test_case", [{"case_id": "2", "description": "test_load_graph_data"}],
                             ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_load_graph_data(self, test_case):
        request = TestGraphViews.create_mock_request(
            f"/data/plugin/graph_ascend/load_graph_data?run=st_test_cases&tag={TestGraphViews.mock_vis_tag}")
        response_iter = GraphView.load_graph_data(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter)
        runs = GraphState.get_global_value('runs')
        current_run = GraphState.get_global_value('current_run')
        current_tag = GraphState.get_global_value('current_tag')
        assert current_run == runs.get('st_test_cases')
        assert current_tag == TestGraphViews.mock_vis_tag
        assert TestGraphViews.captured.status == "200 OK"
        assert TestGraphViews.captured.headers["Content-Type"] == "text/event-stream; charset=utf-8"

    @pytest.mark.parametrize("test_case",
            TestCaseFactory.get_load_graph_config_info_cases(), ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_load_graph_config_info(self, test_case):
        request = TestGraphViews.create_mock_request(
            f"/data/plugin/graph_ascend/load_graph_config_info?run=st_test_cases&tag={TestGraphViews.mock_vis_tag}")
        response_iter = GraphView.load_graph_config_info(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        excepted = test_case['expected']
        assert response_body == json.dumps(excepted)

    @pytest.mark.parametrize("test_case",
            TestCaseFactory.get_load_graph_all_node_list_cases(), ids=lambda c: f"{c['case_id']}:{c['description']}")    
    def test_load_graph_all_node_list(self, test_case):
        request = TestGraphViews.create_mock_request(
            f"/data/plugin/graph_ascend/load_graph_all_node_list?run=st_test_cases&tag={TestGraphViews.mock_vis_tag}")
        response_iter = GraphView.load_graph_all_node_list(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        excepted = test_case['expected']
        assert response_body == json.dumps(excepted)
        
    @pytest.mark.parametrize("test_case",
            TestCaseFactory.get_change_node_expand_state_cases(), ids=lambda c: f"{c['case_id']}:{c['description']}")        
    def test_change_node_expand_state(self, test_case):
        excepted = test_case['expected']
        request = TestGraphViews.create_mock_request(test_case['input'])
        response_iter = GraphView.change_node_expand_state(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        assert response_body == json.dumps(excepted)

    @pytest.mark.parametrize("test_case",
            TestCaseFactory.get_test_add_match_nodes_cases(), ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_add_match_nodes(self, test_case):
        excepted = test_case['expected']
        request = TestGraphViews.create_mock_request(test_case['input'])
        response_iter = GraphView.add_match_nodes(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        assert response_body == json.dumps(excepted)

    @pytest.mark.parametrize("test_case",
        TestCaseFactory.get_test_update_hierarchy_data_cases(), ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_update_hierarchy_data(self, test_case):
        excepted = test_case['expected']
        request = TestGraphViews.create_mock_request(test_case['input'])
        response_iter = GraphView.update_hierarchy_data(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        assert response_body == json.dumps(excepted)

    @pytest.mark.parametrize("test_case", [
            {
                "case_id": "1",
                "description": "测试save_matched_relations接口",
                "expected": {"success": True, "data": "mock_compare_resnet_data.vis.config"}
            }
        ], ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_save_matched_relations(self, test_case):
        url = 'data/plugin/graph_ascend/saveMatchedRelations'
        params = 'metaData={"run":"st_test_cases","tag":"mock_compare_resnet_data"}'
        request_url = f"{url}?{params}"
        request = TestGraphViews.create_mock_request(request_url)
        response_iter = GraphView.save_matched_relations(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        excepted = test_case['expected']
        assert response_body == json.dumps(excepted)

    @pytest.mark.parametrize("test_case",
        TestCaseFactory.get_test_add_match_nodes_by_config_cases(), ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_add_match_nodes_by_config(self, test_case):
        excepted = test_case['expected']
        request = TestGraphViews.create_mock_request(test_case['input'])
        response_iter = GraphView.add_match_nodes_by_config(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        assert response_body == json.dumps(excepted)
        
    @pytest.mark.parametrize("test_case",
        TestCaseFactory.get_test_delete_match_nodes_cases(), ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_delete_match_nodes(self, test_case):
        excepted = test_case['expected']
        request = TestGraphViews.create_mock_request(test_case['input'])
        response_iter = GraphView.delete_match_nodes(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        assert response_body == json.dumps(excepted)

    @pytest.mark.parametrize("test_case",
        TestCaseFactory.get_test_update_colors_cases(), ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_update_colors(self, test_case):
        excepted = test_case['expected']
        request = TestGraphViews.create_mock_request(test_case['input'])
        response_iter = GraphView.update_colors(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        assert response_body == json.dumps(excepted)

    @pytest.mark.parametrize("test_case",
        TestCaseFactory.get_test_get_node_info_cases(), ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_get_node_info(self, test_case):
        excepted = test_case['expected']
        request = TestGraphViews.create_mock_request(test_case['input'])
        response_iter = GraphView.get_node_info(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        assert response_body == json.dumps(excepted)

    @pytest.mark.parametrize("test_case", [
        {
            "case_id": "1",
            "description": "测试save_data接口",
            "expected": {"success": True}
        }
        ], ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_save_data(self, test_case):
        excepted = test_case['expected']
        url = 'data/plugin/graph_ascend/saveData'
        params = 'metaData={"run":"st_test_cases","tag":"mock_compare_resnet_data"}'
        request_url = f"{url}?{params}"
        request = TestGraphViews.create_mock_request(request_url)
        response_iter = GraphView.save_data(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        assert response_body == json.dumps(excepted)

