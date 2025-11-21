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
from server.app.utils.graph_utils import GraphUtils


class TestGraphUtils:

    @pytest.mark.parametrize("test_case",
                             [
                                {
                                    "case_id": "1",
                                    "description": "正常的多层节点 A -> B -> C",
                                    "input": {
                                        "graph_data": {
                                            "node": {
                                                "C": {"upnode": "B"},
                                                "B": {"upnode": "A"},
                                                "A": {"upnode": None}
                                            }
                                        },
                                        "node_name": "C"
                                    },
                                    "expected": ["A", "B", "C"]
                                },
                                {
                                    "case_id": "2",
                                    "description": "单一节点无上级",
                                    "input": {
                                        "graph_data": {
                                            "node": {
                                                "A": {"upnode": None}
                                            }
                                        },
                                        "node_name": "A"
                                    },
                                    "expected": ["A"]
                                },
                                {
                                    "case_id": "3",
                                    "description": "节点不存在于图中",
                                    "input": {
                                        "graph_data": {
                                            "node": {
                                                "A": {"upnode": None}
                                            }
                                        },
                                        "node_name": "B"
                                    },
                                    "expected": ["B"]
                                },
                                {
                                    "case_id": "4",
                                    "description": "图为空",
                                    "input": {
                                        "graph_data": {},
                                        "node_name": "A"
                                    },
                                    "expected": []
                                },
                                {
                                    "case_id": "5",
                                    "description": "节点名为空",
                                    "input": {
                                        "graph_data": {
                                            "node": {
                                                "A": {"upnode": None}
                                            }
                                        },
                                        "node_name": ""
                                    },
                                    "expected": []
                                }
                            ],
                             ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_get_parent_node_list(self, test_case):
        graph_data, node_name = test_case['input'].values()
        expected = test_case['expected']
        actual = GraphUtils.get_parent_node_list(graph_data, node_name)
        assert actual == expected

    @pytest.mark.parametrize("test_case",
                            [
                                {
                                    "case_id": "1",
                                    "description": "数字大小比较，10 大于 2",
                                    "input": {"a": "file_10", "b": "file_2"},
                                    "expected": 1
                                },
                                {
                                    "case_id": "2",
                                    "description": "相同前缀，数字部分较小",
                                    "input": {"a": "item_3_part", "b": "item_12_part"},
                                    "expected": "-1"
                                },
                                {
                                    "case_id": "3",
                                    "description": "路径比较，a/b/c 小于 a/b/d",
                                    "input": {"a": "a/b/c", "b": "a/b/d"},
                                    "expected": "-1"
                                },
                                {
                                    "case_id": "4",
                                    "description": "混合路径和下划线分隔，等价内容",
                                    "input": {"a": "a_b_1", "b": "a/b/1"},
                                    "expected": 0
                                },
                                {
                                    "case_id": "5",
                                    "description": "子路径多一级，a/b 小于 a/b/c",
                                    "input": {"a": "a/b", "b": "a/b/c"},
                                    "expected": "-1"
                                },
                                {
                                    "case_id": "6",
                                    "description": "数字 vs 字母，数字优先",
                                    "input": {"a": "file_1", "b": "file_a"},
                                    "expected": "-1"
                                },
                                {
                                    "case_id": "7",
                                    "description": "完全相同",
                                    "input": {"a": "dir/subdir_10/file_5", "b": "dir/subdir_10/file_5"},
                                    "expected": 0
                                },
                                {
                                    "case_id": "8",
                                    "description": "字母 vs 数字，字母在后",
                                    "input": {"a": "a2b", "b": "a10"},
                                    "expected": "-1"
                                }
                            ],
                            ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_compare_tag_names(self, test_case):

        def normalize(val: int) -> int:
            return 0 if val == 0 else (1 if val > 0 else -1)

        a, b = test_case['input'].values()
        expected = int(test_case['expected'])
        actual = GraphUtils.compare_tag_names(a, b)
        assert normalize(actual) == expected

    @pytest.mark.parametrize("test_case",
                            [
                                {
                                    "case_id": "1",
                                    "description": "输入为 0 字节",
                                    "input": {"size_bytes": 0},
                                    "expected": "0 B"
                                },
                                {
                                    "case_id": "2",
                                    "description": "输入为字节（小于 1KB）",
                                    "input": {"size_bytes": 512},
                                    "expected": "512.00 B"
                                },
                                {
                                    "case_id": "3",
                                    "description": "输入为 1KB",
                                    "input": {"size_bytes": 1024},
                                    "expected": "1.00 KB"
                                },
                                {
                                    "case_id": "4",
                                    "description": "输入为 1MB",
                                    "input": {"size_bytes": 1024 * 1024},
                                    "expected": "1.00 MB"
                                },
                                {
                                    "case_id": "5",
                                    "description": "输入为 1.5MB",
                                    "input": {"size_bytes": 1.5 * 1024 * 1024},
                                    "expected": "1.50 MB"
                                },
                                {
                                    "case_id": "6",
                                    "description": "输入为 1GB，保留 3 位小数",
                                    "input": {"size_bytes": 1024 ** 3, "decimal_places": 3},
                                    "expected": "1.000 GB"
                                },
                                {
                                    "case_id": "7",
                                    "description": "输入为 2TB",
                                    "input": {"size_bytes": 2 * 1024 ** 4},
                                    "expected": "2.00 TB"
                                },
                                {
                                    "case_id": "8",
                                    "description": "输入为浮点字节数",
                                    "input": {"size_bytes": 12345678.9},
                                    "expected": "11.77 MB"
                                },
                                {
                                    "case_id": "9",
                                    "description": "输入为 PB 范围",
                                    "input": {"size_bytes": 1.2 * 1024 ** 5},
                                    "expected": "1.20 PB"
                                }
                            ],
                            ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_bytes_to_human_readable(self, test_case):
        size_bytes = test_case["input"]["size_bytes"]
        decimal_places = test_case["input"].get("decimal_places", 2)
        expected = test_case['expected']
        actual = GraphUtils.bytes_to_human_readable(size_bytes, decimal_places)
        assert actual == expected
        
