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
from server.app.utils.global_state import GraphState
from data.test_case_factory import TestCaseFactory


@pytest.fixture(scope="function", autouse=True)
def reset_global_state(request):
    """每个测试后重置全局状态"""
    # 执行测试
    yield
    # 恢复原始状态
    if request.module.__name__ != "test_graph_views":
        GraphState.init_defaults()


def pytest_addoption(parser):
    """添加自定义命令行选项"""
    parser.addoption("--runslow", action="store_true", default=False,
                    help="Run slow tests")
    parser.addoption("--dataset", action="store", default="small",
                    help="Test dataset size: small|medium|large")


@pytest.fixture
def test_case_factory():
    """提供测试用例工厂实例"""
    return TestCaseFactory
