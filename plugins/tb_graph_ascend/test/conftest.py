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
