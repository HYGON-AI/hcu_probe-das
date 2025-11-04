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
import re
from msprobe.visualization.builder.msprobe_adapter import op_patterns
from msprobe.core.common.log import logger


class NodeOp(Enum):
    module = 0
    function_api = 1
    api_collection = 9

    @staticmethod
    def get_node_op(node_name: str):
        """
        基于代表节点的字符串，解析节点种类
        """
        for op in NodeOp:
            index = op.value
            if index < 0 or index >= len(op_patterns):
                continue
            pattern = op_patterns[index]
            if re.match(pattern, node_name):
                return op
        logger.warning(f"Cannot parse node_name {node_name} into NodeOp, default parsing as module.")
        return NodeOp.module
