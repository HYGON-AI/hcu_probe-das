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

import unittest
from unittest.mock import patch

from msprobe.core.common.const import Const
from msprobe.core.dump.data_dump.data_processor.factory import DataProcessorFactory
from msprobe.core.dump.data_dump.data_processor.mindspore_processor import (
    StatisticsDataProcessor as MindsporeStatisticsDataProcessor,
    TensorDataProcessor as MindsporeTensorDataProcessor,
)


class TestDataProcessorFactory(unittest.TestCase):
    def test_register_processors(self):
        with patch.object(DataProcessorFactory, "register_processor") as mock_register:
            DataProcessorFactory.register_processors(Const.MS_FRAMEWORK)
            self.assertEqual(mock_register.call_args_list[0][0],
                             (Const.MS_FRAMEWORK, Const.STATISTICS, MindsporeStatisticsDataProcessor))
            self.assertEqual(mock_register.call_args_list[1][0],
                             (Const.MS_FRAMEWORK, Const.TENSOR, MindsporeTensorDataProcessor))
