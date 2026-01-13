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
