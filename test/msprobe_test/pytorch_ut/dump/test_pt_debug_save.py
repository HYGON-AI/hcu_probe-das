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

from unittest import TestCase
from unittest.mock import patch

import torch

from msprobe.pytorch import PrecisionDebugger
from msprobe.core.dump.common_config import CommonConfig, BaseConfig
from msprobe.core.dump.debugger.precision_debugger import BasePrecisionDebugger


class TestPytorchDebuggerSave(TestCase):
    def setUp(self):
        PrecisionDebugger._instance = None
        statistics_task_json = {
            "task": "statistics",
            "dump_path": "./dump_path",
            "rank": [],
            "step": [],
            "level": "debug",
            "enable_dataloader": False,
            "statistics": {
                "summary_mode": "statistics"
            }
        }
        common_config = CommonConfig(statistics_task_json)
        task_config = BaseConfig(statistics_task_json)
        with patch.object(BasePrecisionDebugger, "_parse_config_path", return_value=(common_config, task_config)):
            self.debugger = PrecisionDebugger()

    def test_forward_and_backward(self):
        def forward_func(x, y):
            PrecisionDebugger.save(x, "x_tensor")
            return x * y

        x = torch.tensor([1.])
        y = torch.tensor([2.])
        x.requires_grad = True
        y.requires_grad = True
        result_json = {
            "task": "statistics",
            "level": "debug",
            "framework": "pytorch",
            "dump_data_dir": None,
            "data": {
                "x_tensor.0.debug": {
                    "type": "torch.Tensor",
                    "dtype": "torch.float32",
                    "shape": torch.Size([1]),
                    "requires_grad": True
                },
                "x_tensor_grad.0.debug": {
                    "type": "torch.Tensor",
                    "dtype": "torch.float32",
                    "shape": torch.Size([1]),
                    "requires_grad": False
                }
            }
        }

        loss = forward_func(x, y)
        loss.backward()

        result = self.debugger.service.data_collector.data_writer.cache_debug
        # Remove 'tensor_stat_index' from all entries in the data dictionary
        for key in result["data"]:
            if 'tensor_stat_index' in result["data"][key]:
                del result["data"][key]['tensor_stat_index']

        self.assertEqual(result, result_json)
