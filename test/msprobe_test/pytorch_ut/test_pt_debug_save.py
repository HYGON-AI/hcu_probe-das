# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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