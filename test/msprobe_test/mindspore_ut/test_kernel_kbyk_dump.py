# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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


import os

from unittest import TestCase
from unittest.mock import patch

from msprobe.core.dump.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.ms_config import StatisticsConfig
from msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump import KernelKbykDump

from collections import Counter

import mindspore as ms
ms_version = ms.__version__


class TestKernelKbykDump(TestCase):
    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        self.assertIn("kernel_kbyk_dump.json", mock_save_json.call_args_list[0][0][0])
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")

        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)
        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_when_async_dump_then_pass(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2",
            "async_dump": True
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["enable"], False)

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        self.assertIn("kernel_kbyk_dump.json", mock_save_json.call_args_list[0][0][0])
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")

        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)
        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_when_device_then_pass(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2",
            "statistics": {
                "list": [],
                "data_mode": ["all"],
                "device": "device",
                "summary_mode": "statistics"
            }
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config["statistics"])
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["stat_calc_mode"], "device")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        self.assertIn("kernel_kbyk_dump.json", mock_save_json.call_args_list[0][0][0])
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")

        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)
        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_when_precision_then_pass(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2",
            "statistics": {
                "list": [],
                "data_mode": ["all"],
                "precision": "low",
                "summary_mode": "statistics"
            }
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config["statistics"])
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["device_stat_precision_mode"], "low")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        self.assertIn("kernel_kbyk_dump.json", mock_save_json.call_args_list[0][0][0])
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")

        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)
        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_when_default_then_pass(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2",
            "statistics": {
                "list": [],
                "data_mode": ["all"],
                "summary_mode": "statistics"
            }
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["device_stat_precision_mode"], "high")
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["stat_calc_mode"], "host")
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["enable"], True)

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        self.assertIn("kernel_kbyk_dump.json", mock_save_json.call_args_list[0][0][0])
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")

        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)
        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor_input(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "data_mode": [
            "input"
            ],
            "level": "L2"

        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()

        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor_output(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "data_mode": [
            "output"
            ],
            "level": "L2"

        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor_output_rank_0(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "rank": [0],
            "step": [0, 2],
            "data_mode": [
            "output"
            ],
            "level": "L2"

        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor_output_rank_0(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "rank": [0],
            "step": [0, 2],
            "data_mode": [
            "output"
            ],
            "level": "L2"

        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor_list_output(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "list": ['add'],
            "rank": [0],
            "step": [0, 2],
            "data_mode": [
            "output"
            ],
            "level": "L2"

        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor_list_input(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "list": ['add'],
            "rank": [0],
            "step": [0, 2],
            "data_mode": [
            "input"
            ],
            "level": "L2"

        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_statistics(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2",
            "statistics": {
                "list": [],
                "data_mode": ["all"],
                "device": "host",
                "summary_mode": ["hash", "md5", "max", "mean"]
            }
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config["statistics"])
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["stat_calc_mode"], "host")
        self.assertEqual(dumper.dump_json["common_dump_settings"]["saved_data"], "statistic")
        if ms_version > "2.7.0":
            self.assertEqual(Counter(dumper.dump_json["common_dump_settings"]["statistic_category"]), Counter(["max", "hash", "hash:md5", "avg"]))
        else:
            self.assertEqual(Counter(dumper.dump_json["common_dump_settings"]["statistic_category"]), Counter(["max", "md5", "avg"]))
        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        self.assertIn("kernel_kbyk_dump.json", mock_save_json.call_args_list[0][0][0])
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")

        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)
        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]