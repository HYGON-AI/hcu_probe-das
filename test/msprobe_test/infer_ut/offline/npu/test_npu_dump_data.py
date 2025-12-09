# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import unittest
from collections import namedtuple
from unittest.mock import patch, MagicMock, Mock

from msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data import (
    DynamicInput
)

DynamicArgInfo = namedtuple('DynamicArgInfo', ['atc_arg', 'benchmark_arg'])


class TestDynamicInput(unittest.TestCase):

    def setUp(self):
        self.mock_om_parser = MagicMock()
        self.mock_om_parser.shape_range = None

        self.dym_batch_enum = Mock()
        self.dym_batch_enum.value = DynamicArgInfo(atc_arg="--dym_batch", benchmark_arg="--batch")

        self.dym_shape_enum = Mock()
        self.dym_shape_enum.value = DynamicArgInfo(atc_arg="--dym_shape", benchmark_arg="--shape")

        self.dym_dims_enum = Mock()
        self.dym_dims_enum.value = DynamicArgInfo(atc_arg="--dym_dims", benchmark_arg="--dims")

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    def test_init_success(self, mock_get_arg_value, mock_get_dynamic_arg):
        mock_get_dynamic_arg.return_value = ("--dym_batch=10", "DYM_BATCH")
        mock_get_arg_value.return_value = "8"

        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        self.assertEqual(di.atc_dynamic_arg, "--dym_batch=10")
        self.assertEqual(di.cur_dynamic_arg, "DYM_BATCH")
        self.assertEqual(di.dynamic_arg_value, "8")

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicArgumentEnum')
    def test_get_dynamic_arg_from_om_found_exact(self, mock_enum):
        mock_enum.__iter__.return_value = [
            Mock(value=DynamicArgInfo(atc_arg="--dym_batch", benchmark_arg="--batch")),
            Mock(value=DynamicArgInfo(atc_arg="--dym_dims", benchmark_arg="--dims")),
        ]

        self.mock_om_parser.get_atc_cmdline.return_value = "atc --dym_batch 10 --input_shape input:1,3,224,224"

        result = DynamicInput.get_dynamic_arg_from_om(self.mock_om_parser)

        self.assertEqual(result[0], "--dym_batch=10")
        self.assertEqual(result[1], mock_enum.__iter__.return_value[0])

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicArgumentEnum')
    def test_get_dynamic_arg_from_om_found_with_equal(self, mock_enum):
        mock_enum.__iter__.return_value = [
            Mock(value=DynamicArgInfo(atc_arg="--dym_batch", benchmark_arg="--batch")),
            Mock(value=DynamicArgInfo(atc_arg="--dym_dims", benchmark_arg="--dims"))
        ]
        self.mock_om_parser.get_atc_cmdline.return_value = "atc --dym_batch=10 --input_shape input:1,3,224,224"

        result = DynamicInput.get_dynamic_arg_from_om(self.mock_om_parser)

        self.assertEqual(result[0], "--dym_batch=10")
        self.assertEqual(result[1], mock_enum.__iter__.return_value[0])

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicArgumentEnum')
    def test_get_dynamic_arg_from_om_not_found(self, mock_enum):
        mock_enum.__iter__.return_value = [
            Mock(value=DynamicArgInfo(atc_arg="--dym_batch", benchmark_arg="batch")),
            Mock(value=DynamicArgInfo(atc_arg="--dym_dims", benchmark_arg="dims")),
        ]
        self.mock_om_parser.get_atc_cmdline.return_value = "atc --input_shape input:1,3,224,224"

        result = DynamicInput.get_dynamic_arg_from_om(self.mock_om_parser)

        self.assertEqual(result, ("", None))

    def test_get_input_shape_from_om_with_space(self):
        self.mock_om_parser.get_atc_cmdline.return_value = "atc --input_shape input:1,3,224,224"

        result = DynamicInput.get_input_shape_from_om(self.mock_om_parser)

        self.assertEqual(result, "input:1,3,224,224")

    def test_get_input_shape_from_om_with_equal(self):
        self.mock_om_parser.get_atc_cmdline.return_value = "atc --input_shape=input:1,3,224,224"

        result = DynamicInput.get_input_shape_from_om(self.mock_om_parser)

        self.assertEqual(result, "input:1,3,224,224")

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.logger')
    def test_get_arg_value_non_dynamic(self, mock_logger):
        self.mock_om_parser.get_dynamic_scenario_info.return_value = (False, None)

        result = DynamicInput.get_arg_value(self.mock_om_parser, "input:1,3,224,224")

        self.assertEqual(result, "")
        mock_logger.info.assert_called_with("The input of model is not dynamic.")

    def test_get_arg_value_shape_range(self):
        """测试shape_range存在的情况"""
        self.mock_om_parser.get_dynamic_scenario_info.return_value = (True, None)
        self.mock_om_parser.shape_range = "range_info"

        result = DynamicInput.get_arg_value(self.mock_om_parser, "input:1,3,224,224")

        self.assertEqual(result, "input:1,3,224,224")

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils.parse_input_shape')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_dim_values')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.INPUT_SHAPE', '--input_shape')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.logger')
    def test_get_arg_value_dynamic_batch_success(self, mock_logger, mock_get_dim, mock_parse):
        self.mock_om_parser.get_dynamic_scenario_info.return_value = (True, None)
        self.mock_om_parser.get_atc_cmdline.return_value = "atc --input_shape input:-1,3,224,224"

        mock_parse.side_effect = [
            {"input": ["-1", "3", "224", "224"]},
            {"input": ["8", "3", "224", "224"]}
        ]

        batch_set = set()
        batch_set.add("8")
        mock_get_dim.side_effect = lambda a, b, s: s.update(["8"])

        result = DynamicInput.get_arg_value(self.mock_om_parser, "input:8,3,224,224")

        self.assertEqual(result, "8")
        mock_logger.error.assert_not_called()

    def test_get_dynamic_dim_values_list(self):
        """测试get_dynamic_dim_values使用列表收集值"""
        dym_shape = ["-1", "3", "-1"]
        cur_shape = ["8", "3", "224"]
        shape_values = []

        DynamicInput.get_dynamic_dim_values(dym_shape, cur_shape, shape_values)
        self.assertEqual(shape_values, [8, 224])

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.check_input_dynamic_arg_valid')
    def test_add_dynamic_arg_for_benchmark_dynamic(self, mock_check_valid, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("--dym_batch=10", self.dym_batch_enum)
        mock_get_arg_value.return_value = "8"

        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        benchmark_cmd = ["benchmark"]
        di.add_dynamic_arg_for_benchmark(benchmark_cmd)

        self.assertEqual(benchmark_cmd, ["benchmark", "--batch", "8"])
        mock_check_valid.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    def test_is_dynamic_shape_scenario_true(self, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("--dym_batch=10", self.dym_batch_enum)
        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        self.assertTrue(di.is_dynamic_shape_scenario())

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    def test_is_dynamic_shape_scenario_false(self, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("", self.dym_batch_enum)
        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        self.assertFalse(di.is_dynamic_shape_scenario())

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    def test_judge_dynamic_shape_scenario_match(self, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("--dym_batch=10", self.dym_batch_enum)
        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        self.assertTrue(di.judge_dynamic_shape_scenario("--dym_batch"))

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    def test_judge_dynamic_shape_scenario_not_match(self, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("--dym_dims=10", self.dym_batch_enum)
        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        self.assertFalse(di.judge_dynamic_shape_scenario("--dym_batch"))

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.check_dynamic_dims_valid')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.check_dynamic_batch_valid')
    def test_check_input_dynamic_arg_valid_dym_shape(
            self,
            mock_check_batch,
            mock_check_dims,
            mock_get_dynamic_arg,
            mock_get_arg_value
    ):
        mock_get_dynamic_arg.return_value = ("--dym_shape=", self.dym_batch_enum)
        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        di.check_input_dynamic_arg_valid()

        mock_check_batch.assert_not_called()
        mock_check_dims.assert_not_called()

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.check_dynamic_dims_valid')
    def test_check_input_dynamic_arg_valid_dym_dims(self, mock_check_dims, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("--dym_dims=5~10", self.dym_batch_enum)
        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        di.check_input_dynamic_arg_valid()

        mock_check_dims.assert_not_called()

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.check_dynamic_batch_valid')
    def test_check_input_dynamic_arg_valid_dym_batch(self, mock_check_batch, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("--dym_batch=5~10", self.dym_batch_enum)
        mock_get_arg_value.return_value = ""

        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        di.check_input_dynamic_arg_valid()

        mock_check_batch.assert_not_called()

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils.parse_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils.parse_value_by_comma')
    def test_check_dynamic_batch_valid_success(
        self,
        mock_parse_comma,
        mock_parse_arg,
        mock_get_dynamic_arg,
        mock_get_arg_value
    ):
        mock_get_dynamic_arg.return_value = ("--dym_batch=10", self.dym_batch_enum)
        mock_get_arg_value.return_value = "8"

        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        mock_parse_arg.return_value = ["5", "6", "7", "8", "9", "10"]
        mock_parse_comma.return_value = "8"

        di.check_dynamic_batch_valid("5~10")

        mock_parse_arg.assert_called_once_with("5~10")
        mock_parse_comma.assert_called_once_with("8")
