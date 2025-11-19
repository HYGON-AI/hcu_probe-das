# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import mindspore as ms
import pandas as pd

from msprobe.mindspore.dump.dump_processor.cell_dump_process import cell_construct_wrapper
from msprobe.mindspore.dump.dump_processor.cell_dump_process import check_relation
from msprobe.mindspore.dump.dump_processor.cell_dump_process import convert_special_values, sort_filenames
from msprobe.mindspore.dump.dump_processor.cell_dump_process import create_kbyk_json
from msprobe.mindspore.dump.dump_processor.cell_dump_process import process_csv, np_ms_dtype_dict

from msprobe.core.common.const import Const as CoreConst
from msprobe.mindspore.dump.dump_processor import cell_dump_process


class TestCellWrapperProcess(unittest.TestCase):

    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.ops.is_tensor')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.td')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.td_in')
    def test_cell_construct_wrapper(self, mock_td_in, mock_td, mock_istensor):

        # Mock the TensorDump operations
        mock_td.return_value = MagicMock()
        mock_td_in.return_value = MagicMock()
        mock_istensor.return_value = False

        # Create a mock cell with necessary attributes
        mock_cell = MagicMock()
        mock_cell.data_mode = "all"
        mock_cell.dump_path = "mock_dump_path"
        mock_cell.cell_prefix = "mock_cell_prefix"

        # Define a mock function to wrap
        def mock_func(*args, **kwargs):
            return args

        # Wrap the mock function using cell_construct_wrapper
        wrapped_func = cell_construct_wrapper(mock_func, mock_cell)

        # Create mock inputs
        mock_input = ms.Tensor([1, 2, 3])
        mock_args = (mock_input,)

        # Call the wrapped function
        wrapped_func(mock_cell, *mock_args)

        # Verify that the TensorDump operations were not called
        mock_td_in.assert_not_called()
        mock_td.assert_not_called()


class TestSortFilenames(unittest.TestCase):

    @patch('os.listdir')
    def test_sort_filenames(self, mock_listdir):
        # Mock the list of filenames returned by os.listdir
        mock_listdir.return_value = [
            'Cell.network._backbone.model.LlamaModel.backward.0.input.0_float16_177.npy',
            'Cell.network._backbone.model.LlamaModel.forward.0.input.0_in_int32_1.npy',
            'Cell.network._backbone.model.LlamaModel.forward.0.output.10_float16_165.npy',
            'Cell.network._backbone.model.norm_out.LlamaRMSNorm.backward.0.input.0_float16_178.npy'
        ]

        # Mock the CoreConst values
        CoreConst.REPLACEMENT_CHARACTER = '_'
        CoreConst.NUMPY_SUFFIX = '.npy'

        # Expected sorted filenames
        expected_sorted_filenames = [
            'Cell.network._backbone.model.LlamaModel.forward.0.input.0_in_int32_1.npy',
            'Cell.network._backbone.model.LlamaModel.forward.0.output.10_float16_165.npy',
            'Cell.network._backbone.model.LlamaModel.backward.0.input.0_float16_177.npy',
            'Cell.network._backbone.model.norm_out.LlamaRMSNorm.backward.0.input.0_float16_178.npy'
        ]

        # Call the function
        sorted_filenames = sort_filenames('/mock/path')

        # Assert the filenames are sorted correctly
        self.assertEqual(sorted_filenames, expected_sorted_filenames)


class TestCheckRelation(unittest.TestCase):

    def setUp(self):
        CoreConst.SEP = '.'
        global KEY_LAYERS
        KEY_LAYERS = "layers"

    def test_direct_parent_child_relation(self):
        self.assertTrue(check_relation("network._backbone", "network"))
        self.assertTrue(check_relation("network._backbone.model", "network._backbone"))

    def test_no_relation(self):
        self.assertFalse(check_relation("network._backbone", "network.loss"))
        self.assertFalse(check_relation("network._backbone.model", "network.loss"))

    def test_layer_pattern_relation(self):
        self.assertTrue(check_relation("network.model.layers.0", "network.model"))
        self.assertTrue(check_relation("network._backbone.model.layers.1", "network._backbone.model"))

    def test_edge_cases(self):
        self.assertFalse(check_relation("", "network"))
        self.assertFalse(check_relation("network.layer1", ""))
        self.assertFalse(check_relation("", ""))


class TestRenameFilename(unittest.TestCase):
    def setUp(self):
        self.logger_patcher = patch.object(cell_dump_process, "logger", MagicMock())
        self.logger_patcher.start()

    def tearDown(self):
        self.logger_patcher.stop()

    @patch.object(cell_dump_process, "sort_filenames")
    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file")
    def test_rename_filename_tensor(self, mock_move_file, mock_sort_filenames):
        cell_dump_process.dump_task = CoreConst.TENSOR

        with tempfile.TemporaryDirectory() as tmpdir:
            filenames = [
                "Cell.a.b.c.X.forward.input.0_float32_1.npy",
                "Cell.a.b.c.X.forward.input.1_float32_2.npy",
                "Cell.a.b.c.X.forward.output.0_float32_3.npy",
                "Cell.a.b.c.X.forward.input.0_float32_11.npy",
                "Cell.a.b.c.X.forward.input.1_float32_12.npy",
                "Cell.a.b.c.X.forward.output.0_float32_13.npy",
                "Cell.a.b.c.X.backward.input.0_float32_30.npy"
            ]
            for fname in filenames:
                with open(os.path.join(tmpdir, fname), "wb") as f:
                    f.write(b"dummy")

            mock_sort_filenames.return_value = filenames

            rename_calls = []
            def fake_rename(src, dst):
                rename_calls.append((os.path.basename(src), os.path.basename(dst)))
            mock_move_file.side_effect = fake_rename

            cell_dump_process.rename_filename(path=tmpdir)

            expected = [
                ("Cell.a.b.c.X.forward.input.0_float32_1.npy", "Cell.a.b.c.X.forward.0.input.0_float32_1.npy"),
                ("Cell.a.b.c.X.forward.input.1_float32_2.npy", "Cell.a.b.c.X.forward.0.input.1_float32_2.npy"),
                ("Cell.a.b.c.X.forward.output.0_float32_3.npy", "Cell.a.b.c.X.forward.0.output.0_float32_3.npy"),
                ("Cell.a.b.c.X.forward.input.0_float32_11.npy", "Cell.a.b.c.X.forward.1.input.0_float32_11.npy"),
                ("Cell.a.b.c.X.forward.input.1_float32_12.npy", "Cell.a.b.c.X.forward.1.input.1_float32_12.npy"),
                ("Cell.a.b.c.X.forward.output.0_float32_13.npy", "Cell.a.b.c.X.forward.1.output.0_float32_13.npy"),
                ("Cell.a.b.c.X.backward.input.0_float32_30.npy", "Cell.a.b.c.X.backward.0.input.0_float32_30.npy")
            ]
            self.assertEqual(rename_calls, expected)

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file")
    def test_rename_filename_statistics(self, mock_move_file):
        cell_dump_process.dump_task = CoreConst.STATISTICS

        data = {
            'Op Name': [
                "Cell.a.b.c.X.forward.input.0",
                "Cell.a.b.c.X.forward.input.1",
                "Cell.a.b.c.X.forward.output.0",
                "Cell.a.b.c.X.forward.input.0",
                "Cell.a.b.c.X.forward.input.1",
                "Cell.a.b.c.X.forward.output.0",
                "Cell.a.b.c.X.backward.input.0"
            ]
        }
        df = pd.DataFrame(data)

        cell_dump_process.rename_filename(data_df=df)

        self.assertEqual(df['Op Name'].iloc[0], "Cell.a.b.c.X.forward.0.input.0")
        self.assertEqual(df['Op Name'].iloc[1], "Cell.a.b.c.X.forward.0.input.1")
        self.assertEqual(df['Op Name'].iloc[2], "Cell.a.b.c.X.forward.0.output.0")
        self.assertEqual(df['Op Name'].iloc[3], "Cell.a.b.c.X.forward.1.input.0")
        self.assertEqual(df['Op Name'].iloc[4], "Cell.a.b.c.X.forward.1.input.1")
        self.assertEqual(df['Op Name'].iloc[5], "Cell.a.b.c.X.forward.1.output.0")
        self.assertEqual(df['Op Name'].iloc[6], "Cell.a.b.c.X.backward.0.input.0")


class TestConvertSpecialValues(unittest.TestCase):
    TEST_CASES = [
        ("true", True),
        ("True", True),
        ("false", False),
        ("False", False),
        ("1.23", 1.23),
        ("0", 0.0),
        ("-5.6", -5.6),
        (42, 42),
        (3.14, 3.14),
        (pd.NA, None)
    ]

    def test_convert_special_values(self):
        for input_value, expected in self.TEST_CASES:
            result = convert_special_values(input_value)
            self.assertEqual(result, expected)


class TestProcessCsv(unittest.TestCase):

    @staticmethod
    def make_df(rows):
        import pandas as pd
        return pd.DataFrame(rows)

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv")
    def test_process_csv_input_and_output(self, mock_read_csv):
        rows = [
            {
                'Op Name': 'Cell.net.layer.forward.0.input.0',
                'Shape': '(2,3)',
                'Data Type': 'float32',
                'Max Value': 1.0,
                'Min Value': 0.0,
                'Avg Value': 0.5,
                'L2Norm Value': 2.0
            },
            {
                'Op Name': 'Cell.net.layer.forward.0.output.0',
                'Shape': '(2,3)',
                'Data Type': 'float32',
                'Max Value': 2.0,
                'Min Value': -1.0,
                'Avg Value': 0.0,
                'L2Norm Value': 3.0
            }
        ]
        df = self.make_df(rows)
        mock_read_csv.return_value = df

        result = process_csv("dummy_path")
        self.assertEqual(len(result), 2)

        op_name, key, tensor_json = result[0]
        self.assertEqual(op_name, 'Cell.net.layer.forward.0')
        self.assertEqual(key, CoreConst.INPUT_ARGS)
        self.assertEqual(tensor_json[CoreConst.TYPE], 'mindspore.Tensor')
        self.assertEqual(tensor_json[CoreConst.DTYPE], str(np_ms_dtype_dict['float32']))
        self.assertEqual(tensor_json[CoreConst.SHAPE], [2, 3])
        self.assertEqual(tensor_json[CoreConst.MAX], 1.0)
        self.assertEqual(tensor_json[CoreConst.MIN], 0.0)
        self.assertEqual(tensor_json[CoreConst.MEAN], 0.5)
        self.assertEqual(tensor_json[CoreConst.NORM], 2.0)

        op_name, key, tensor_json = result[1]
        self.assertEqual(op_name, 'Cell.net.layer.forward.0')
        self.assertEqual(key, CoreConst.OUTPUT)
        self.assertEqual(tensor_json[CoreConst.MAX], 2.0)
        self.assertEqual(tensor_json[CoreConst.MIN], -1.0)
        self.assertEqual(tensor_json[CoreConst.MEAN], 0.0)
        self.assertEqual(tensor_json[CoreConst.NORM], 3.0)

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv")
    def test_process_csv_handles_missing_columns(self, mock_read_csv):
        rows = [
            {
                'Op Name': 'Cell.net.layer.forward.0.input.0',
                'Shape': '(1,)',
                'Data Type': 'int32'
            }
        ]
        df = self.make_df(rows)
        mock_read_csv.return_value = df

        result = process_csv("dummy_path")
        self.assertEqual(len(result), 1)
        op_name, key, tensor_json = result[0]
        self.assertEqual(tensor_json[CoreConst.DTYPE], str(np_ms_dtype_dict['int32']))
        self.assertEqual(tensor_json[CoreConst.SHAPE], [1])

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv")
    def test_process_csv_handles_unknown_io_key(self, mock_read_csv):
        rows = [
            {
                'Op Name': 'Cell.net.layer.forward.0.unknown.0',
                'Shape': '(1,2)',
                'Data Type': 'float16'
            }
        ]
        df = self.make_df(rows)
        mock_read_csv.return_value = df

        result = process_csv("dummy_path")
        self.assertEqual(len(result), 1)
        op_name, key, tensor_json = result[0]
        self.assertIsNone(op_name)
        self.assertIsNone(key)
        self.assertIsNone(tensor_json)

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv")
    def test_process_csv_shape_parsing(self, mock_read_csv):
        rows = [
            {
                'Op Name': 'Cell.net.layer.forward.0.input.0',
                'Shape': '(4, 5, 6)',
                'Data Type': 'float64'
            }
        ]
        df = self.make_df(rows)
        mock_read_csv.return_value = df

        result = process_csv("dummy_path")
        self.assertEqual(result[0][2][CoreConst.SHAPE], [4, 5, 6])

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv")
    def test_process_csv_convert_special_values_bool_and_nan(self, mock_read_csv):
        rows = [
            {
                'Op Name': 'Cell.net.layer.forward.0.input.0',
                'Shape': '(1,)',
                'Data Type': 'float32',
                'Max Value': 'True',
                'Min Value': 'False',
                'Avg Value': float('nan'),
                'L2Norm Value': 1.23
            }
        ]
        df = self.make_df(rows)
        mock_read_csv.return_value = df

        result = process_csv("dummy_path")
        tensor_json = result[0][2]
        self.assertIs(tensor_json[CoreConst.MAX], True)
        self.assertIs(tensor_json[CoreConst.MIN], False)
        self.assertIsNone(tensor_json[CoreConst.MEAN])
        self.assertEqual(tensor_json[CoreConst.NORM], 1.23)


class TestCreateKbykJsonMultiRank(unittest.TestCase):
    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.create_directory", lambda path: None)
    @patch(
        "msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json",
        lambda path, data, indent=4: open(path, "w").write("test")
    )
    def test_create_kbyk_json_multi_rank(self):
        
        test_cases = [
            (None, "0kernel_kbyk_dump.json"),
            ("1", "1kernel_kbyk_dump.json"),
            ("3", "3kernel_kbyk_dump.json"),
        ]

        for rank_id_env, expected_prefix in test_cases:
            with tempfile.TemporaryDirectory() as dump_path:
                summary_mode = ["max"]
                step = 0
                # Patch environment variable
                if rank_id_env is not None:
                    with patch.dict(os.environ, {"RANK_ID": rank_id_env}):
                        config_json_path = create_kbyk_json(dump_path, summary_mode, step)
                else:
                    with patch.dict(os.environ, {}, clear=True):
                        config_json_path = create_kbyk_json(dump_path, summary_mode, step)
                self.assertEqual(os.path.basename(config_json_path), expected_prefix)
                self.assertTrue(config_json_path.startswith(dump_path))
