#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from msprobe.core.common.utils import CompareException
from msprobe.core.compare import torchair_acc_cmp as tac


class TestTorchairAccCmp(unittest.TestCase):
    def test_get_rank_id_from_torchair_data_when_valid_worldsize_rank_then_pass(self):
        self.assertEqual(tac.get_rank_id_from_torchair_data("worldsize8_rank3"), 3)
        self.assertEqual(tac.get_rank_id_from_torchair_data("worldsize2_rank0"), 0)

    def test_get_rank_id_from_torchair_data_when_invalid_name_then_pass(self):
        self.assertEqual(tac.get_rank_id_from_torchair_data("rank3"), -1)
        self.assertEqual(tac.get_rank_id_from_torchair_data("worldsize_rankX"), -1)
        self.assertEqual(tac.get_rank_id_from_torchair_data("foo_worldsize8_rank"), -1)

    def test_get_unique_key_when_key_conflicts_then_pass(self):
        cur_dict = {"k": 1, "k#1": 2}
        new_key = tac.get_unique_key(cur_dict, "k")
        self.assertEqual(new_key, "k#2")
        cur_dict[new_key] = 3
        self.assertIn("k#2", cur_dict)

    def test_parse_pbtxt_to_dict_when_nested_blocks_then_pass(self):
        content = """
node {
  name: "node_a"
  attr {
    key: "k"
    value: "v"
  }
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pbtxt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            result = tac.parse_pbtxt_to_dict(path)

        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)
        root = result[0]
        self.assertIn("node", root)
        self.assertEqual(root["node"]["name"], "node_a")
        self.assertEqual(root["node"]["attr"]["key"], "k")
        self.assertEqual(root["node"]["attr"]["value"], "v")

    def test_judge_single_or_multi_device_when_multi_device_in_root_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "device0"))
            os.makedirs(os.path.join(tmp, "device1"))

            self.assertTrue(tac.judge_single_or_multi_device(tmp))

    def test_judge_single_or_multi_device_when_time_based_multi_device_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            time_dir = "20250101120000"
            time_path = os.path.join(tmp, time_dir)
            os.makedirs(os.path.join(time_path, "0"))
            os.makedirs(os.path.join(time_path, "1"))

            self.assertTrue(tac.judge_single_or_multi_device(tmp))

    def test_judge_single_or_multi_device_when_single_device_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            time_dir = "20250101120000"
            time_path = os.path.join(tmp, time_dir)
            os.makedirs(os.path.join(time_path, "0"))

            self.assertFalse(tac.judge_single_or_multi_device(tmp))

    def test__has_rank_directory_when_rank_subdir_exists_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "worldsize2_rank0"))
            os.makedirs(os.path.join(tmp, "foo"))

            self.assertTrue(tac._has_rank_directory(tmp))

    def test__has_rank_directory_when_no_rank_subdir_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "foo"))

            self.assertFalse(tac._has_rank_directory(tmp))

    @patch("msprobe.core.compare.torchair_acc_cmp.FileChecker")
    @patch("msprobe.core.compare.torchair_acc_cmp.os.path.isdir")
    def test__validate_read_path_when_dir_and_file_then_pass(self, mock_isdir, mock_filechecker):
        instance = MagicMock()
        mock_filechecker.return_value = instance

        # directory path
        mock_isdir.return_value = True
        tac._validate_read_path("/tmp/dir")
        mock_filechecker.assert_called_with("/tmp/dir", tac.FileCheckConst.DIR, ability=tac.FileCheckConst.READ_ABLE)
        instance.common_check.assert_called_once()

        # file path
        instance.common_check.reset_mock()
        mock_isdir.return_value = False
        tac._validate_read_path("/tmp/file")
        mock_filechecker.assert_called_with("/tmp/file", tac.FileCheckConst.FILE, ability=tac.FileCheckConst.READ_ABLE)
        instance.common_check.assert_called_once()

    def test_gather_data_with_token_id_fx_when_rank_info_existed_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            # create token dirs "0" and "1", each with one .npy file
            token0 = os.path.join(tmp, "0")
            token1 = os.path.join(tmp, "1")
            os.makedirs(token0)
            os.makedirs(token1)
            f0 = os.path.join(token0, "a.npy")
            f1 = os.path.join(token1, "b.npy")
            open(f0, "wb").close()
            open(f1, "wb").close()

            gathered = tac.gather_data_with_token_id_fx(tmp, [], rank_info_existed=True)

        self.assertEqual(len(gathered), 1)
        gathered_map = gathered[0]
        # token id is basename(token_dir)+1
        self.assertIn(1, gathered_map)
        self.assertIn(2, gathered_map)
        self.assertEqual(gathered_map[1], [f0])
        self.assertEqual(gathered_map[2], [f1])

    def test_gather_data_with_token_id_when_rank_info_existed_true_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = os.path.join(tmp, "parent")
            os.makedirs(parent)
            token0 = os.path.join(parent, "0")
            token1 = os.path.join(parent, "1")
            os.makedirs(token0)
            os.makedirs(token1)
            f0 = os.path.join(token0, "x.bin")
            f1 = os.path.join(token1, "y.bin")
            open(f0, "wb").close()
            open(f1, "wb").close()

            gathered = tac.gather_data_with_token_id(parent, fx=False, rank_info_existed=True)

        self.assertEqual(len(gathered), 1)
        subdir_map = gathered[0]
        self.assertIn(0, subdir_map)
        self.assertIn(1, subdir_map)
        self.assertIn(f0, subdir_map[0])
        self.assertIn(f1, subdir_map[1])

    @patch("msprobe.core.compare.torchair_acc_cmp.gather_data_with_token_id")
    def test_init_ge_dump_data_from_bin_path_when_valid_files_then_pass(self, mock_gather):
        mock_gather.return_value = [
            {
                0: [
                    "d/1/Add.Add_2.44.6.1706596912161941",
                    "d/1/Cast.Cast_9.19.6.1706596911887829",
                ]
            }
        ]
        result = tac.init_ge_dump_data_from_bin_path("dummy_path")

        self.assertEqual(len(result), 1)
        token_map = result[0]
        self.assertIn(0, token_map)
        op_map = token_map[0]
        self.assertEqual(op_map["Add_2"], "d/1/Add.Add_2.44.6.1706596912161941")
        self.assertEqual(op_map["Cast_9"], "d/1/Cast.Cast_9.19.6.1706596911887829")

    @patch("msprobe.core.compare.torchair_acc_cmp.gather_data_with_token_id")
    def test_init_fx_dump_data_from_path_when_valid_files_then_pass(self, mock_gather):
        mock_gather.return_value = [
            {
                1: [
                    "d/1/mm-aten.mm.default.INPUT.0.20240125031118787351.npy",
                    "d/1/mm-aten.mm.default.INPUT.1.20240125031118787351.npy",
                    "d/1/mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy",
                ]
            }
        ]
        result = tac.init_fx_dump_data_from_path("dummy_path", rank_info_existed=False)

        self.assertEqual(len(result), 1)
        token_map = result[0]
        # FX token id is decremented by 1
        self.assertIn(0, token_map)
        op_map = token_map[0]
        self.assertIn("mm-aten.mm.default", op_map)
        self.assertEqual(len(op_map["mm-aten.mm.default"]["input"]), 2)
        self.assertEqual(len(op_map["mm-aten.mm.default"]["output"]), 1)

    @patch("msprobe.core.compare.torchair_acc_cmp.acc_compare")
    @patch("msprobe.core.compare.torchair_acc_cmp._has_rank_directory", return_value=False)
    @patch("msprobe.core.compare.torchair_acc_cmp._validate_read_path")
    @patch("msprobe.core.compare.torchair_acc_cmp.logger")
    def test_compare_torchair_mode_when_invalid_rank_then_raise(
        self, mock_logger, mock_validate, mock_has_rank_dir, mock_acc_compare
    ):
        args = SimpleNamespace(target_path="/tmp/my", golden_path="/tmp/golden", output_path="/tmp/out", rank="abc")

        with self.assertRaises(CompareException) as cm:
            tac.compare_torchair_mode(args)

        self.assertEqual(cm.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_logger.error.assert_called_once()
        mock_acc_compare.assert_not_called()
        mock_validate.assert_any_call(os.path.realpath(args.target_path))
        mock_validate.assert_any_call(os.path.realpath(args.golden_path))

    @patch("msprobe.core.compare.torchair_acc_cmp.acc_compare", return_value="ok")
    @patch("msprobe.core.compare.torchair_acc_cmp._has_rank_directory", return_value=False)
    @patch("msprobe.core.compare.torchair_acc_cmp._validate_read_path")
    @patch("msprobe.core.compare.torchair_acc_cmp.logger")
    def test_compare_torchair_mode_when_rank_without_rank_dirs_then_pass(
        self, mock_logger, mock_validate, mock_has_rank_dir, mock_acc_compare
    ):
        args = SimpleNamespace(target_path="/tmp/my", golden_path="/tmp/golden", output_path="/tmp/out", rank="1")

        result = tac.compare_torchair_mode(args)

        self.assertEqual(result, "ok")
        mock_logger.warning.assert_called_once()
        mock_acc_compare.assert_called_once()
        _, kwargs = mock_acc_compare.call_args
        # acc_compare(golden_path, my_path, output_path, rank_id, rank_info_existed)
        self.assertEqual(len(mock_acc_compare.call_args[0]), 5)

    @patch("msprobe.core.compare.torchair_acc_cmp.acc_compare", return_value="ok")
    @patch("msprobe.core.compare.torchair_acc_cmp._has_rank_directory", return_value=True)
    @patch("msprobe.core.compare.torchair_acc_cmp._validate_read_path")
    @patch("msprobe.core.compare.torchair_acc_cmp.logger")
    def test_compare_torchair_mode_when_no_rank_then_pass(
        self, mock_logger, mock_validate, mock_has_rank_dir, mock_acc_compare
    ):
        args = SimpleNamespace(target_path="/tmp/my", golden_path="/tmp/golden", output_path="/tmp/out")

        result = tac.compare_torchair_mode(args)

        self.assertEqual(result, "ok")
        mock_logger.warning.assert_not_called()
        mock_acc_compare.assert_called_once()
        # rank_id should be None when args.rank is missing
        self.assertIsNone(mock_acc_compare.call_args[0][3])

    @patch("msprobe.core.compare.torchair_acc_cmp.get_torchair_ge_graph_path", return_value=[])
    def test_acc_compare_once_when_no_ge_graph_then_raise(self, mock_get_graph):
        args = ("/golden", "/my", "/out", -1)

        with self.assertRaises(Exception) as cm:
            tac.acc_compare_once(args)

        self.assertIn("Can not get ge graph", str(cm.exception))
        mock_get_graph.assert_called_once_with("/my", -1)


if __name__ == "__main__":
    unittest.main()
