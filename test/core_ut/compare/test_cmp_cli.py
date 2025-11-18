import unittest
from unittest.mock import patch, MagicMock

from msprobe.core.compare.auto_compare import mix_compare


class TestMixCompare(unittest.TestCase):
    @patch('msprobe.core.compare.auto_compare.get_paired_dirs')
    @patch('msprobe.core.compare.auto_compare.auto_compare.compare_auto_mode')
    def test_mix_compare_with_matching_dirs(self, mock_compare_cli, mock_get_paired_dirs):
        mock_args = MagicMock()
        mock_args.output_path = "/output"
        mock_args.target_path = "/npu_dump"
        mock_args.golden_path = "/bench_dump"
        mock_get_paired_dirs.side_effect = [
            ["graph", "pynative"],  # 第一次调用的返回值
            ["step1", "step2"],  # 第二次调用的返回值
            ["step1", "step2"]  # 第三次调用的返回值
        ]

        result = mix_compare(mock_args, 1)

        self.assertTrue(result)

    @patch('msprobe.core.compare.auto_compare.get_paired_dirs')
    @patch('msprobe.core.compare.auto_compare.auto_compare.compare_auto_mode')
    def test_mix_compare_no_matching_dirs(self, mock_compare_cli, mock_get_paired_dirs):
        mock_args = MagicMock()
        mock_args.output_path = "/output"
        mock_args.target_path = "/npu_dump"
        mock_args.golden_path = "/bench_dump"
        mock_get_paired_dirs.return_value = set()

        result = mix_compare(mock_args, 1)

        self.assertFalse(result)
