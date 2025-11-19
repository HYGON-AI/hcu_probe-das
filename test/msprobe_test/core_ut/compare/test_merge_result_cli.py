import unittest
from unittest.mock import patch
import argparse
from msprobe.core.compare.merge_result.merge_result_cli import _merge_result_parser, merge_result_cli


class TestMergeResultCLI(unittest.TestCase):
    @patch('msprobe.core.compare.merge_result.merge_result_cli.merge_result')
    def test_merge_result_cli_success(self, mock_merge_result):
        args = [
            '-i', '/path/to/input',
            '-o', '/path/to/output',
            '-config', '/path/to/config.yaml'
        ]

        parser = argparse.ArgumentParser()
        _merge_result_parser(parser)
        parsed_args = parser.parse_args(args)

        merge_result_cli(parsed_args)

        mock_merge_result.assert_called_once_with(
            '/path/to/input', '/path/to/output', '/path/to/config.yaml'
        )
