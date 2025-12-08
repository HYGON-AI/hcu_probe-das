import unittest
import argparse
from msprobe.mindspore.api_accuracy_checker.cmd_parser import add_api_accuracy_checker_argument


class TestApiAccuracyCheckerArgument(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()

    def test_api_info_argument(self):
        # 测试 -api_info 参数是否能正确解析
        self.parser.add_argument("-api_info", "--api_info_file", dest="api_info_file", type=str, required=True)
        args = self.parser.parse_args(["-api_info", "test.json"])
        self.assertEqual(args.api_info_file, "test.json")

    def test_out_path_argument(self):
        # 测试 -o 参数是否能正确解析并使用默认值
        self.parser.add_argument("-o", "--out_path", dest="out_path", default="./", type=str, required=False)
        args = self.parser.parse_args(["-o", "/tmp/"])
        self.assertEqual(args.out_path, "/tmp/")

        # 测试不传入 -o 参数时是否使用默认值
        args_default = self.parser.parse_args([])
        self.assertEqual(args_default.out_path, "./")


if __name__ == "__main__":
    unittest.main()