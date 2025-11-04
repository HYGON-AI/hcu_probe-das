# Copyright (c) 2024, Huawei Technologies Co., Ltd.
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

import argparse
import sys
import importlib.util

from msprobe.core.compare.utils import _compare_parser
from msprobe.core.compare.compare_cli import compare_cli
from msprobe.core.compare.merge_result.merge_result_cli import _merge_result_parser, merge_result_cli
from msprobe.core.config_check.config_check_cli import _config_checking_parser, _run_config_checking_command


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="msprobe(mindstudio probe), [Powered by MindStudio].\n"
                    "Providing one-site accuracy difference debugging toolkit for training on Ascend Devices.\n"
                    f"For any issue, refer README.md first",
    )

    parser.set_defaults(print_help=parser.print_help)
    subparsers = parser.add_subparsers()

    compare_cmd_parser = subparsers.add_parser('compare')
    _compare_parser(compare_cmd_parser)
    merge_result_parser = subparsers.add_parser('merge_result')
    _merge_result_parser(merge_result_parser)
    config_checking_parser = subparsers.add_parser('config_check')
    _config_checking_parser(config_checking_parser)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(sys.argv[1:])
    if sys.argv[1] == "compare":
        compare_cli(args)
    elif sys.argv[1] == "merge_result":
        merge_result_cli(args)
    elif sys.argv[1] == "config_check":
        _run_config_checking_command(args)


if __name__ == "__main__":
    main()
