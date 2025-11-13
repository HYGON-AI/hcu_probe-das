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

from msprobe.core.compare.utils import _compare_parser
from msprobe.core.compare.compare_cli import compare_cli
from msprobe.core.config_check.config_check_cli import _config_checking_parser, _run_config_checking_command
from msprobe.overflow_check.analyzer import _overflow_check_parser, _run_overflow_check
from msprobe.core.acc_check.acc_check_cli import acc_check_cli, multi_acc_check_cli
from msprobe.core.common.log import logger
from msprobe.visualization.graph_service import _graph_service_parser, _graph_service_command


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="msprobe(mindstudio probe), [Powered by MindStudio].\n"
                    "Providing one-site accuracy difference debugging toolkit for training on Ascend Devices.\n"
                    f"For any issue, refer README.md first",
    )

    parser.set_defaults(print_help=parser.print_help)
    subparsers = parser.add_subparsers()

    compare_parser = subparsers.add_parser('compare')
    _compare_parser(compare_parser)
    acc_check_cmd_parser = subparsers.add_parser('acc_check')
    multi_acc_check_cmd_parser = subparsers.add_parser('multi_acc_check')

    overflow_check_parse = subparsers.add_parser('overflow_check')
    _overflow_check_parser(overflow_check_parse)
    config_checking_parser = subparsers.add_parser('config_check')
    _config_checking_parser(config_checking_parser)

    graph_service_cmd_parser = subparsers.add_parser('graph_visualize')
    _graph_service_parser(graph_service_cmd_parser)
    graph_service_cmd_parser_deprecated = subparsers.add_parser('graph')
    _graph_service_parser(graph_service_cmd_parser_deprecated)

    if len(sys.argv) >= 2 and sys.argv[1] == "acc_check":
        acc_check_cli(sys.argv[2:])
    elif len(sys.argv) >= 2 and sys.argv[1] == "multi_acc_check":
        multi_acc_check_cli(sys.argv[2:])
    elif len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(sys.argv[1:])
    if sys.argv[1] == "compare":
        compare_cli(args)
    elif sys.argv[1] == "overflow_check":
        _run_overflow_check(args)
    elif sys.argv[1] == "graph_visualize":
        _graph_service_command(args)
    elif sys.argv[1] == "graph":
        logger.warning('The "graph" parameter has been deprecated and will be removed in future versions. '
                       'Please use the "graph_visualize" parameter instead.')
        _graph_service_command(args)
    elif sys.argv[1] == "config_check":
        _run_config_checking_command(args)


if __name__ == "__main__":
    main()
