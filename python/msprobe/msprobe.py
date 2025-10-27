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

from python.msprobe.core.common.const import Const
from python.msprobe.core.common.log import logger
from python.msprobe.core.compare.utils import _compare_parser
from python.msprobe.core.compare.compare_cli import compare_cli
from python.msprobe.core.compare.merge_result.merge_result_cli import _merge_result_parser, merge_result_cli
from python.msprobe.core.config_check.config_check_cli import _config_checking_parser, _run_config_checking_command


def is_module_available(module_name):
    spec = importlib.util.find_spec(module_name)
    return spec is not None


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="msprobe(mindstudio probe), [Powered by MindStudio].\n"
                    "Providing one-site accuracy difference debugging toolkit for training on Ascend Devices.\n"
                    f"For any issue, refer README.md first",
    )

    parser.set_defaults(print_help=parser.print_help)
    parser.add_argument('-f', '--framework', required=True,
                        choices=[Const.PT_FRAMEWORK, Const.MS_FRAMEWORK],
                        help='Deep learning framework.')
    subparsers = parser.add_subparsers()
    subparsers.add_parser('parse')
    compare_cmd_parser = subparsers.add_parser('compare')
    merge_result_parser = subparsers.add_parser('merge_result')
    _compare_parser(compare_cmd_parser)
    _merge_result_parser(merge_result_parser)

    is_torch_available = is_module_available("torch")

    if len(sys.argv) < 4:
        parser.print_help()
        sys.exit(0)
    framework_args = parser.parse_args(sys.argv[1:3])
    if framework_args.framework == Const.PT_FRAMEWORK:
        pass
    elif framework_args.framework == Const.MS_FRAMEWORK:
        pass

    args = parser.parse_args(sys.argv[1:])
    if sys.argv[2] == Const.PT_FRAMEWORK:
        if not is_torch_available:
            logger.error("PyTorch does not exist, please install PyTorch library")
            raise Exception("PyTorch does not exist, please install PyTorch library")
        if sys.argv[3] == "compare":
            if args.cell_mapping is not None or args.api_mapping is not None:
                logger.error("Argument -cm or -am is not supported in PyTorch framework")
                raise Exception("Argument -cm or -am is not supported in PyTorch framework")
            compare_cli(args)
        elif sys.argv[3] == "merge_result":
            merge_result_cli(args)
    else:
        if not is_module_available(Const.MS_FRAMEWORK):
            logger.error("MindSpore does not exist, please install MindSpore library")
            raise Exception("MindSpore does not exist, please install MindSpore library")
        if sys.argv[3] == "compare":
            compare_cli(args)
        elif sys.argv[3] == "merge_result":
            merge_result_cli(args)


if __name__ == "__main__":
    main()
