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
import json
import os
from msprobe.core.common.const import Const


def _detect_framework_from_api_info(api_info_path: str) -> str:
    """从 -api_info 指定的 dump.json 中读取 framework 字段."""
    if not api_info_path:
        raise ValueError("Argument -api_info is required to detect framework.")

    if not os.path.exists(api_info_path):
        raise FileNotFoundError(f"api_info file does not exist: {api_info_path}")

    with open(api_info_path, "r", encoding="utf-8") as f:
        try:
            info = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from {api_info_path}: {e}") from e

    framework = info.get("framework")
    if not framework:
        raise ValueError(f'Key "framework" not found in {api_info_path}')

    framework = str(framework).lower()
    # 统一映射到 Const 中定义的名字，方便复用其他逻辑
    if framework in ("pytorch", "pt", Const.PT_FRAMEWORK.lower()):
        return Const.PT_FRAMEWORK
    if framework in ("mindspore", "ms", Const.MS_FRAMEWORK.lower(), Const.MT_FRAMEWORK.lower()):
        return Const.MS_FRAMEWORK

    raise ValueError(f"Unsupported framework in api_info: {framework}")


def acc_check_cli(argv):
    """
    msprobe acc_check ... 的统一入口。
    1. 先解析出 -api_info
    2. 根据 dump.json 中的 framework 动态选择 PT/MS 的 parser + 命令。
    """
    # 第一阶段：只解析 -api_info，其他参数先不管
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-api_info", required=True, help="Path to api_info json file.")
    pre_args, _ = pre_parser.parse_known_args(argv)

    framework = _detect_framework_from_api_info(pre_args.api_info)

    if framework == Const.PT_FRAMEWORK:
        # PyTorch 路径：使用原来的 PT acc_check 实现
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import _acc_check_parser, acc_check_command

        pt_parser = argparse.ArgumentParser(
            prog="msprobe acc_check",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Run PyTorch UT with msprobe."
        )
        _acc_check_parser(pt_parser)  # 这里会给 parser 加上原来的所有 PT acc_check 参数（包括 -api_info）

        pt_args = pt_parser.parse_args(argv)
        acc_check_command(pt_args)

    elif framework == Const.MS_FRAMEWORK:
        # MindSpore 路径：复用原来的 MS api_checker_main
        from msprobe.mindspore.api_accuracy_checker.cmd_parser import add_api_accuracy_checker_argument
        from msprobe.mindspore.api_accuracy_checker.main import api_checker_main

        ms_parser = argparse.ArgumentParser(
            prog="msprobe acc_check",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Run MindSpore UT with msprobe."
        )
        add_api_accuracy_checker_argument(ms_parser)  # 给 acc_check 的 parser 加上原来 MS 的所有参数

        ms_args = ms_parser.parse_args(argv)
        api_checker_main(ms_args)


def multi_acc_check_cli(argv):
    """
    msprobe multi_acc_check ... 的统一入口。
    同样通过 -api_info -> dump.json -> framework 做分发。
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-api_info", required=True, help="Path to api_info json file.")
    pre_args, _ = pre_parser.parse_known_args(argv)

    framework = _detect_framework_from_api_info(pre_args.api_info)

    if framework == Const.PT_FRAMEWORK:
        # PyTorch 多进程路径：沿用原来的 prepare_config + run_parallel_ut
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import _acc_check_parser
        from msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check import prepare_config, run_parallel_ut

        pt_parser = argparse.ArgumentParser(
            prog="msprobe multi_acc_check",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Run PyTorch UT in parallel with msprobe."
        )
        _acc_check_parser(pt_parser)
        pt_parser.add_argument(
            "-n", "--num_splits",
            type=int,
            choices=range(1, 65),
            default=8,
            help="Number of splits for parallel processing. Range: 1-64"
        )

        pt_args = pt_parser.parse_args(argv)
        config = prepare_config(pt_args)
        run_parallel_ut(config)

    elif framework == Const.MS_FRAMEWORK:
        # MindSpore 多进程路径：沿用原来的 multi_add_api_accuracy_checker_argument + mul_api_checker_main
        from msprobe.mindspore.api_accuracy_checker.cmd_parser import multi_add_api_accuracy_checker_argument
        from msprobe.mindspore.api_accuracy_checker.main import mul_api_checker_main

        ms_parser = argparse.ArgumentParser(
            prog="msprobe multi_acc_check",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Run MindSpore UT in parallel with msprobe."
        )
        multi_add_api_accuracy_checker_argument(ms_parser)

        ms_args = ms_parser.parse_args(argv)
        mul_api_checker_main(ms_args)

