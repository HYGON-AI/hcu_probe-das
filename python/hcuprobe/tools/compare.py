"""dump 数据精度比对(HCU vs 标杆 / 任意两端 dump)。"""

import argparse
import sys
from typing import Optional

from hcu_probe.core.compare.compare_cli import compare_cli
from hcu_probe.core.compare.utils import _compare_parser


def run(
    *,
    target_path: str,
    golden_path: str,
    output_path: str = "./output",
    mode: str = "auto",
    fuzzy_match: bool = False,
    layer_mapping: Optional[str] = None,
    api_mapping: Optional[str] = None,
    data_mapping: Optional[str] = None,
    cell_mapping: Optional[str] = None,
    diff_analyze: bool = False,
    is_print_compare_log: bool = False,
) -> str:
    """对比两端 dump 数据,落盘 compare_result_{ts}.xlsx + advisor_{ts}.txt。"""
    parser = argparse.ArgumentParser()
    _compare_parser(parser)
    sys_argv = [
        "-tp", target_path,
        "-gp", golden_path,
        "-o", output_path,
        "-m", mode,
    ]
    if fuzzy_match:
        sys_argv.append("-fm")
    if layer_mapping:
        sys_argv += ["-lm", layer_mapping]
    if api_mapping:
        sys_argv += ["-am", api_mapping]
    if data_mapping:
        sys_argv += ["-dm", data_mapping]
    if cell_mapping:
        sys_argv += ["-cm", cell_mapping]
    if diff_analyze:
        sys_argv.append("-da")
    if is_print_compare_log:
        sys_argv.append("-tensor_log")
    args = parser.parse_args(sys_argv)
    compare_cli(args, sys_argv)
    return output_path
