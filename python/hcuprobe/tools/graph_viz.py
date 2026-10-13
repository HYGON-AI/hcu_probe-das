"""dump 图可视化(构图 / 跨平台比对图,产出 .vis.db 供 tensorboard 加载)。"""

import argparse
from typing import Optional, Sequence

from hcu_probe.visualization.graph_service import _graph_service_command, _graph_service_parser


def run(
    *,
    target_path: str,
    output_path: str,
    golden_path: Optional[str] = None,
    layer_mapping: Optional[str] = None,
    overflow_check: bool = False,
    fuzzy_match: bool = False,
    rank_size: Optional[Sequence[int]] = None,
    tp: Optional[Sequence[int]] = None,
    pp: Optional[Sequence[int]] = None,
    vpp: Optional[Sequence[int]] = None,
    order: Optional[Sequence[str]] = None,
    is_print_compare_log: bool = False,
    is_print_progress_log: bool = False,
) -> str:
    """构图或跨端比对图。无 golden_path 时单源构图,否则双源 diff。"""
    parser = argparse.ArgumentParser()
    _graph_service_parser(parser)
    sys_argv = ["-tp", target_path, "-o", output_path]
    if golden_path:
        sys_argv += ["-gp", golden_path]
    if layer_mapping:
        sys_argv += ["-lm", layer_mapping]
    if overflow_check:
        sys_argv.append("-oc")
    if fuzzy_match:
        sys_argv.append("-fm")
    if is_print_compare_log:
        sys_argv.append("-tensor_log")
    if is_print_progress_log:
        sys_argv.append("-progress_log")
    if rank_size:
        sys_argv += ["--rank_size", *[str(x) for x in rank_size]]
    if tp:
        sys_argv += ["--tp", *[str(x) for x in tp]]
    if pp:
        sys_argv += ["--pp", *[str(x) for x in pp]]
    if vpp:
        sys_argv += ["--vpp", *[str(x) for x in vpp]]
    if order:
        sys_argv += ["--order", *order]
    args = parser.parse_args(sys_argv)
    _graph_service_command(args)
    return output_path
