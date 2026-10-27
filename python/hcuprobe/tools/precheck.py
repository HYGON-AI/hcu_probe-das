"""API 精度预检三件套(acc_check / multi_acc_check / api_precision_compare)。
"""

import argparse
from typing import Optional, Sequence

from hcuprobe.core.acc_check.acc_check_cli import acc_check_cli, multi_acc_check_cli


def acc_check(
    *,
    api_info: str,
    output_path: str,
    jit_compile: bool = False,
) -> str:
    """单进程 API 精度预检(随机数据 vs CPU fp64 标杆)。"""
    argv = ["-api_info", api_info, "-o", output_path]
    if jit_compile:
        argv += ["--jit_compile"]
    acc_check_cli(argv)
    return output_path


def multi_acc_check(
    *,
    api_info: str,
    output_path: str,
    processes: int = 8,
    devices: Optional[Sequence[int]] = None,
) -> str:
    """多进程多卡并行 API 精度预检。"""
    argv = ["-api_info", api_info, "-o", output_path, "-n", str(processes)]
    if devices:
        argv += ["-d", *[str(d) for d in devices]]
    multi_acc_check_cli(argv)
    return output_path


def api_precision_compare(
    *,
    hcu_csv: str,
    gpu_csv: str,
    output_path: str,
) -> str:
    """对比两端 acc_check 输出 csv,自动 diff 出"HCU 独有劣化 API"等。"""
    from hcuprobe.pytorch.api_accuracy_checker.compare.api_precision_compare import (
        _api_precision_compare_command,
    )

    args = argparse.Namespace(
        hcu_csv_path=hcu_csv, gpu_csv_path=gpu_csv, out_path=output_path
    )
    _api_precision_compare_command(args)
    return output_path
