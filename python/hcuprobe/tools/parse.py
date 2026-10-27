"""dump 数据解析(从二进制 dump 提取人类可读结构)。"""

import argparse
from typing import Optional

from hcuprobe.core.parse.parse_cli import parse_cli


def run(
    *,
    dump_path: str,
    output_path: str,
    parse_type: str = "npy",
) -> str:
    """解析 dump_path 下数据,输出到 output_path。parse_type 合法值: 'npy' / 'pt'。"""
    if parse_type not in ("npy", "pt"):
        raise ValueError(f"parse_type must be 'npy' or 'pt', got {parse_type!r}")
    args = argparse.Namespace(
        dump_path=dump_path, output_path=output_path, parse_type=parse_type
    )
    parse_cli(args)
    return output_path
