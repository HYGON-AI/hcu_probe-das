"""合并多卡比对结果。"""

import argparse
from typing import Optional

from hcuprobe.core.compare.merge_result.merge_result_cli import merge_result_cli


def run(
    *,
    input_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
) -> str:
    """聚合 input_dir 下多份 compare 结果,落盘到 output_dir。"""
    args = argparse.Namespace(
        input_dir=input_dir, output_dir=output_dir, config_path=config_path
    )
    merge_result_cli(args)
    return output_dir
