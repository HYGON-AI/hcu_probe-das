"""整网首个溢出节点分析(基于 dump 数据扫描 NaN/Inf,跨 rank 关联通信节点)。"""

import argparse
import os

from hcuprobe.overflow_check.analyzer import _run_overflow_check


def run(
    *,
    input_path: str,
    output_path: str = "./overflow_output",
) -> str:
    """扫描 dump_path 下所有 rank,输出 anomaly_analyze_{ns}.json。无 anomaly 时 output_path 仍存在(空目录)。"""
    os.makedirs(output_path, exist_ok=True)
    args = argparse.Namespace(input_path=input_path, output_path=output_path)
    _run_overflow_check(args)
    return output_path
