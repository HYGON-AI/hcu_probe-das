"""脚本/环境配置采集与比对(环境变量、三方库版本、训练超参、权重、数据集)。"""

from hcuprobe.core.config_check.config_check_cli import compare as _upstream_compare
from hcuprobe.core.config_check.config_check_cli import pack as _upstream_pack


def collect(*, shell_path: str, output_path: str) -> str:
    """采集当前训练环境快照到 zip。shell_path 指向训练启动脚本。"""
    _upstream_pack(shell_path, output_path)
    return output_path


def compare(*, bench_zip: str, target_zip: str, output_path: str) -> str:
    """比对两个环境快照 zip,输出差异报告。"""
    _upstream_compare(bench_zip, target_zip, output_path)
    return output_path
