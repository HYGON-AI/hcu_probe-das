"""verl 训推一致性比对(HCU 侧自维护,不修改与上游 msprobe 同步的 core/)。

在上游 ``pt_compare(..., consistent_check=True, backend=...)`` 之外补两处 HCU 场景必需的处理:

1. statistics 模式使用带 squeeze 容差的 ``VerlShapeErrChecker``。
   上游只在 tensor 模式的 consistent_check 下使用它;statistics 模式仍用严格的 ShapeErrChecker,
   FSDP 训练侧 ``[1, S, H]`` 与推理侧 vLLM packed ``[S, H]`` 会被全部判为 shape error。
   这里仅在本次调用期间替换比对规则,调用结束即恢复。

2. 比对前剔除 ``Triton.*`` kernel 级条目。
   推理侧 vllm-hcu 会采到 ``kernel_unified_attention_2d`` / ``reshape_and_cache_kernel_flash`` 等 Triton kernel,
   训练侧有 ``cross_entropy_fwd_kernel``;它们没有 layer / module 语义,会在 consistent match 中互相误配,
   导致 ``dual_monotonic_sort`` 抛出 "Conflicting comparison data"。训推一致只比 nn.Module 层级。

上游若合入等价修复,可直接改回调用 ``pt_compare``。
"""

import contextlib
import json
import os
import shutil
import tempfile

from hcuprobe.core.compare.indicator_analysis import algorithm as _algorithm
from hcuprobe.core.compare.indicator_analysis import calculator as _calculator
from hcuprobe.core.compare.indicator_analysis.utils import CompareMode
from hcuprobe.pytorch.compare.pt_compare import pt_compare

_TRITON_PREFIX = "Triton."
_SIDE_FILES = ("stack.json", "construct.json")

VERL_STATISTICS_CHECKERS = getattr(_algorithm, "VERL_STATISTICS_CHECKERS", None) or [
    _algorithm.InfNanErrChecker,
    _algorithm.RelativeErrChecker,
    _algorithm.RequiresGradErrChecker,
    _algorithm.ParametersErrChecker,
    _algorithm.DTypeErrChecker,
    _algorithm.VerlShapeErrChecker,
    _algorithm.RelativeWarnChecker,
]


@contextlib.contextmanager
def _verl_statistics_checkers():
    cls = _calculator.ApiIndicatorCalculator
    original = cls._add_algorithm

    def _add_algorithm(self):
        if self.consistent_check and self.mode == CompareMode.STATISTICS.value:
            for checker in VERL_STATISTICS_CHECKERS:
                self.add_algorithm(checker())
        else:
            original(self)

    cls._add_algorithm = _add_algorithm
    try:
        yield
    finally:
        cls._add_algorithm = original


def _without_triton(dump_json, work_dir):
    """返回剔除 Triton.* 条目后的 dump.json 路径;无 Triton 条目时直接返回原路径。"""
    with open(dump_json) as f:
        dump = json.load(f)
    data = dump.get("data", {})
    kept = {k: v for k, v in data.items() if not k.startswith(_TRITON_PREFIX)}
    if len(kept) == len(data):
        return dump_json
    os.makedirs(work_dir, exist_ok=True)
    dump["data"] = kept
    out = os.path.join(work_dir, "dump.json")
    with open(out, "w") as f:
        json.dump(dump, f)
    src_dir = os.path.dirname(os.path.abspath(dump_json))
    for name in _SIDE_FILES:
        if os.path.isfile(os.path.join(src_dir, name)):
            shutil.copy(os.path.join(src_dir, name), os.path.join(work_dir, name))
    return out


def consistent_compare(train_dump, infer_dump, output_path, backend, skip_triton=True, **kwargs):
    """训练侧 dump.json(npu_path)对推理侧 dump.json(bench_path)做逐层训推一致性比对。

    Args:
        train_dump: 训练侧 ``dump.json`` 路径(verl ``dump_actor/stepN/rankM/dump.json``)。
        infer_dump: 推理侧 ``dump.json`` 路径(verl ``dump_generate/<pid>/stepN/rankM/dump.json``)。
        output_path: 比对结果输出目录。
        backend: ``"fsdp"`` 或 ``"megatron"``,决定训推命名映射(FSDP_MODULE_MAP / MEGATRON_MODULE_MAP)。
        skip_triton: 是否剔除 ``Triton.*`` kernel 级条目(默认 True)。
        **kwargs: 透传给上游 ``pt_compare``。
    """
    with tempfile.TemporaryDirectory(prefix="hcuprobe_verl_cmp_") as tmp:
        if skip_triton:
            train_dump = _without_triton(train_dump, os.path.join(tmp, "train"))
            infer_dump = _without_triton(infer_dump, os.path.join(tmp, "infer"))
        with _verl_statistics_checkers():
            return pt_compare(
                {"npu_path": train_dump, "bench_path": infer_dump},
                output_path,
                consistent_check=True,
                backend=backend,
                **kwargs,
            )
