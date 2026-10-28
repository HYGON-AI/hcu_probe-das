# 修法B:hcu_probe FSDP statistics + consistent_check 的 shape squeeze 盲区

> **TL;DR:这个修正已合入 hcu_probe 主仓**(类 `VerlShapeErrChecker` + 列表 `VERL_STATISTICS_CHECKERS` + `calculator._add_algorithm` 的 `consistent_check` 分支)。**用含此修正的 hcu_probe 版本即可,不必手改**。本文档给「验证它在不在」+「老版本没有时怎么手动加」。

## 问题

FSDP 训练侧 dump 的张量是 `[1,S,H]`(HF eager `transformer_impl.py` 的 `unsqueeze(0)`),推理侧 vllm 是 `[S,H]`(packed)。hcu_probe 的 squeeze 容差检查器 `VerlShapeErrChecker`(去掉 =1 维 + FSDP 跳过 qkv/gate_up 聚合)**早期只挂在 `task=tensor` 模式**;`task=statistics`(本 skill 用的 L0 统计量)恒用严格 `ShapeErrChecker` → 训推 shape 字符串不等 → **几乎全判 shape error、可比配对≈0**。

## 验证它在不在(先跑这个)

```bash
docker exec <容器> python3 -c "from hcuprobe.core.compare.indicator_analysis.algorithm import VERL_STATISTICS_CHECKERS; from hcuprobe.core.compare.indicator_analysis.algorithm import VerlShapeErrChecker; print('修法B 已在 hcu_probe:OK')"
```
- 打印 `OK` → 你的 hcu_probe 已含修法B,**compare 直接能跑,跳过下面**(本次集群容器即此情况,compare 跑通 98 算子即证)。
- `ImportError` → hcu_probe 太旧,按下面手动加这 2 处(纯 .py 覆盖容器 dist-packages 即可,不重打 whl)。

## 老版本手动加(3 处,`hcu_probe/core/compare/indicator_analysis/`)

### 1. `algorithm.py`:确保有 `VerlShapeErrChecker` 类(squeeze 容差检查器)
若类已存在(grep `class VerlShapeErrChecker`)则跳过;否则它是个 `BaseAlgorithm` 子类,`run()` 里对两侧 shape 先 squeeze 掉 =1 维再比(FSDP 还跳过 qkv/gate_up 聚合 op)。完整实现见 hcu_probe 源码(本仓 / 已装包) `core/compare/indicator_analysis/algorithm.py`,grep `class VerlShapeErrChecker` 定位(不依赖行号)。

### 2. `algorithm.py`:加 `VERL_STATISTICS_CHECKERS` 列表(= STATISTICS_CHECKERS 把 `ShapeErrChecker` 换成 `VerlShapeErrChecker`)
```python
# 紧挨现有 STATISTICS_CHECKERS 定义后加:
VERL_STATISTICS_CHECKERS = [InfNanErrChecker, RelativeErrChecker, RequiresGradErrChecker, ParametersErrChecker,
                            DTypeErrChecker, VerlShapeErrChecker, RelativeWarnChecker]
```
(注:`VERL_TENSOR_CHECKERS` 通常已存在,做参照。)

### 3. `calculator.py`:`_add_algorithm` 的 STATISTICS 分支加 `consistent_check` 选它
```python
from .algorithm import (..., VERL_STATISTICS_CHECKERS)   # 顶部 import 补上

def _add_algorithm(self):
    if self.mode == CompareMode.STATISTICS.value:
        if self.consistent_check:                       # ← 新增这两行
            checkers = VERL_STATISTICS_CHECKERS          #   verl 训推一致用带 squeeze 的版本
        elif self.parallel_merge:
            checkers = STATISTICS_CHECKERS_PARALLEL_MERGE
        else:
            checkers = STATISTICS_CHECKERS
        for checker in checkers:
            self.add_algorithm(checker())
    ...
```

## 部署(老版本场景)

纯 Python 改动,把改好的 `algorithm.py` / `calculator.py` 同步进容器 `dist-packages/hcu_probe/core/compare/indicator_analysis/`(复用官方 `.so`,不重编/不重打 whl)。

## 另一前置:`Megatron-LM` 必须在 `PYTHONPATH`
`VerlShapeErrChecker` 链路依赖 megatron 工具,run 脚本/compare 已在 `PYTHONPATH` 加 `${VERL}/Megatron-LM`。缺则 compare 在 import 链深处报错。
