# compare — dump 数据精度比对

对比两端 dump 数据(target vs golden),量化 MeanRelativeErr / 双千指标 / md5 一致性等,输出 `compare_result_{ts}.xlsx` 和 `advisor_{ts}.txt`。**纯数据分析侧,无硬件依赖,跨平台开箱即用**。

## 典型场景

- HCU dump ↔ H20 / A100 dump 跨硬件比对
- HCU 不同 DTK / driver 版本前后回归
- 同硬件不同 patch(reuse-fp32-param / EDGC 开关)对比

## Python API

```python
from hcuprobe.tools import compare

compare.run(
    target_path="./hcu_dump/step0",
    golden_path="./h20_dump/step0",
    output_path="./diff",
)
```

完整签名:

```python
def run(
    *,
    target_path: str,
    golden_path: str,
    output_path: str = "./output",
    mode: str = "auto",            # 或 "torchair"(MindSpore 场景,HCU 训练无需)
    fuzzy_match: bool = False,     # 对 api 名做模糊匹配
    layer_mapping: str = None,     # 手动点点匹配文件路径
    api_mapping: str = None,
    data_mapping: str = None,
    cell_mapping: str = None,
    diff_analyze: bool = False,
    is_print_compare_log: bool = False,
) -> str
```

## CLI

```bash
hcuprobe compare \
    -tp ./hcu_dump/step0 \
    -gp ./h20_dump/step0 \
    -o ./diff
```

## 输出产物

| 文件 | 内容 |
|---|---|
| `compare_result_{ts}.xlsx` | 每个算子的 npu / bench 统计量 + 相对误差 + 双千指标 |
| `advisor_{ts}.txt` | 专家建议(优先看 Error/Warning 行 + MeanRelativeErr 异常算子) |

## HCU 上的实证(L0 + L1 双实证)

| 维度 | L0 statistics | L1 statistics |
|---|---|---|
| 实证日期 | 2026-05-12 / 2026-05-13(两次)| 2026-05-13 |
| 节点 / 容器 | scnet docker `hcu_probe_lab` | 同 |
| 输入 dump | 8 卡 × 3 step × 33 MB dump.json | 8 卡 × 2 step,~52 MB dump.json/rank |
| compare 时长 | **2m55s** | **4m50s**(L1 API 数多约 1.5x) |
| 产物 | 8 × ~24 MB xlsx,共 188 MB | **8 × ~35 MB xlsx,共 278 MB** |
| log 末 | `msprobe compare ends successfully.` × 8 | 同 ✅ |

`compare` 内部 `auto` mode 不依赖 op_name 前缀,**L0(`Module.*` 前缀)和 L1(`Functional.*` / `Tensor.*` / `Torch.*` 前缀)都直接支持**,无需额外配置。

## HCU 上的关注点

1. **dump 数据格式跨平台一致**:dump 出来的 `dump.json` / `construct.json` / `stack.json` 在 HCU 和 NV 平台格式相同,可直接比对
2. **construct.json 模块名差异**:HCU(hcu_megatron patch 后)和 NV 原生 Megatron 的模块类名可能不一致;不匹配时用 `layer_mapping` 手动点点匹配
3. **常见 HCU 特异点关注列**:FlashAttention 输出 / GroupedGEMM / 通信算子规约(RCCL vs NCCL)/ FP32↔BF16 cast 边界
4. **容器内需补依赖**:`compare` import 链通过 `pytorch/monitor/data_writers.py` 拉 `torch.utils.tensorboard.SummaryWriter` + `openpyxl`(写 xlsx)。容器无外网时从主机 conda env cp:
   ```bash
   HOST_SP=~/miniforge3/envs/megatron_train/lib/python3.10/site-packages
   docker exec <container> bash -c "for m in tensorboard tensorboard_data_server openpyxl et_xmlfile; do cp -r $HOST_SP/$m /usr/local/lib/python3.10/dist-packages/; done"
   ```

## 判定标准速查

| 指标 | Pass 阈值 |
|---|---|
| 余弦相似度 | > 0.99 |
| 双百指标(相对误差 > 1% 的元素占比) | < 1% |
| 双千指标(相对误差 > 0.1% 的元素占比) | < 0.1% |
| MeanRelativeErr | 算子相关,无统一阈值 |

bf16 dtype 下 sigmoid / 乘法 / rsub 等算子双千经常未达标(共性现象,非硬件问题),给 Warning 不 Error。

## 完整能力参考(上游 PyTorch 文档)

hcu_probe 的 compare 实现完全沿用上游 mindstudio-probe v26.0.0-alpha.1 的 PyTorch 路径代码。以下细节本文档没展开,需要时查上游 PyTorch 场景文档:

- 5 种 compare mode(`auto` / `offline_data` / `torchair` / `offline_model` / `atb`)详解 — **hcu_probe 只用 `auto` mode**,其余 4 种是推理 / TorchAir 静态图场景,whl 内 try/except 兜底
- 跨套件比对(DeepSpeed vs Megatron / MindSpeed vs LLamaFactory)的 mapping yaml 配置实例
- 自定义映射文件 4 种(`layer_mapping` / `api_mapping` / `cell_mapping` / `data_mapping`)各自的格式与用法
- `compare_result_{ts}.xlsx` 完整字段说明(NPU/HCU vs Bench 各列、相对误差、双百/双千、Result/Pass/Warning/Error 判定)
- `advisor_{ts}.txt` 专家建议系统的产出格式
- 模糊匹配(`-fm`)的匹配算法细节
- 差异分析(`-da` / `--diff_analyze`)的输出
- 三种 dump 类型(`statistics` / `tensor` / `md5`)比对差异
- verl 训推一致性比对场景(prefill vs decode shape 对齐流程)

上游文档地址:

```
https://gitcode.com/Ascend/msprobe/blob/26.0.0-alpha.1/docs/zh/accuracy_compare/pytorch_accuracy_compare_instruct.md
```

> ⚠️ **不要参考同目录下其他文档** —— `mindspore_accuracy_compare_instruct.md`(MindSpore 训练)/ `atb_data_compare_instruct.md`(ATB 推理)/ `infer_compare_offline_model_instruct.md`(离线模型推理)/ `torchair_compare_instruct.md`(静态图,主要推理用)都不是 hcu_probe 的训练场景,字段集 / 工作机制都不一样。
