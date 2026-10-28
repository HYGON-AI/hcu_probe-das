---
name: hcu_probe-megatron-integration
version: 2.0.0
description: 在 hcu_megatron(Megatron-LM-das,海光 HCU 版 Megatron-LM fork)训练中接入 hcuprobe:4 个 Feature 薄壳 + ADAPTOR_FEATURES 注册,训练命令加 --hcu-probe-config / --hcu-probe-monitor-config 即激活 L0 / L1 采集与三通道 monitor,不改训练脚本;配套离线分析(hcuprobe compare / overflow_check / graph_visualize / multi_acc_check)。当用户问"如何接入 hcuprobe 到 hcu_megatron / Megatron-LM-das、怎么写 feature / wrapper、L1 报 Only support fusion of gelu and swiglu、dump/monitor 只有 initialized 没数据、risk_level invalid、patch_optimizer_step、TORCHDYNAMO_DISABLE、megatron monitor 三通道"时使用。完整流程见 docs_hcu/zh/quick_start/megatron_quick_start.md。
---

# hcu_megatron(Megatron-LM-das)接入 hcuprobe

hcu_megatron 用 `hcu_megatron/features_manager/__init__.py` 里的 **`ADAPTOR_FEATURES`** 列表以 monkey-patch 扩展 Megatron:启动时遍历每个 Feature 的 `register_args()` / `register_patches()`。hcuprobe 提供两个 Feature,注册后新增命令行参数,并把采集逻辑挂到 `train` / `train_step` / `setup_model_and_optimizer`。业务逻辑在 whl 的 `hcuprobe.integrations.megatron.{dump,monitor}`,薄壳只做"注册参数 + 注册 patch + 转调"。

完整可复制流程(镜像、commit、数据、命令、实测输出):**`docs_hcu/zh/quick_start/megatron_quick_start.md`**。

## 验证状态(2026-10,BW 8 卡)

环境:`harbor.sourcefind.cn:5443/hcu/admin/base/custom:megatron-0.18.2-ubuntu22.04-dtk2604-py3.10-20260908-0350` + `HYGON-AI/Megatron-LM-das@52b4450`(core_v0.18.2) + Qwen3-0.6B(官方 `examples/qwen3/train_qwen3_0.6B.sh`,TP4)。

| 模式 / 工具 | 结果 |
|---|---|
| baseline | ✅ loss 11.95→11.39(10 步) |
| L0 | ✅ 采集不改变 loss;GBS=256 时每个采集 step 约 15 min、单 rank dump.json 约 120 MB |
| L1 | ✅ 需下文 yaml 修改;未修改时实测 `Only support fusion of gelu and swiglu` 后崩溃 |
| monitor | ✅ 8 rank × 3 通道 × 每步 CSV |
| combined(L0 + monitor) | ✅ |
| `hcuprobe compare` | ✅ 两次同配置 L0:前向逐位一致,差异仅在反向梯度 |
| `hcuprobe overflow_check` | ✅ 注入 NaN 后定位到起点层 |
| `hcuprobe graph_visualize` | ✅ 单图 / 双图 `.vis.db` |
| `hcuprobe multi_acc_check` | ✅ L1 输入 6709 API:error 0,warning 为 bf16 舍入 |
| 未验证 | 多机;MoE / EP 模型(L1 需额外注释 `distributed:` 段,见 runbook) |

## 接入(一次性)

1. 把 `scripts/` 下 4 个文件拷到 `hcu_megatron/features_manager/probe/`(它们 `from ..feature import AbstractFeature`):

| 文件 | 作用 |
|---|---|
| `hcu_probe_feature.py` | 注册 `--hcu-probe-config`、`--hcu-probe-seed`;dump wrapper 挂 `megatron.training.training.train` 与 `train_step` |
| `hcu_probe_wrapper.py` | 转调 `dump.on_train_start / on_step_start / on_step_end / on_train_end` |
| `monitor_feature.py` | 注册 `--hcu-probe-monitor-config`;monitor wrapper 挂 `setup_model_and_optimizer`、`train`、`train_step` |
| `monitor_wrapper.py` | 转调 `monitor.on_setup_done / on_step_end / on_train_end` |

2. 在 `hcu_megatron/features_manager/__init__.py` 注册(**只拷文件不注册 = 静默无效**):

```python
from .probe.hcu_probe_feature import HcuProbeFeature
from .probe.monitor_feature import MonitorFeature
ADAPTOR_FEATURES = [..., HcuProbeFeature(), MonitorFeature()]
```

3. 检查:`PYTHONPATH=<M>:<M>/3rdparty/Megatron-LM python3 -c "from hcu_megatron.features_manager import ADAPTOR_FEATURES; print([type(f).__name__ for f in ADAPTOR_FEATURES][-2:])"` → `['HcuProbeFeature', 'MonitorFeature']`。

**`train_step` 挂点**:Megatron-LM-das 的 `MegatronBasicFeature` 把 megatron 的 `train` 换成 `hcu_megatron.training.training.train`,它调用的是**本模块**的 `train_step`。因此 `train_step` 的 wrapper 必须挂在 `hcu_megatron.training.training.train_step`,否则 `train` / `setup` 钩子生效但每步钩子永不触发 → 日志只有 `initialized`、没有任何数据。两个 feature 文件中的 `_train_step_target()` 已自动选择(无该模块的旧 fork 回退到 `megatron.training.training.train_step`)。

## 使用

在训练脚本的 `pretrain_gpt.py` 参数中追加(官方 `train_qwen3_*.sh` 可加在 `${INITIALIZATION_ARGS[@]}` 后):

```bash
--hcu-probe-config <cfg_L0_or_L1.json> --hcu-probe-seed 1234      # L0 / L1 采集
--hcu-probe-monitor-config <cfg_monitor.json>                     # monitor(可与上面同时传 = combined)
```

| 配置模板(`config/`) | 要点 |
|---|---|
| `hcu_probe_config_statistics_L0.json` | `task=statistics`、`level=L0`;改 `dump_path`;大 GBS 时 `step` 只写 `[0]` |
| `hcu_probe_config_statistics_L1.json` | `level=L1`;需下文 yaml 修改 + `TORCHDYNAMO_DISABLE=1`;建议调小 GBS |
| `monitor_v2_config_full_probe.json` | 必须 `"patch_optimizer_step": false`(Megatron 嵌套优化器让自动 patch 落空,由 wrapper 每步驱动);只开 weight_grad / optimizer / param,**不开 module 通道**(与 `deallocate_pipeline_outputs` 冲突);改 `output_dir` |

**确认真正生效**(只有 `wrapper applied` 不算):日志出现 `[hcu_probe.dump] PrecisionDebugger initialized` / `[hcu_probe.monitor] TrainerMonitorV2 initialized and started`,且产物目录有 `step<N>/rank<M>/dump.json` 或 `rank_<r>/*.csv`。

## L1 前置:修改 `support_wrap_ops.yaml`

L1 会替换 `F.silu`,Megatron MLP 的 `activation_func is F.silu` 判断失败 → `Only support fusion of gelu and swiglu`。注释**已安装** hcuprobe 的 `pytorch/dump/api_dump/support_wrap_ops.yaml` 中 **`functional:` 区**的 `- gelu`、`- silu` 两行(其他区同名条目不动;只改安装位置,不改源码仓);用完还原。命令见 quick_start 2.7 节。副作用:L1 数据中没有 `Functional.silu/gelu`。非 MoE 模型只改这两行;MoE / EP 模型还需注释 `distributed:` 段(runbook 坑 3,未在新环境验证)。

## 离线分析

| 命令 | 用途 |
|---|---|
| `hcuprobe compare -tp <待测 step 目录> -gp <标杆 step 目录> -o <out>` | 两份 dump 逐 rank 比对,输出 `compare_result_rank<N>_*.csv` |
| `hcuprobe overflow_check -i <step 目录> -o <out>` | 定位首个"输入正常、输出 NaN/Inf"的节点;无异常时不生成文件 |
| `hcuprobe graph_visualize -tp <rank 目录> [-gp <rank 目录>] -o <out>` | 分级可视化 `.vis.db` |
| `hcuprobe multi_acc_check -api_info <L1 dump.json> -o <out> -d 0 1 ... 7` | API 精度预检(对 CPU fp64 标杆) |

分析方法与实测解读见 quick_start 2.9 节。

## 坑速查

| 现象 | 处理 |
|---|---|
| 训练正常但无 dump / CSV,只有 `initialized` | `train_step` 挂点(见上);或 Feature 未注册 |
| monitor 静默无输出 | 安装 tensorboard;`output_dir` 父目录可写 |
| L1 报 `Only support fusion of gelu and swiglu` | 修改 `support_wrap_ops.yaml` |
| L1 与 dynamo 冲突 | 训练脚本中 `export TORCHDYNAMO_DISABLE=1`(mpirun 下写进脚本,确保每个 rank 生效) |
| `risk_level` invalid | 必须大写 `ALL`/`FOCUS`/`CORE`,且只对 L1 生效 |
| L0 一个 step 写十几分钟 | 每个 micro batch 都记录全部 Module;只采 `step:[0]` 或调小 GBS |
| Megatron-LM-das 官方脚本本身的问题 | `train_qwen3_0.6B.sh` 的 `--use-qk-norm` 应为 `--qk-layernorm`;`requirements/env.sh` 网卡默认 `eth0`;py3.10 需把 `typing.override` 换成 `typing_extensions`(见 quick_start) |

## 参考

- `references/runbook.md`:6 个核心坑的原理与验证方法
- `references/legacy-record.md`:重构前正文(dcu_megatron + llama2-7B 时代的 launcher 解析、5 模式 PoC 记录)
