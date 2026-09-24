<div align="center">

<h1 style="text-align: center;">hcuprobe: 海光 HCU 训练精度调试工具</h1>

<h3 align="center">面向 PyTorch 分布式训练 / 推理的逐层 dump · 精度比对 · 训推一致性验证</h3>

<p align="center">
| <a href="docs/UPSTREAM_SYNC.md"><b>上游同步</b></a>
| <a href="docs/offline_analysis/"><b>离线分析</b></a>
| <a href="docs/infer/"><b>推理侧 dump</b></a>
| <a href="docs/FAQ.md"><b>FAQ</b></a>
|
</p>

[![version](https://img.shields.io/badge/version-26.2.0%2Bhcu.0.2.0-green)](docs/UPSTREAM_SYNC.md)
[![license](https://img.shields.io/badge/license-Mulan%20PSL%20v2-blue)](docs/UPSTREAM_SYNC.md#license-与署名)
[![upstream](https://img.shields.io/badge/upstream-msProbe%2026.2.0-brightgreen)](https://gitcode.com/Ascend/msprobe)

</div>

<div align="left">

## 🎯 适用范围

hcuprobe 通过 **PyTorch 的 module / API 钩子**采集精度数据,只对**有 PyTorch 层可挂钩**的栈生效:

| 状态 | 框架 / 栈 |
| --- | --- |
| ✅ **适用** | 基于 PyTorch 的分布式训练 / 推理 —— FSDP、Megatron-LM、verl、原生 PyTorch |
| ❌ **不适用** | 纯 C++ / CUDA 自研训练栈、非 PyTorch 框架(TensorFlow 等)—— 无可挂钩的 module / API |

## ℹ️ 简介

hcuprobe 是海光 HCU 平台的训练精度调试 Python whl 包,给在 HCU 上训练的团队提供一套「采集 → 比对 → 定界」的精度工具:训练时逐层 dump 张量统计量、监测梯度 / 优化器 / 参数,训练后离线做两端精度比对、溢出定位、API 精度预检,以及 RL 场景的训推一致性验证。

它基于华为 [`mindstudio-probe`(msProbe)](https://gitcode.com/Ascend/msprobe) `26.2.0` 适配:核心 dump / 比对内核与上游一致,HCU 侧自维护多框架训练接入的翻译层。命名空间 `msprobe.*` → `hcuprobe.*`,CLI 命令 `msprobe` → `hcuprobe`。版本对齐详见 **[上游同步说明](docs/UPSTREAM_SYNC.md)**。

- **逐层采集**: L0 模块级统计 / L1 API 级 / tensor 真实数据三粒度,`task=statistics` 免爆盘。
- **多框架接入**: hcu_megatron、原生 PyTorch、verl(FSDP / Megatron 后端)各有自包含接入 skill。
- **训推一致性**: RL 场景下训练侧前向 vs 推理侧(vLLM)前向逐层比对,定界精度差在哪侧哪层。
- **跨平台比对**: 采集与离线分析跨平台、纯 Python,可 HCU dump 对 NV / H20 dump。

## 🚀 快速开始

### 安装

```bash
# 方案 A:源码 build(本仓)
git clone https://github.com/HYGON-AI/hcu_probe-das
cd hcu_probe-das
pip wheel . --no-deps -w dist/
pip install dist/hcuprobe-*.whl --no-deps

# 方案 B:editable(开发用)
pip install -e .

# 方案 C:GitLab Releases 页下载 prebuilt whl 直接 pip install

# 离线分析 precheck 扩展(onnx / onnxruntime / skl2onnx)
pip install hcuprobe[precheck]
```

> **注意:** `torch` 不在依赖里,由用户在训练环境自备(HCU / NV / H20 各自的 torch 构建)。

### 训练侧 dump(最小片段)

```python
from hcuprobe.pytorch import PrecisionDebugger
dbg = PrecisionDebugger(config_path="config.json")   # task=statistics level=L0
dbg.start(model=model)  # 前向前
# <前向>
dbg.stop(); dbg.step()  # 前向后 + 步进
```

### 离线精度分析(dump 落盘后)

```bash
# L0 dump 后两端精度比对 → xlsx
hcuprobe compare -tp ./hcu_dump/step0 -gp ./bench_dump/step0 -o ./diff

# 整网首个 NaN/Inf 起点定位(训练发散时)
hcuprobe overflow_check -i ./dump/step_0 -o ./overflow_out

# API 级精度预检(HCU bf16 vs CPU fp64 双跑)
hcuprobe multi_acc_check -api_info ./l1_dump/step0/rank0/dump.json -o ./out
```

各工具详解见 **[离线分析文档](docs/offline_analysis/)**(compare / overflow / precheck / graph_viz / config_check / data2db 各篇);推理侧 dump(vLLM / SGLang)见 **[推理侧文档](docs/infer/)**。

## ⚙️ 能力

| 类 | 子能力 | 入口 |
| --- | --- | --- |
| **训练侧 dump** | L0 模块级张量统计 / L1 API 级张量 dump / tensor 真实数据 | `hcuprobe.pytorch.PrecisionDebugger` |
| **训练侧 monitor** | weight_grad / optimizer / param 三通道轻量监测 | `hcuprobe.core.monitor_v2.trainer.TrainerMonitorV2` |
| **离线分析 CLI** | `compare` / `acc_check` / `overflow_check` / `graph_visualize` / `parse` / `merge_result` / `config_check` 等 | `hcuprobe <subcmd>` |

## 🧩 已适配训练框架

接入操作见各 skill `.claude/skills/<skill 名>/`:

| 框架 · 后端 / 版本 | 验证场景 | 接入 skill |
| --- | --- | --- |
| **hcu_megatron** | 精度对比 | `hcuprobe-megatron-integration` |
| **原生 PyTorch** | 精度对比 | `hcuprobe-pytorch-integration` |
| **verl 0.7.1 · FSDP** | 训推一致性 | `hcuprobe-verl-fsdp-0.7-integration` |
| **verl 0.7.1 · Megatron** | 训推一致性 | `hcuprobe-verl-megatron-0.7-integration` |
| **verl 0.8.0 · FSDP** | 训推一致性 | `hcuprobe-verl-fsdp-0.8-integration` |
| **verl 0.8.0 · Megatron** | 训推一致性 | `hcuprobe-verl-megatron-0.8-integration` |

各框架接入的**代码改动点标注页**(插桩位置 / compare 逻辑 / 算法内部,渲染即懂)见 **[`docs/annotated/`](docs/annotated/)** —— skill 讲怎么动手接入,标注页讲改动点的位置与原理。自包含 HTML,下载到本地用浏览器打开 `index.html` 查看。

## 📂 项目结构

```
hcuprobe/
├── src/hcuprobe/
│   ├── core/                 # 与上游一致的内核:dump / monitor_v2 / compare
│   ├── pytorch/              # PrecisionDebugger 等 PyTorch 侧入口
│   ├── visualization/        # 构图可视化
│   ├── overflow_check/       # 溢出检测
│   ├── msaccucmp/            # 精度比对底层
│   ├── integrations/         # HCU 侧自维护:多框架训练接入翻译层
│   ├── tools/                # 离线分析工具实现
│   └── cli.py                # hcuprobe CLI 入口
├── docs/
│   ├── offline_analysis/     # 离线分析工具文档(compare / overflow / precheck / ...)
│   ├── infer/                # 推理侧 dump(vLLM / SGLang)
│   ├── annotated/            # 各框架接入的代码改动点标注页(渲染即懂 HTML)
│   ├── UPSTREAM_SYNC.md       # 上游版本同步说明
│   └── FAQ.md                # 常见问题
└── pyproject.toml
```

## 🔄 上游同步

hcuprobe 是 msProbe 的 HCU 适配 fork。当前对齐上游 `26.0.0`(commit `ffb0c39f`),内核子树 `core / pytorch / visualization / overflow_check / msaccucmp` 与上游一致,`integrations / tools` 为 HCU 侧自维护。**本仓未覆盖的上游功能**(如 `checkpoint_compare`、`response_anomaly`、accuracy_checker 细节等)可参考上游 user_guide,注意 `msprobe` → `hcuprobe` 命令映射、跳过 NPU / 昇腾专属能力。

**→ [上游同步说明 docs/UPSTREAM_SYNC.md](docs/UPSTREAM_SYNC.md)**

## ⚖️ 许可证

- **代码**: 沿用上游 msProbe 的[木兰宽松许可证 第 2 版(Mulan PSL v2)](https://gitcode.com/Ascend/msprobe)。
- **文档**: 上游 `docs/` 为 CC-BY 4.0;从上游文档复制内容须保留署名。

详见 **[上游同步说明 · License 与署名](docs/UPSTREAM_SYNC.md#license-与署名)**。
