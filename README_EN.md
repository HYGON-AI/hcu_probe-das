<div align="center">

<h1 style="text-align: center;">hcuprobe: Training Accuracy Debugging Tool for HYGON HCU</h1>

<h3 align="center">Layer-by-layer dump · accuracy comparison · training–inference consistency verification for PyTorch distributed training / inference</h3>

<p align="center">
| <a href="docs/UPSTREAM_SYNC.md"><b>Upstream Sync</b></a>
| <a href="docs/offline_analysis/"><b>Offline Analysis</b></a>
| <a href="docs/infer/"><b>Inference-side Dump</b></a>
| <a href="docs/FAQ.md"><b>FAQ</b></a>
|
</p>

[![version](https://img.shields.io/badge/version-26.2.0%2Bhcu.0.2.0-green)](docs/UPSTREAM_SYNC.md)
[![license](https://img.shields.io/badge/license-Mulan%20PSL%20v2-blue)](docs/UPSTREAM_SYNC.md#license-与署名)
[![upstream](https://img.shields.io/badge/upstream-msProbe%2026.2.0-brightgreen)](https://gitcode.com/Ascend/msprobe)

</div>

<div align="left">

[简体中文](README.md) | English

## ℹ️ Introduction

hcuprobe is the training-accuracy debugging Python wheel for the HYGON HCU platform. It gives teams training on HCU a complete "collect → compare → localize" accuracy toolchain: during training it dumps per-layer tensor statistics and monitors gradients / optimizer states / parameters; after training it performs offline two-side accuracy comparison, overflow localization, and API accuracy pre-check, as well as training–inference consistency verification for RL scenarios.

It is adapted from Huawei [`mindstudio-probe` (msProbe)](https://gitcode.com/Ascend/msprobe) `26.2.0`: the core dump / comparison kernels stay aligned with upstream, while the HCU side maintains its own translation layer for multi-framework training integration. Namespace `msprobe.*` → `hcuprobe.*`, CLI command `msprobe` → `hcuprobe`. For version alignment, see the **[Upstream Sync notes](docs/UPSTREAM_SYNC.md)**.

## ⚙️ Features

hcuProbe provides three core capabilities around model accuracy debugging, covering the full "data collection → accuracy comparison → visualization" workflow:

| Core capability | Description | Quick entry |
|---|---|---|
| Data collection | Driven by a config.json file; collects accuracy data during model training/inference in one step | [PyTorch data collection](.claude/pytorch-integration/dump/SKILL.md) |
| Accuracy comparison | Compares the collected accuracy data to locate the network layers/operators where accuracy diverges | [PyTorch accuracy comparison](.claude/pytorch-integration/accuracy_compare/SKILL.md) |
| Visualization | Parses dump data to reconstruct the model graph structure and locates accuracy differences with hierarchical visualization | [Hierarchical graph comparison](.claude/pytorch-integration//visualization/SKILL.md) |

> The table above lists the most common PyTorch entry points. For more frameworks and usage scenarios, see the [hcuProbe User Guide](.claude/README.md).

## 🚀 Quick Start

For a one-stop walkthrough of the complete workflow — data collection, HCU/GPU accuracy comparison, and hierarchical graph comparison — see the [hcuProbe Quick Start](.claude/quick_start/SKILL.md).


## 📦 Installation

```bash
# Option A: build from source (this repo)
git clone https://github.com/HYGON-AI/hcu_probe-das
cd hcu_probe-das
pip wheel . --no-deps -w dist/
pip install dist/hcuprobe-*.whl --no-deps

# Option B: editable install (for development)
pip install -e .

# Option C: download a prebuilt whl from the GitLab Releases page and pip install it

# precheck extras for offline analysis (onnx / onnxruntime / skl2onnx)
pip install hcuprobe[precheck]
```

> **Note:** `torch` is not declared as a dependency; provide it yourself in the training environment (the respective torch builds for HCU / NV / H20).
## 📘 User Guide

For detailed usage of the tool, see the [hcuProbe User Guide](.claude/README.md).


## 🔄 Upstream Sync

hcuprobe is an HCU adaptation fork of msProbe. The kernel subtrees `core / pytorch / visualization / overflow_check / msaccucmp` stay aligned with upstream, while `integrations / tools` are maintained on the HCU side. **For upstream features not covered in this repo** (e.g. `checkpoint_compare`, `response_anomaly`, accuracy_checker details), refer to the upstream user_guide — mind the `msprobe` → `hcuprobe` command mapping and skip the NPU / Ascend-specific capabilities.

**Baseline commit**: upstream branch `26.2.0`, commit `76f701cd74b765b98e9652fa0730b377daa25559` (2026-09-23); the corresponding commit in this repository's history is `87dec7eb1602b0b123c312eab0840b1d5ab1fd89` (hash differs due to history cleanup; the tree content is identical to the upstream baseline). All commits after the baseline are HYGON-side modifications.

**→ [Upstream Sync notes: docs/UPSTREAM_SYNC.md](docs/UPSTREAM_SYNC.md)**

## ⚖️ License

- **Code**: follows upstream msProbe's [Mulan Permissive Software License, Version 2 (Mulan PSL v2)](LICENSE).
- **Documentation**: upstream `docs/` is CC-BY 4.0 ([docs/LICENSE](docs/LICENSE)); content copied from upstream documentation must retain attribution.
- **Modification attribution**: HCU adaptation changes and additions on top of the upstream baseline are Hygon Information Technology Co., Ltd., likewise released under Mulan PSL v2.
- **Third-party notices**: for the sources, pinned versions, copyrights, and licenses of third-party code contained in this repo or referenced at build time, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

See **[Upstream Sync notes · License and attribution](docs/UPSTREAM_SYNC.md#license-与署名)** for details.
