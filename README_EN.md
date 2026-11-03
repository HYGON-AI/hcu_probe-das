 <div align="center">

# hcuprobe: Hygon HCU Training Accuracy Debugging Tool

### Layer-by-layer dump · Accuracy comparison · Training–inference consistency validation for PyTorch distributed training and inference

| [Upstream Synchronization](docs_hcu/UPSTREAM_SYNC.md) | [Quick Start](docs_hcu/zh/quick_start/pytorch_quick_start.md) | [User Guide](docs_hcu/zh/user_guide/README.md) | [FAQ](docs_hcu/zh/support/faq.md) |

</div>

[![version](https://img.shields.io/badge/version-26.2.0%2Bhcu.0.2.0-green)](docs_hcu/UPSTREAM_SYNC.md)
[![license](https://img.shields.io/badge/license-Mulan%20PSL%20v2-blue)](docs_hcu/UPSTREAM_SYNC.md#license-and-attribution)
[![upstream](https://img.shields.io/badge/upstream-msProbe%2026.2.0-brightgreen)](https://gitcode.com/Ascend/msprobe)

<div align="left">

English | [简体中文](README.md)

## ℹ️ Introduction

**hcuprobe** is a Python wheel package for training accuracy debugging on the Hygon HCU platform. It provides teams training on HCU with a set of tools for accuracy debugging through a complete *collect → compare → localize* workflow.

Its capabilities include layer-by-layer tensor statistics dumping during training, monitoring of gradients, optimizers, and model parameters, offline accuracy comparison between two execution environments, overflow detection and localization, API accuracy prechecks, and training–inference consistency validation for reinforcement learning (RL) scenarios.

![hcuprobe Architecture](docs_hcu/zh/figures/readme/hcuprobe_architecture.jpg)

hcuProbe adopts a layered architecture in which Python core modules work together with native C/C++ extensions to provide accuracy debugging capabilities for training and inference frameworks such as PyTorch, verl, and Megatron-LM. The project covers the complete workflow, including data collection, accuracy analysis, anomaly detection, result visualization, and report generation, and provides corresponding build, test, and delivery tools.

hcuProbe is adapted from Huawei's [`mindstudio-probe` (msProbe)](https://gitcode.com/Ascend/msprobe) version `26.2.0`. The core dump and accuracy comparison implementations remain consistent with upstream, while the HCU-specific translation layer for integrating multiple training frameworks is maintained independently.

The namespace is mapped from `msprobe.*` to `hcuprobe.*`, and the CLI command is changed from `msprobe` to `hcuprobe`. For version alignment details, see **[Upstream Synchronization Guide](docs_hcu/UPSTREAM_SYNC.md)**.

## ⚙️ Features

hcuProbe provides three core capabilities for model accuracy debugging, covering the complete *data collection → accuracy comparison → visualization* workflow.

| Core Feature | Description | Quick Access |
|---|---|---|
| Data Collection | Collect accuracy data during model training or inference through a single configuration file, `config.json`. | [PyTorch Data Dump](docs_hcu/zh/user_guide/dump/pytorch_data_dump_instruct.md) |
| Accuracy Comparison | Compare collected accuracy data to identify network layers and operators where accuracy discrepancies occur. | [PyTorch Accuracy Comparison](docs_hcu/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| Visualization | Reconstruct the model graph from dump data and use hierarchical visualization to locate accuracy discrepancies. | [Hierarchical Graph Visualization and Comparison](docs_hcu/zh/user_guide/accuracy_compare/pytorch_visualization_instruct.md) |

> The table above lists the most commonly used entry points for PyTorch. For additional frameworks and use cases, see the [hcuProbe User Guide](docs_hcu/zh/user_guide/README.md).

## 🚀 Quick Start

For an end-to-end walkthrough of data collection, HCU/GPU accuracy comparison, and hierarchical graph visualization and comparison, see the [hcuProbe Quick Start Guide](docs_hcu/zh/quick_start/pytorch_quick_start.md).

Framework quick starts (verified on official images, with pinned image / code versions; in Chinese):

- [Megatron Quick Start](docs_hcu/zh/quick_start/megatron_quick_start.md): integrate hcuprobe into Megatron-LM-das for L0 data dump and Monitor.
- [verl Training–Inference Consistency Quick Start](docs_hcu/zh/quick_start/verl_quick_start.md): layer-by-layer comparison for verl-das (verl 0.8.0) with FSDP / Megatron backends.

## 📦 Installation

```bash
# Option A: Build from source in this repository
git clone https://github.com/HYGON-AI/hcu_probe-das.git
cd hcu_probe-das
pip wheel . --no-deps -w dist/
pip install dist/hcu_probe-*.whl --no-deps   # wheel file uses the dist name hcu-probe; import name is hcuprobe

# Option B: Editable installation for development
pip install -e .

# Option C: Download a prebuilt wheel from the GitLab Releases page
# and install it directly with pip

# Install optional dependencies for offline analysis prechecks
# (ONNX / ONNX Runtime / skl2onnx)
pip install "$(ls dist/hcu_probe-*.whl)[precheck]"   # dist name is hcu-probe; from an index: pip install "hcu-probe[precheck]"
```

> **Note:** `torch` is not included in the package dependencies. Users must provide a compatible PyTorch build for their target training environment (HCU, NVIDIA, or H20).

## 📘 User Guide

For detailed instructions on using the tool, see the [hcuProbe User Guide](docs_hcu/zh/user_guide/README.md).

## ❓ FAQ

For common issues and their solutions, see the [FAQ](docs_hcu/zh/support/faq.md).

## 🔄 Upstream Synchronization

hcuprobe is the HCU adaptation of msProbe. The `core`, `pytorch`, `visualization`, `overflow_check`, and `msaccucmp` subtrees remain consistent with upstream, while `integrations` and `tools` are maintained independently for HCU-specific requirements.

For upstream features not covered by this repository, such as `checkpoint_compare`, `response_anomaly`, and certain `accuracy_checker` details, refer to the upstream user guide. Note that command names must be mapped from `msprobe` to `hcuprobe`, and NPU- or Ascend-specific capabilities should be skipped where they do not apply.

**→ [Upstream Synchronization Guide](docs_hcu/UPSTREAM_SYNC.md)**

## ⚖️ License

- **Code:** The code follows the upstream msProbe [Mulan Permissive Software License, Version 2 (Mulan PSL v2)](LICENSE).
- **Documentation:** The upstream `docs/` directory is licensed under CC-BY 4.0 ([docs/LICENSE](docs/LICENSE)). Attribution must be retained when reusing content from upstream documentation.
- **Modifications:** HCU adaptation changes and newly added content based on the upstream baseline are maintained by Hygon Information Technology Co., Ltd. and are also released under Mulan PSL v2.
- **Third-party notices:** Sources, pinned versions, copyrights, and licenses of third-party code included in the repository or referenced during the build process are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

For details, see **[Upstream Synchronization Guide — License and Attribution](docs_hcu/UPSTREAM_SYNC.md#license-and-attribution)**.

</div>