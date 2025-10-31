# 📖 msprobe 使用手册

![version](https://img.shields.io/badge/version-1.0.4-blueviolet)
![python](https://img.shields.io/badge/python-3.8|3.9|3.10-blue)
![platform](https://img.shields.io/badge/platform-Linux-yellow)

**msprobe** 是 MindStudio Training Tools 工具链下精度调试部分的工具包。主要包括精度预检、溢出检测和精度比对等功能，目前适配 [PyTorch](https://pytorch.org/) 和 [MindSpore](https://www.mindspore.cn/) 框架。这些子工具侧重不同的训练场景，可以定位模型训练中的精度问题。

为方便使用，本工具提供了统一、简易的程序接口：**PrecisionDebugger**。以 PyTorch 框架为例，通过以下示例模板和 **config.json** 可以轻松使用各种功能。

```python
from msprobe.pytorch import PrecisionDebugger

debugger = PrecisionDebugger(config_path='./config.json')
...
debugger.start() # 一般在训练循环开头启动工具
... # 循环体
debugger.stop() # 一般在训练循环末尾结束工具。必须调用，否则可能导致精度数据落盘不全
debugger.step() # 在训练循环的最后需要重置工具，非循环场景不需要
```

此外，根据以下规则，可以通过环境变量设置日志级别。
- MSPROBE_LOG_LEVEL=4，不打印任何日志；
- MSPROBE_LOG_LEVEL=3，仅打印 ERROR；
- MSPROBE_LOG_LEVEL=2，仅打印 WARNING、ERROR；
- MSPROBE_LOG_LEVEL=1，仅打印 INFO、WARNING、ERROR（默认配置）；
- MSPROBE_LOG_LEVEL=0，打印 DEBUG、INFO、WARNING、ERROR。

例如在 shell 脚本：

```shell
export MSPROBE_LOG_LEVEL={x}
```
**config.json** 的配置要求和各功能具体的使用指导详见后续章节。

## 环境和依赖

- 硬件环境请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/quickstart/quickstart/quickstart_18_0002.html)》。
- 软件环境请参见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)》安装昇腾设备开发或运行环境，即toolkit软件包。

以上环境依赖请根据实际环境选择适配的版本。

## 版本配套说明

- msprobe支持AscendPyTorch 1.11.0或更高版本，支持的PyTorch和CANN以及PyTorch和python软件版本配套关系请参见《[Ascend Extension for PyTorch插件](https://gitcode.com/Ascend/pytorch)》。
- msprobe支持MindSpore 2.4.0或更高版本，支持的MindSpore和CANN以及MindSpore和python软件版本配套关系请参见《[MindSpore版本发布列表](https://www.mindspore.cn/versions)》。
- msprobe支持MSAdapter 2.1.0。
- msprobe支持的固件驱动版本与配套CANN软件支持的固件驱动版本相同，开发者可通过“[昇腾社区-固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community?product=2&model=28&cann=8.0.RC3.alpha003&driver=1.0.25.alpha)”页面根据产品型号与CANN软件版本获取配套的固件与驱动。


## 🚨 工具限制与注意事项

**1. Pytorch 框架下，工具暂不支持 Fully Sharded Data Parallel(FSDP)。**

**2. 工具读写的所有路径，如config_path、dump_path等，只允许包含大小写字母、数字、下划线、斜杠、点和短横线。**

## ⚙️ 安装

请参见[安装指导说明](./docs/01.installation.md)。

## 🌟 新版本特性

请参见[特性变更说明](./docs/01.installation.md#特性变更说明)。

## 🛠️ config.json [介绍](./docs/02.config_introduction.md) 和 [示例](./docs/03.config_examples.md)

## 🧰 主要功能

### 0 用前必看

使用工具前，建议先浏览[**工具功能模块简介、适用场景和当前版本局限性**](./docs/25.tool_function_introduction.md)，了解功能特性。

### 1 数据采集

msprobe 通过在训练脚本中添加 PrecisionDebugger 接口的方式对 API 执行精度数据 dump 操作。对应 config.json 中的 "statistics" 或 "tensor" task。

[PyTorch 场景的数据采集](./docs/05.data_dump_PyTorch.md)

[MindSpore 场景的数据采集](./docs/06.data_dump_MindSpore.md)

[MSAdapter 场景的数据采集](./docs/29.data_dump_MSAdapter.md)

### 2 精度预检

精度预检旨在昇腾 NPU 上扫描训练模型中的所有 API 进行 API 复现，给出精度情况的诊断和分析。对应 config.json 中的 "run_ut" task。

PyTorch 场景的[离线预检](./docs/07.accuracy_checker_PyTorch.md)

MindSpore 动态图场景的[离线预检](./docs/09.accuracy_checker_MindSpore.md)

### 3 分级可视化构图比对

该功能将msprobe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对，方便用户理解模型结构、分析精度问题。

[PyTorch 场景的分级可视化构图比对](./docs/21.visualization_PyTorch.md)

[MindSpore 场景的分级可视化构图比对](./docs/22.visualization_MindSpore.md)

### 4 精度比对

该功能进行 PyTorch 整网 API 粒度的数据 dump、精度比对，进而定位训练场景下的精度问题。

[PyTorch 场景的精度比对](./docs/10.accuracy_compare_PyTorch.md)

[MindSpore 场景的精度比对](./docs/11.accuracy_compare_MindSpore.md)

### 5 数据解析

该功能用于比对前后两次 NPU ACL 层级 dump 数据的一致性。

[PyTorch 场景的数据解析](./docs/14.data_parse_PyTorch.md)

### 6 无标杆比对

[PyTorch 场景的无标杆比对](./docs/15.free_benchmarking_PyTorch.md)

[MindSpore 场景的无标杆比对](./docs/16.free_benchmarking_MindSpore.md)

### 7 梯度状态监测

本功能用于采集梯度数据并进行梯度相似度比对，可以精准定位出现问题的 step。

[兼容 PyTorch 和 MindSpore 框架的梯度监测](./docs/17.grad_probe.md)

### 8 在线精度比对

在线精度比对是实现在PyTorch训练过程中直接完成精度比对并输出比对结果的功能，是NPU与CPU之间的精度比对。

[PyTorch 场景的在线精度比对](./docs/18.online_dispatch.md)

### 9 训练状态监控

该功能收集和聚合模型训练过程中的网络层，优化器， 通信算子的中间值，帮助诊断模型训练过程中计算， 通信，优化器各部分出现的异常情况。

[兼容 PyTorch 和 MindSpore 框架的训练状态监控](./docs/19.monitor.md)

### 10 单算子API自动生成脚本

该功能将msprobe工具dump的精度数据进行解析，自动生成单API脚本，用于复现整网中出现的算子问题，降低用户复现问题的成本，供开发分析算子问题。

[PyTorch 单算子API自动生成脚本](./docs/23.generate_operator_PyTorch.md)

[MindSpore 单算子API自动生成脚本](./docs/33.generate_operator_MindSpore.md)

### 11 数码关联

该功能只支持 MindSpore 静态图场景，用于将IR图与dump数据进行关联，获取dump数据和代码调用栈的关联关系。

[MindSpore 场景的数码关联](./docs/24.code_mapping_Mindspore.md)

### 12 溢出检测与解析

溢出检测用于采集溢出 API 或 模块的精度数据，而溢出解析则是通过对溢出数据的分析，进一步判断是否为正常溢出。对应 config.json 中的 "overflow_check" task。 
推荐直接使用[数据采集](#1-数据采集)功能采集统计量信息,检测溢出问题。

[PyTorch 场景的溢出检测与解析](./docs/12.overflow_check_PyTorch.md)

[MindSpore 场景的溢出检测](./docs/13.overflow_check_MindSpore.md)

[MSAdapter 场景的溢出检测](./docs/30.overflow_check_MSAdapter.md)

### 13 训练检查

该工具主要包括：

训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异。

[训练前配置检查](./docs/31.config_check.md)

训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。

[checkpoint比对](./docs/32.ckpt_compare.md)

### 14 强化学习数据采集

主要能力：

灵活采集强化学习中重要关键过程数据，并支持比对。

[强化学习数据采集](./docs/34.RL_collect.md)

### 15 整网首个溢出节点分析

多rank场景下通过dump数据找到首个出现Nan或Inf的节点。

[PyTorch 场景整网首个溢出节点分析](./docs/35.nan_analyze.md)


## 📑 补充材料

[无标杆比对功能在 PyTorch 场景的性能基线报告](./docs/S02.report_free_benchmarking_validation_performance_baseline.md)

## ❗ 免责声明
本工具建议执行用户与安装用户保持一致，如果您要使用 root 执行，请自行关注 root 高权限触及的安全风险。

## ❓ FAQ

[FAQ for PyTorch](./docs/FAQ.md)
