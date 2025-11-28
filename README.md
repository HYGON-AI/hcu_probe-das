# 📖MindStudio Probe

![python](https://img.shields.io/badge/python-3.8|3.9|3.10-blue)
![platform](https://img.shields.io/badge/platform-Linux-yellow)

## 简介

MindStudio Probe（简称msProbe）是精度调试工具包，是针对昇腾提供的全场景精度工具链，帮助用户快速提高模型精度定位效率。

msProbe主要包括精度数据采集（dump）、精度预检、溢出检测和精度比对等功能，这些功能侧重不同的训练或推理场景，可以定位模型训练或推理中的精度问题。

## [版本说明](docs/zh/release_notes.md)

包含msProbe的软件版本配套关系和软件包下载以及每个版本的特性变更说明。

## ⚙️环境部署

### 环境和依赖

- 选择作为用户运行AI应用的昇腾设备，详细设备型号请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)》。
- 安装配套版本的CANNToolkit开发套件包并配置环境变量，详情请参见《CANN 软件安装指南》中”[选择安装场景](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)“章节的”训练&推理&开发调试“场景。

以上环境依赖请根据实际环境选择适配的版本。

### 工具安装

安装msProbe工具，详情请参见[安装指导](docs/zh/msprobe_install_guide.md)。

## 🚀 快速入门

详细快速入门可参见《训练场景工具快速入门》中的“[模型精度调试](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/msquickstart/atlasquick_train_0023.html?framework=pytorch)”。

## 🚨 工具限制与注意事项

1. 工具读写的所有路径，如`config_path`、`dump_path`等，只允许包含大小写字母、数字、下划线、斜杠、点和短横线。

2. 出于安全性及权限最小化角度考虑，msProbe工具不应使用root等高权限账户使用，建议使用普通用户权限安装执行。

3. 使用msProbe工具前请确保执行用户的umask值大于等于0027，否则可能会导致工具生成的精度数据文件和目录权限过大。

4. 用户须自行保证使用最小权限原则，如给工具输入的文件要求other用户不可写，在一些对安全要求更严格的功能场景下还需确保输入的文件group用户不可写。

5. 使用工具前，建议先浏览[工具功能模块简介、适用场景和当前版本局限性](docs/zh/limitations_and_precautions.md)，了解功能特性。

## 🧰 功能介绍

### vLLM推理场景

#### torchair图模式

[**数据采集**](docs/zh/dump/torchair_dump_instruct.md)

完成msProbe精度数据采集操作。

精度数据采集配置需要通过config.json配置文件，详细介绍请参见[介绍](docs/zh/dump/config_json_introduct.md)和[示例](docs/zh/dump/config_json_examples.md)。

**[精度比对](docs/zh/accuracy_compare/torchair_compare_instruct.md)**

将msProbe工具dump的精度数据进行精度比对，进而定位精度问题。

### PyTorch训练场景

#### [训练前配置检查](docs/zh/config_check_instruct.md)

训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异。

#### [数据采集](docs/zh/dump/pytorch_data_dump_instruct.md)

通过config.json配置，完成msProbe精度数据采集操作。

精度数据采集配置需要通过config.json配置文件，详细介绍请参见[介绍](docs/zh/dump/config_json_introduct.md)和[示例](docs/zh/dump/config_json_examples.md)。


#### [精度预检](docs/zh/accuracy_checker/pytorch_accuracy_checker_instruct.md)

在昇腾NPU上扫描训练模型中的所有API，给出精度情况的诊断和分析。

#### [分级可视化构图比对](docs/zh/accuracy_compare/pytorch_visualization_instruct.md)

将msProbe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对，方便用户理解模型结构、分析精度问题。

#### [精度比对](docs/zh/accuracy_compare/pytorch_accuracy_compare_instruct.md)

将msProbe工具dump的精度数据进行精度比对，进而定位精度问题。

#### [训练状态监控](docs/zh/monitor_instruct.md)

收集和聚合模型训练过程中的网络层，优化器， 通信算子的中间值，帮助诊断模型训练过程中计算， 通信，优化器各部分出现的异常情况。

#### [checkpoint比对](docs/zh/checkpoint_compare_instruct.md)

训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。

#### [整网首个溢出节点分析](docs/zh/overflow_check/overflow_check_instruct.md)

多rank场景下通过dump数据找到首个出现Nan或Inf的节点。

### MindSpore训练场景

#### [训练前配置检查](docs/zh/config_check_instruct.md)

训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异。

#### [数据采集](docs/zh/dump/mindspore_data_dump_instruct.md)

完成msProbe精度数据采集操作。

精度数据采集配置需要通过config.json配置文件，详细介绍请参见[介绍](docs/zh/dump/config_json_introduct.md)和[示例](docs/zh/dump/config_json_examples.md)。


#### [精度预检](docs/zh/accuracy_checker/mindspore_accuracy_checker_instruct.md)

在昇腾NPU上扫描训练模型中的所有API，给出精度情况的诊断和分析。

#### [分级可视化构图比对](docs/zh/accuracy_compare/mindspore_visualization_instruct.md)

将msProbe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对，方便用户理解模型结构、分析精度问题。

#### [精度比对](docs/zh/accuracy_compare/mindspore_accuracy_compare_instruct.md)

将msProbe工具dump的精度数据进行精度比对，进而定位精度问题。

#### [训练状态监控](docs/zh/monitor_instruct.md)

收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况。

#### [溢出检测与解析](docs/zh/overflow_check/mindspore_overflow_check_instruct.md)

溢出检测用于采集溢出API或模块的精度数据，而溢出解析则是通过对溢出数据的分析，进一步判断是否为正常溢出。

推荐直接使用[数据采集](#数据采集-1)功能采集统计量信息，检测溢出问题。

#### [checkpoint比对](docs/zh/checkpoint_compare_instruct.md)

训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。

### MSAdapter场景

#### [数据采集](docs/zh/dump/msadapter_data_dump_instruct.md)

完成msProbe精度数据采集操作。

精度数据采集配置需要通过config.json配置文件，详细介绍请参见[介绍](docs/zh/dump/config_json_introduct.md)和[示例](docs/zh/dump/config_json_examples.md)。

#### [checkpoint比对](docs/zh/checkpoint_compare_instruct.md)

训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。

## 📑补充材料

- [PyTorch场景的精度数据采集基线报告](docs/zh/baseline/pytorch_data_dump_perf_baseline.md)

- [MindSpore场景的精度预检基线报告](docs/zh/baseline/mindspore_accuracy_checker_perf_baseline.md)

- [MindSpore场景的精度数据采集基线报告](docs/zh/baseline/mindspore_data_dump_perf_baseline.md)

- [训练状态监控工具标准性能基线报告](docs/zh/baseline/monitor_perf_baseline.md)


## ❓FAQ

[FAQ for PyTorch](docs/zh/faq.md)

## ❗免责声明

本工具建议执行用户与安装用户保持一致，如果您要使用root执行，请自行关注root高权限触及的安全风险。

## License

介绍msProbe产品的BSD许可证。详见[LICENSE](LICENSE)文件。

## 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交issues，我们会尽快回复。感谢您的支持。

## 致谢

msProbe由华为公司的下列部门联合贡献：

- 昇腾计算MindStudio开发部
- 分布式并行计算实验室

感谢来自社区的每一个PR，欢迎贡献msProbe！
