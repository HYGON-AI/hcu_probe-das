# 📖MindStudio Probe

![python](https://img.shields.io/badge/python-3.8|3.9|3.10-blue)
![platform](https://img.shields.io/badge/platform-Linux-yellow)

## 简介

MindStudio Probe（简称msProbe）是模型开发的精度调试环节使用的工具包，是针对昇腾提供的全场景精度工具链，帮助用户快速提高模型精度定位效率。

msProbe主要包括精度数据采集（dump）、精度预检、溢出检测和精度比对等功能，这些功能侧重不同的训练或推理场景，可以定位模型训练或推理中的精度问题。

## 版本说明

msProbe的版本说明包含msProbe的软件版本配套关系和软件包下载以及每个版本的特性变更说明，具体请参见《[版本说明](docs/zh/release_notes.md)》 。

## ⚙️环境部署

### 环境和依赖

使用msProbe工具前，要求已存在可执行的用户AI应用，其中昇腾环境要求：

- 可运行用户AI应用的昇腾设备，详细设备型号请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)》。
- 安装配套版本的CANN Toolkit开发套件包和ops算子包并配置环境变量，详情请参见《CANN 软件安装指南》。

以上环境依赖请根据实际环境选择适配的版本。

### 工具安装

安装msProbe工具，包括从PyPI安装、下载whl包安装和从源码安装三种方式，具体请参见《[msProbe工具安装指南](docs/zh/msprobe_install_guide.md)》。

## 🚀 快速入门

详细快速入门可参见《训练场景工具快速入门》中的“[模型精度调试](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/msquickstart/atlasquick_train_0023.html?framework=pytorch)（PyTorch场景）”或“[模型精度调试](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/msquickstart/atlasquick_train_0006.html?framework=mindspore)（MindSpore场景）。

## 🚨 工具限制与注意事项

- 工具读写的所有路径，如`config_path`、`dump_path`等，只允许包含大小写字母、数字、下划线、斜杠、点和短横线。

- 出于安全性及权限最小化角度考虑，msProbe工具不应使用root等高权限账户使用，建议使用普通用户权限安装执行。

- 使用msProbe工具前请确保执行用户的umask值大于等于0027，否则可能会导致工具生成的精度数据文件和目录权限过大。

- 用户须自行保证使用最小权限原则，如给工具输入的文件要求other用户不可写，在一些对安全要求更严格的功能场景下还需确保输入的文件group用户不可写。

- 使用工具前，建议先浏览[工具功能模块简介、适用场景和当前版本局限性](docs/zh/limitations_and_precautions.md)，了解功能特性。

## 🧰 功能介绍

### vLLM推理场景

#### torchair图模式

1. [数据采集](docs/zh/dump/torchair_dump_instruct.md)

   通过config.json配置，完成msProbe精度数据采集操作。

   config.json配置文件详细介绍请参见[配置文件介绍](docs/zh/dump/config_json_introduct.md)和[config.json配置样例](docs/zh/dump/config_json_examples.md)。

2. [精度比对](docs/zh/accuracy_compare/torchair_compare_instruct.md)

   将msProbe工具dump的精度数据进行精度比对，进而定位精度问题。


### ATB推理场景

1. [数据采集](docs/zh/dump/atb_data_dump_instruct.md)

   通过在ATB模型运行前，加载ATB dump模块的方式，实现对ATB模型运行过程中的精度数据的采集。

2. [精度比对](docs/zh/accuracy_compare/atb_data_compare_instruct.md)

   提供ATB场景的精度比对功能，帮助定位精度问题。


### 离线模型推理场景

1. [数据采集](docs/zh/dump/infer_offline_dump_instruct.md)

   完成msProbe精度数据采集操作。

2. [精度比对](docs/zh/accuracy_compare/infer_compare_offline_model_instruct.md)

   提供一键式离线模型比对功能，仅需输入模型即可完成比对，无需提前采集数据，快速输出结果。


### PyTorch训练场景

1. [训练前配置检查](docs/zh/config_check_instruct.md)

   训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异。

2. [数据采集](docs/zh/dump/pytorch_data_dump_instruct.md)

   通过config.json配置，完成msProbe精度数据采集操作。

   config.json配置文件详细介绍请参见[配置文件介绍](docs/zh/dump/config_json_introduct.md)和[config.json配置样例](docs/zh/dump/config_json_examples.md)。

3. [精度预检](docs/zh/accuracy_checker/pytorch_accuracy_checker_instruct.md)

   在昇腾NPU上扫描训练模型中的所有API，给出精度情况的诊断和分析。

4. [分级可视化构图比对](docs/zh/accuracy_compare/pytorch_visualization_instruct.md)

   将msProbe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对，方便用户理解模型结构、分析精度问题。

5. [精度比对](docs/zh/accuracy_compare/pytorch_accuracy_compare_instruct.md)

   将msProbe工具dump的精度数据进行精度比对，进而定位精度问题。

6. [训练状态监控](docs/zh/monitor_instruct.md)

   收集和聚合模型训练过程中的网络层，优化器， 通信算子的中间值，帮助诊断模型训练过程中计算， 通信，优化器各部分出现的异常情况。

7. [checkpoint比对](docs/zh/checkpoint_compare_instruct.md)

   训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。

8. [整网首个溢出节点分析](docs/zh/overflow_check/overflow_check_instruct.md)

   多rank场景下通过dump数据找到首个出现Nan或Inf的节点。

9. [趋势可视化](docs/zh/accuracy_compare/trend_visualization_instruct.md)

   将msProbe工具数据采集或训练状态监控的统计量数据从迭代步数、节点rank和张量目标三个维度进行趋势可视化，方便用户从整体的趋势分布观测精度数据。


### MindSpore训练场景

1. [训练前配置检查](docs/zh/config_check_instruct.md)

   训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异。

2. [数据采集](docs/zh/dump/mindspore_data_dump_instruct.md)

   通过config.json配置，完成msProbe精度数据采集操作。

   config.json配置文件详细介绍请参见[配置文件介绍](docs/zh/dump/config_json_introduct.md)和[config.json配置样例](docs/zh/dump/config_json_examples.md)。

3. [精度预检](docs/zh/accuracy_checker/mindspore_accuracy_checker_instruct.md)

   在昇腾NPU上扫描训练模型中的所有API，给出精度情况的诊断和分析。

4. [分级可视化构图比对](docs/zh/accuracy_compare/mindspore_visualization_instruct.md)

   将msProbe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对，方便用户理解模型结构、分析精度问题。

5. [精度比对](docs/zh/accuracy_compare/mindspore_accuracy_compare_instruct.md)

   将msProbe工具dump的精度数据进行精度比对，进而定位精度问题。

6. [训练状态监控](docs/zh/monitor_instruct.md)

   收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况。

7. [溢出检测与解析](docs/zh/overflow_check/mindspore_overflow_check_instruct.md)

   溢出检测用于采集溢出API或模块的精度数据，而溢出解析则是通过对溢出数据的分析，进一步判断是否为正常溢出。

   推荐直接使用[数据采集](#数据采集-1)功能采集统计量信息，检测溢出问题，具体请参见。

8. [checkpoint比对](docs/zh/checkpoint_compare_instruct.md)

   训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。

9. [趋势可视化](docs/zh/accuracy_compare/trend_visualization_instruct.md)

   将msProbe工具数据采集或训练状态监控的统计量数据从迭代步数、节点rank和张量目标三个维度进行趋势可视化，方便用户从整体的趋势分布观测精度数据。


### MSAdapter场景

1. [数据采集](docs/zh/dump/msadapter_data_dump_instruct.md)

   通过config.json配置，完成msProbe精度数据采集操作。

   config.json配置文件详细介绍请参见[配置文件介绍](docs/zh/dump/config_json_introduct.md)和[config.json配置样例](docs/zh/dump/config_json_examples.md)。

2. [checkpoint比对](docs/zh/checkpoint_compare_instruct.md)

   训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。


## 📑补充材料

- [PyTorch场景的精度数据采集基线报告](docs/zh/baseline/pytorch_data_dump_perf_baseline.md)

- [MindSpore场景的精度预检基线报告](docs/zh/baseline/mindspore_accuracy_checker_perf_baseline.md)

- [MindSpore场景的精度数据采集基线报告](docs/zh/baseline/mindspore_data_dump_perf_baseline.md)

- [训练状态监控工具标准性能基线报告](docs/zh/baseline/monitor_perf_baseline.md)


## ❓FAQ

[FAQ for PyTorch](docs/zh/faq.md)

## ❗免责声明

1. msProbe提供的所有内容仅供您用于非商业目的。
2. 如您在使用msProbe过程中，发现任何问题（包括但不限于功能问题、合规问题），请在GitCode提交Issue，我们将及时审视并解决。
4. msProbe功能依赖的第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，msProbe仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。
5. 对于您在使用msProbe功能过程中产生的数据属于用户责任范畴。建议您在使用完毕后及时删除相关数据，以防泄露或不必要的信息泄露。
6. 对于您在使用msProbe功能过程中产生的数据，建议您避免通过本工具随意外发或传播，对于因此产生的信息泄露、数据泄露或其他不良后果，华为不承担任何责任。
7. 对于您在使用msProbe功能过程中输入的命令行，需要您自行保证的命令行安全性，并承担因输入不当而导致的任何安全风险或损失。对于由于输入命令行不当所导致的问题，华为不承担任何责任。
8. 本工具建议执行用户与安装用户保持一致，如果您要使用root执行，请自行关注root高权限触及的安全风险。

## License

介绍msProbe产品的BSD许可证，具体请参见[LICENSE](LICENSE)文件。

## 贡献声明

1. 提交错误报告：如果您在msProbe中发现了一个不存在安全问题的漏洞，请在msProbe仓库中的Issues中搜索，以防该漏洞被重复提交，如果找不到漏洞可以创建一个新的Issues。如果发现了一个安全问题请不要将其公开，请参阅安全问题处理方式。提交错误报告时应该包含完整信息。
2. 安全问题处理：本项目中对安全问题处理的形式，请通过邮箱通知项目核心人员确认编辑。
3. 解决现有问题：通过查看仓库的Issues列表可以发现需要处理的问题信息, 可以尝试解决其中的某个问题。
4. 如何提出新功能：请使用Issues的Feature标签进行标记，我们会定期处理和确认开发。
5. 开始贡献：
   1. Fork本项目的仓库
   2. Clone到本地
   3. 创建开发分支
   4. 本地自测，提交前请通过所有的已经单元测试，以及为您要解决的问题新增单元测试。
   5. 提交代码
   6. 新建Pull Request
   7. 代码检视，您需要根据评审意见修改代码，并重新提交更新。此流程可能涉及多轮迭代。
   8. 当您的PR获得足够数量的检视者批准后，Committer会进行最终审核。
   9. 审核和测试通过后，CI会将您的PR合并入到项目的主干分支。

## 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交issues，我们会尽快回复。感谢您的支持。

## 致谢

msProbe由华为公司的下列部门联合贡献：

- 昇腾计算MindStudio开发部
- 分布式并行计算实验室

感谢来自社区的每一个PR，欢迎贡献msProbe！
