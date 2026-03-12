# MindStudio Probe

## 简介

MindStudio Probe（MindStudio精度调试工具，msProbe）是针对昇腾提供的全场景精度工具链，专为模型开发的精度调试环节设计，可显著提升用户定位模型精度问题的效率。

msProbe主要包括精度数据采集（dump）、精度预检、训练状态监测和精度比对等功能，这些功能侧重不同的训练或推理场景，可以帮助定位模型训练或推理中的精度问题。

## 环境部署

### 环境和依赖

使用msProbe工具前，要求已存在可执行的用户AI应用，其中要求昇腾环境：

- 可正常运行用户AI应用，详细设备型号请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)》。
- 已安装配套版本的CANN Toolkit开发套件包和算子包并配置环境变量，详情请参见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler)》。

### 版本说明

|  版本   |支持PyTorch版本|支持MindSpore版本|支持Python版本|支持CANN版本|
|:-----:|:--:|:--:|:--:|:--:|
| 26.0.0(在研版本) |2.1/2.2/2.5/2.6/2.7/2.8/2.9|2.4.0/2.5.0/2.6.0/2.7.1|3.8-3.12|大于等于 CANN 8.3.RC1|
| 26.0.0-alpha.2 |2.1/2.2/2.5/2.6/2.7/2.8/2.9|2.4.0/2.5.0/2.6.0/2.7.1|3.8-3.12|大于等于 CANN 8.3.RC1|
| 26.0.0-alpha.1 |2.1/2.2/2.5/2.6/2.7/2.8|2.4.0/2.5.0/2.6.0/2.7.1|3.8-3.11|大于等于 CANN 8.3.RC1|

### 工具安装

安装msProbe工具，具体请参见《[msProbe工具安装指南](docs/zh/msprobe_install_guide.md)》。

## 快速入门

msProbe工具快速入门当前提供在PyTorch和MindSpore训练场景中，通过一个可执行样例，串联msProbe工具的训练前配置检查、精度数据采集、精度预检、训练状态监测及精度比对功能，帮助用户快速上手。详细快速入门可参见《PyTorch场景msTT工具快速入门》中的“[模型精度调试](https://gitcode.com/Ascend/mstt/blob/master/docs/zh/pytorch_mstt_quick_start.md#%E6%A8%A1%E5%9E%8B%E7%B2%BE%E5%BA%A6%E8%B0%83%E8%AF%95)”或《MindSpore场景msTT工具快速入门》中的“[模型精度调试](https://gitcode.com/Ascend/mstt/blob/master/docs/zh/mindspore_mstt_quick_start.md#%E6%A8%A1%E5%9E%8B%E7%B2%BE%E5%BA%A6%E8%B0%83%E8%AF%95)”。

## 工具限制与注意事项

- 工具读写的所有路径，如`config_path`、`dump_path`等，只允许包含大小写字母、数字、下划线、斜杠、点和短横线。

- 出于安全性及权限最小化角度考虑，本工具不应使用root等高权限账户，建议使用普通用户权限安装执行。

- 使用本工具前请确保执行用户的umask值大于等于0027，否则可能会导致工具生成的精度数据文件和目录权限过大。

- 用户须自行保证使用最小权限原则，如给工具输入的文件要求other用户不可写，在一些对安全要求更严格的功能场景下还需确保输入的文件group用户不可写。

- 使用工具前，建议先浏览[工具功能模块简介、适用场景和当前版本局限性](docs/zh/limitations_and_precautions.md)，了解功能特性。

- msProbe建议执行用户与安装用户保持一致，如果使用root执行，请自行关注root高权限触及的安全风险。

## 功能介绍

### vLLM推理场景

#### eager模式

1. [数据采集](https://docs.vllm.ai/projects/ascend/zh-cn/latest/developer_guide/performance_and_debug/msprobe_guide.html)

   完成msProbe精度数据采集操作。

2. 数据比对
   
   将msProbe工具dump的精度数据进行精度比对，进而定位精度问题。

   请参考[分级可视化构图比对](docs/zh/accuracy_compare/pytorch_visualization_instruct.md)或[精度比对](docs/zh/accuracy_compare/pytorch_accuracy_compare_instruct.md)。

#### aclgraph图模式

1. [数据采集](docs/zh/dump/aclgraph_dump_instruct.md)

   通过acl_save接口完成精度数据采集操作。

#### torchair图模式

1. [数据采集](docs/zh/dump/torchair_dump_instruct.md)

   通过set_ge_dump_config接口完成精度数据采集操作。

2. [精度比对](docs/zh/accuracy_compare/torchair_compare_instruct.md)

   将msProbe工具dump的精度数据进行精度比对，进而定位精度问题。

### SGLang推理场景

#### eager模式

1. [数据采集](docs/zh/dump/sglang_eager_dump_instruct.md)

   完成msProbe精度数据采集操作。

2. 数据比对
   
   将msProbe工具dump的精度数据进行精度比对，进而定位精度问题。

   请参考[分级可视化构图比对](docs/zh/accuracy_compare/pytorch_visualization_instruct.md)或[精度比对](docs/zh/accuracy_compare/pytorch_accuracy_compare_instruct.md)。

### ATB推理场景

1. [数据采集](docs/zh/dump/atb_data_dump_instruct.md)

   通过在ATB模型运行前，加载ATB dump模块的方式，实现对ATB模型运行过程中的精度数据的采集。

2. [精度比对](docs/zh/accuracy_compare/atb_data_compare_instruct.md)

   将ATB dump的精度数据进行精度比对，进而定位精度问题。

3. [数据转换](docs/zh/dump/data_parse_instruct.md)

   将ATB dump的精度数据转换为numpy（.npy）或PyTorch tensor（.pt）格式文件。

### 离线模型推理场景

1. [数据采集](docs/zh/dump/infer_offline_dump_instruct.md)

   完成msProbe精度数据采集操作。

2. [精度比对](docs/zh/accuracy_compare/infer_compare_offline_model_instruct.md)

   提供一键式离线模型比对功能，仅需输入模型即可完成比对，无需提前采集数据，快速输出结果。

3. [离线模型数据精度比对](docs/zh/accuracy_compare/offlline_data_compare_instruct.md)
   提供离线模型数据比对功能，输入离线模型的dump数据进行精度比对。

4. [数据转换](docs/zh/dump/data_parse_instruct.md)

   将离线模型的dump数据转换为numpy（.npy）或PyTorch tensor（.pt）格式文件。

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

6. [训练状态监测](docs/zh/monitor_instruct.md)

   收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况。

7. [checkpoint比对](docs/zh/checkpoint_compare_instruct.md)

   训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。

8. [整网首个溢出节点分析](docs/zh/overflow_check/overflow_check_instruct.md)

   多rank场景下通过dump数据找到首个出现Nan或Inf的节点。

9. [趋势可视化](docs/zh/accuracy_compare/trend_visualization_instruct.md)

   将msProbe工具数据采集或训练状态监测的统计量数据从迭代步数、节点rank和张量目标三个维度进行趋势可视化，方便用户从整体的趋势分布观测精度数据。

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

6. [训练状态监测](docs/zh/monitor_instruct.md)

   收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况。

7. [溢出检测与解析](docs/zh/overflow_check/mindspore_overflow_check_instruct.md)

   溢出检测用于采集溢出API或模块的精度数据，而溢出解析则是通过对溢出数据的分析，进一步判断是否为正常溢出。

   推荐直接使用数据采集功能采集统计量信息，检测溢出问题，具体请参见[数据采集](docs/zh/dump/mindspore_data_dump_instruct.md)。

8. [checkpoint比对](docs/zh/checkpoint_compare_instruct.md)

   训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。

9. [趋势可视化](docs/zh/accuracy_compare/trend_visualization_instruct.md)

   将msProbe工具数据采集或训练状态监测的统计量数据从迭代步数、节点rank和张量目标三个维度进行趋势可视化，方便用户从整体的趋势分布观测精度数据。

### MSAdapter场景

1. [数据采集](docs/zh/dump/msadapter_data_dump_instruct.md)

   通过config.json配置，完成msProbe精度数据采集操作。

   config.json配置文件详细介绍请参见[配置文件介绍](docs/zh/dump/config_json_introduct.md)和[config.json配置样例](docs/zh/dump/config_json_examples.md)。

2. [checkpoint比对](docs/zh/checkpoint_compare_instruct.md)

   训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。

## 补充材料

- [PyTorch场景的精度数据采集基线报告](docs/zh/baseline/pytorch_data_dump_perf_baseline.md)

- [MindSpore场景的精度预检基线报告](docs/zh/baseline/mindspore_accuracy_checker_perf_baseline.md)

- [MindSpore场景的精度数据采集基线报告](docs/zh/baseline/mindspore_data_dump_perf_baseline.md)

- [训练状态监测工具标准性能基线报告](docs/zh/baseline/monitor_perf_baseline.md)

## FAQ

FAQ汇总了在使用msProbe工具过程中可能遇到的问题，具体请参见[FAQ](docs/zh/faq.md)。

## 贡献指导

介绍如何向msProbe反馈问题、需求以及为msProbe贡献的代码开发流程，具体请参见[为MindStudio Probe贡献](CONTRIBUTING.md)。

## 联系我们

<div>
  <a href="https://raw.gitcode.com/kali20gakki1/Imageshack/raw/main/CDC0BEE2-8F11-477D-BD55-77A15417D7D1_4_5005_c.jpeg">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
</div>

## 安全声明

描述msProbe产品的安全加固信息、公网地址信息等内容，具体请参见[安全声明](docs/zh/security_statement.md)。

## 免责声明

- 本工具仅供调试和开发之用，使用者需自行承担使用风险，并理解以下内容：
  - 数据处理及删除：用户在使用本工具过程中产生的数据属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防信息泄露。
  - 数据保密与传播：使用者了解并同意不得将通过本工具产生的数据随意外发或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本工具及其开发者概不负责。
  - 用户输入安全性：用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于由于输入命令行不当所导致的问题，本工具及其开发者概不负责。
- 免责声明范围：本免责声明适用于所有使用本工具的个人或实体。使用本工具即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本工具。
- 在使用本工具之前，请谨慎阅读并理解以上免责声明的内容。对于使用本工具所产生的任何问题或疑问，请及时联系开发者。

## License

介绍msProbe产品的使用许可证，具体请参见[LICENSE](LICENSE)文件。

介绍msProbe工具docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](docs/LICENSE)文件。

## 致谢

msProbe由华为公司的下列部门联合贡献：

- 昇腾计算MindStudio开发部
- 分布式并行计算实验室

感谢来自社区的每一个PR，欢迎贡献msProbe！