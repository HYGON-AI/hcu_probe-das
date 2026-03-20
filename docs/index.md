# MindStudio Probe

## 简介

MindStudio Probe（MindStudio精度调试工具，msProbe）是针对昇腾提供的全场景精度工具链，专为模型开发的精度调试环节设计，可显著提升用户定位模型精度问题的效率。

## 未来规划

未来规划会刷新在[MindStudio Probe Roadmap](https://gitcode.com/Ascend/msprobe/issues/27)中，欢迎访问msProbe最新规划动态。

## 社区会议

MindStudio Probe系列TC及SIG会议安排请查看[Ascend会议中心](https://meeting.ascend.osinfra.cn/)。

## 最新消息

[2025.12.31]：MindStudio Probe精度调试工具全面开源。

## 目录结构

关键目录如下，详细介绍参见[项目目录](./zh/dir_structure.md)。

```text
MindStudio-probe
├── csrc                         # C/C++源码目录
├── cmake                        # 存放解析C化部分cmake文件
├── docs                         # 文档目录
├── examples                     # 工具配置样例存放目录
├── output                       # 交付件生成目录
├── plugins                      # 插件类代码总入口
├── python                       # Python源码目录
├── scripts                      # 存放安装卸载升级脚本
├── test                         # 测试代码目录
├── setup.py                     # 端到端打包构建脚本
├── README.md                    # 整体仓代码说明
└── LICENSE                      # LICENSE文件
```

## 环境部署

### 版本说明

|  版本   |支持PyTorch版本|支持MindSpore版本|支持Python版本|支持CANN版本|
|:-----:|:--:|:--:|:--:|:--:|
| 26.0.0(在研版本) |2.1/2.2/2.5/2.6/2.7/2.8/2.9|2.4.0/2.5.0/2.6.0/2.7.1|3.8-3.12|大于等于 CANN 8.3.RC1|
| 26.0.0-alpha.2 |2.1/2.2/2.5/2.6/2.7/2.8/2.9|2.4.0/2.5.0/2.6.0/2.7.1|3.8-3.12|大于等于 CANN 8.3.RC1|
| 26.0.0-alpha.1 |2.1/2.2/2.5/2.6/2.7/2.8|2.4.0/2.5.0/2.6.0/2.7.1|3.8-3.11|大于等于 CANN 8.3.RC1|

### 工具安装

安装msProbe工具，具体请参见《[msProbe工具安装指南](./zh/msprobe_install_guide.md)》。

## 快速入门

msProbe工具快速入门当前提供在PyTorch和MindSpore训练场景中，通过一个可执行样例，串联msProbe工具的训练前配置检查、精度数据采集、精度预检、训练状态监测及精度比对功能，帮助用户快速上手。详细快速入门可参见《PyTorch场景msTT工具快速入门》中的“[模型精度调试](https://gitcode.com/Ascend/mstt/blob/master/docs/zh/pytorch_mstt_quick_start.md#%E6%A8%A1%E5%9E%8B%E7%B2%BE%E5%BA%A6%E8%B0%83%E8%AF%95)”或《MindSpore场景msTT工具快速入门》中的“[模型精度调试](https://gitcode.com/Ascend/mstt/blob/master/docs/zh/mindspore_mstt_quick_start.md#%E6%A8%A1%E5%9E%8B%E7%B2%BE%E5%BA%A6%E8%B0%83%E8%AF%95)”。

## 功能介绍

| 使用场景            |  子模式/细分场景   | 功能项          | 功能说明                                                                                          | 参考文档                                                                                                                                                               |
|-----------------|:-----------:|--------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **vLLM推理**      |   eager模式   | 数据采集         | 完成msProbe精度数据采集操作                                                                             | [数据采集](https://docs.vllm.ai/projects/ascend/zh-cn/latest/developer_guide/performance_and_debug/msprobe_guide.html)                                                 |
|                 |             | 数据比对         | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题<br/>请参考分级可视化构图比对或精度比对                                      | [分级可视化构图比对](./zh/accuracy_compare/pytorch_visualization_instruct.md)<br>[精度比对](./zh/accuracy_compare/pytorch_accuracy_compare_instruct.md)                   |
|                 | aclgraph图模式 | 数据采集         | 通过acl_save接口完成精度数据采集操作                                                                        | [数据采集](./zh/dump/aclgraph_dump_instruct.md)                                                                                                                     |
|                 | torchair图模式 | 数据采集         | 通过set_ge_dump_config接口完成精度数据采集操作                                                              | [数据采集](./zh/dump/torchair_dump_instruct.md)                                                                                                                     |
|                 |             | 精度比对         | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题                                                            | [精度比对](./zh/accuracy_compare/torchair_compare_instruct.md)                                                                                                      |
| **SGLang推理**    |   eager模式   | 数据采集         | 完成msProbe精度数据采集操作                                                                             | [数据采集](./zh/dump/sglang_eager_dump_instruct.md)                                                                                                                 |
|                 |             | 数据比对         | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题                                                            | [分级可视化构图比对](./zh/accuracy_compare/pytorch_visualization_instruct.md)<br>[精度比对](./zh/accuracy_compare/pytorch_accuracy_compare_instruct.md)                   |
| **ATB推理**       |      -      | 数据采集         | 通过在ATB模型运行前，加载ATB dump模块的方式，实现对ATB模型运行过程中的精度数据的采集                                             | [数据采集](./zh/dump/atb_data_dump_instruct.md)                                                                                                                     |
|                 |             | 精度比对         | 将ATB dump的精度数据进行精度比对，进而定位精度问题                                                                 | [精度比对](./zh/accuracy_compare/atb_data_compare_instruct.md)                                                                                                      |
|                 |             | 数据转换         | 将ATB dump的精度数据转换为numpy（.npy）或PyTorch tensor（.pt）格式文件                                          | [数据转换](./zh/dump/data_parse_instruct.md)                                                                                                                        |
| **离线模型推理**      |      -      | 数据采集         | 完成msProbe精度数据采集操作                                                                             | [数据采集](./zh/dump/infer_offline_dump_instruct.md)                                                                                                                |
|                 |             | 精度比对         | 提供一键式离线模型比对功能，仅需输入模型即可完成比对，无需提前采集数据，快速输出结果                                                    | [精度比对](./zh/accuracy_compare/infer_compare_offline_model_instruct.md)                                                                                           |
|                 |             | 离线模型数据精度比对   | 提供离线模型数据比对功能，输入离线模型的dump数据进行精度比对                                                              | [离线模型数据精度比对](./zh/accuracy_compare/offlline_data_compare_instruct.md)                                                                                           |
|                 |             | 数据转换         | 将离线模型的dump数据转换为numpy（.npy）或PyTorch tensor（.pt）格式文件                                            | [数据转换](./zh/dump/data_parse_instruct.md)                                                                                                                        |
| **PyTorch训练**   |      -      | 训练前配置检查      | 训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异                                                                | [训练前配置检查](./zh/config_check_instruct.md)                                                                                                                        |
|                 |             | 数据采集         | 通过config.json配置，完成msProbe精度数据采集操作<br/>config.json配置文件详细介绍请参见配置文件介绍和config.json配置样例            | [数据采集](./zh/dump/pytorch_data_dump_instruct.md)<br>[配置文件介绍](./zh/dump/config_json_introduct.md)<br>[config.json配置样例](./zh/dump/config_json_examples.md)   |
|                 |             | 精度预检         | 在昇腾NPU上扫描训练模型中的所有API，给出精度情况的诊断和分析                                                             | [精度预检](./zh/accuracy_checker/pytorch_accuracy_checker_instruct.md)                                                                                              |
|                 |             | 分级可视化构图比对    | 将msProbe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对                                               | [分级可视化构图比对](./zh/accuracy_compare/pytorch_visualization_instruct.md)                                                                                            |
|                 |             | 精度比对         | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题                                                            | [精度比对](./zh/accuracy_compare/pytorch_accuracy_compare_instruct.md)                                                                                              |
|                 |             | 训练状态监测       | 收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况                                  | [训练状态监测](./zh/monitor_instruct.md)                                                                                                                              |
|                 |             | checkpoint比对 | 训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度                                                           | [checkpoint比对](./zh/checkpoint_compare_instruct.md)                                                                                                             |
|                 |             | 整网首个溢出节点分析   | 多rank场景下通过dump数据找到首个出现Nan或Inf的节点                                                              | [整网首个溢出节点分析](./zh/overflow_check/overflow_check_instruct.md)                                                                                                    |
|                 |             | 趋势可视化        | 将msProbe工具数据采集或训练状态监测的统计量数据从迭代步数、节点rank和张量目标三个维度进行趋势可视化                                       | [趋势可视化](./zh/accuracy_compare/trend_visualization_instruct.md)                                                                                                  |
| **MindSpore训练** |      -      | 训练前配置检查      | 训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异                                                                | [训练前配置检查](./zh/config_check_instruct.md)                                                                                                                        |
|                 |             | 数据采集         | 通过config.json配置，完成msProbe精度数据采集操作<br/>config.json配置文件详细介绍请参见配置文件介绍和config.json配置样例            | [数据采集](./zh/dump/mindspore_data_dump_instruct.md)<br>[配置文件介绍](./zh/dump/config_json_introduct.md)<br>[config.json配置样例](./zh/dump/config_json_examples.md) |
|                 |             | 精度预检         | 在昇腾NPU上扫描训练模型中的所有API，给出精度情况的诊断和分析                                                             | [精度预检](./zh/accuracy_checker/mindspore_accuracy_checker_instruct.md)                                                                                            |
|                 |             | 分级可视化构图比对    | 将msProbe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对                                               | [分级可视化构图比对](./zh/accuracy_compare/mindspore_visualization_instruct.md)                                                                                          |
|                 |             | 精度比对         | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题                                                            | [精度比对](./zh/accuracy_compare/mindspore_accuracy_compare_instruct.md)                                                                                            |
|                 |             | 训练状态监测       | 收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况                                  | [训练状态监测](./zh/monitor_instruct.md)                                                                                                                              |
|                 |             | 溢出检测与解析      | 溢出检测用于采集溢出API或模块的精度数据，而溢出解析则是通过对溢出数据的分析，进一步判断是否为正常溢出<br/>推荐直接使用数据采集功能采集统计量信息，检测溢出问题，具体请参见数据采集 | [溢出检测与解析](./zh/overflow_check/mindspore_overflow_check_instruct.md)<br>[数据采集](./zh/dump/mindspore_data_dump_instruct.md)                                     |
|                 |             | checkpoint比对 | 训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度                                                           | [checkpoint比对](./zh/checkpoint_compare_instruct.md)                                                                                                             |
|                 |             | 趋势可视化        | 将msProbe工具数据采集或训练状态监测的统计量数据从迭代步数、节点rank和张量目标三个维度进行趋势可视化                                       | [趋势可视化](./zh/accuracy_compare/trend_visualization_instruct.md)                                                                                                  |
| **MSAdapter场景** |      -      | 数据采集         | 通过config.json配置，完成msProbe精度数据采集操作<br/>config.json配置文件详细介绍请参见配置文件介绍和config.json配置样例            | [数据采集](./zh/dump/msadapter_data_dump_instruct.md)<br>[配置文件介绍](./zh/dump/config_json_introduct.md)<br>[config.json配置样例](./zh/dump/config_json_examples.md) |
|                 |             | checkpoint比对 | 训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度                                                           | [checkpoint比对](./zh/checkpoint_compare_instruct.md)                                                                                                             |

## 补充材料

- [PyTorch场景的精度数据采集基线报告](./zh/baseline/pytorch_data_dump_perf_baseline.md)

- [MindSpore场景的精度预检基线报告](./zh/baseline/mindspore_accuracy_checker_perf_baseline.md)

- [MindSpore场景的精度数据采集基线报告](./zh/baseline/mindspore_data_dump_perf_baseline.md)

- [训练状态监测工具标准性能基线报告](./zh/baseline/monitor_perf_baseline.md)

## FAQ

FAQ汇总了在使用msProbe工具过程中可能遇到的问题，具体请参见[FAQ](./zh/faq.md)。

## 贡献指导

介绍如何向msProbe反馈问题、需求以及为msProbe贡献的代码开发流程，具体请参见[为MindStudio Probe贡献](https://raw.gitcode.com/Ascend/msprobe/raw/master/CONTRIBUTING.md)。

## 联系我们

<div>
  <a href="https://raw.gitcode.com/kali20gakki1/Imageshack/raw/main/CDC0BEE2-8F11-477D-BD55-77A15417D7D1_4_5005_c.jpeg">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
</div>

## 安全声明

描述msProbe产品的安全加固信息、公网地址信息等内容，具体请参见[安全声明](./zh/security_statement.md)。

## 免责声明

- 本工具仅供调试和开发之用，使用者需自行承担使用风险，并理解以下内容：
  - 数据处理及删除：用户在使用本工具过程中产生的数据属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防信息泄露。
  - 数据保密与传播：使用者了解并同意不得将通过本工具产生的数据随意外发或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本工具及其开发者概不负责。
  - 用户输入安全性：用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于由于输入命令行不当所导致的问题，本工具及其开发者概不负责。
- 免责声明范围：本免责声明适用于所有使用本工具的个人或实体。使用本工具即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本工具。
- 在使用本工具之前，请谨慎阅读并理解以上免责声明的内容。对于使用本工具所产生的任何问题或疑问，请及时联系开发者。

## License

介绍msProbe产品的使用许可证，具体请参见[LICENSE](https://raw.gitcode.com/Ascend/msprobe/raw/master/LICENSE)文件。

介绍msProbe工具docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](https://raw.gitcode.com/Ascend/msprobe/raw/master/docs/LICENSE)文件。

## 致谢

msProbe由华为公司的下列部门联合贡献：

- 昇腾计算MindStudio开发部
- 分布式并行计算实验室

感谢来自社区的每一个PR，欢迎贡献msProbe！

```{toctree}
:maxdepth: 2
:caption: 开始使用
:hidden:
简介 <zh/overview>
安装指南 <zh/msprobe_install_guide>
常见问题 <zh/faq>
安全声明 <zh/security_statement>
```

```{toctree}
:maxdepth: 2
:caption: 功能指南
:hidden:

训练前配置检查 <zh/config_check_instruct>
数据采集 <zh/dump/pytorch_data_dump_instruct>
分级可视化构图比对 <zh/accuracy_compare/pytorch_visualization_instruct>
精度比对 <zh/accuracy_compare/pytorch_accuracy_compare_instruct>
训练状态检测 <zh/monitor_instruct.md>
精度预检 <zh/accuracy_checker/pytorch_accuracy_checker_instruct>
```

```{toctree}
:maxdepth: 2
:caption: 开发者指南
:hidden:

开发指南 <zh/developer_guide/development_guide>
```
