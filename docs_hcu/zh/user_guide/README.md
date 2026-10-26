# hcuProbe 使用指南

hcuProbe针对hcu提供全场景精度调试能力，覆盖数据采集、精度比对、可视化以及多种进阶调试功能。下文按框架和使用场景汇总各场景下支持的完整功能项，请根据实际使用场景选择对应功能并参考相应文档进行配置和使用。

## PyTorch
| 核心功能 | 功能说明 | 快速入口 |
|---|---|---|
| 数据采集 | 通过 config.json 配置，一键采集模型训练/推理过程中的精度数据 | [PyTorch 数据采集](dump/pytorch_data_dump_instruct.md) |
| 精度比对 | 将采集到的精度数据进行比对，定位出现精度差异的网络层/算子 | [PyTorch 精度比对](accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| 可视化 | 解析 dump 数据还原模型图结构，分级可视化定位精度差异 | [分级可视化构图比对](accuracy_compare/pytorch_visualization_instruct.md) |

## 离线分析
[离线分析](offline_analysis/README.md) 

## 推理侧dump
[推理侧dump](infer/README.md) 