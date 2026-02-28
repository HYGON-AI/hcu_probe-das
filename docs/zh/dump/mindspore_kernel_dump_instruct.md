# MindSpore动态图场景kernel精度数据采集

## 简介

本文主要介绍kernel精度数据采集（kernel dump）的配置示例和采集结果，msProbe数据采集功能的详细使用请参见《[MindSpore场景精度数据采集](./mindspore_data_dump_instruct.md)》。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。

**约束**

- 仅支持MindSpore框架。
- 当使用msProbe数据采集功能时，level配置为 "L2" 表示采集kernel层级的算子数据，仅支持昇腾NPU平台。 

## 使用示例

1. **配置文件**

   使用kernel dump时，config.json文件中的list参数必须要填一个API名称，kernel dump目前每个step只支持采集一个API的数据。
   API名称填写参考L1 dump结果文件[dump.json](./mindspore_data_dump_instruct.md#l1级别)中的API名称，命名格式为：`{api_type}.{api_name}.{API调用次数}.{forward/backward}`。

   ```json
   {
       "task": "tensor",
       "dump_path": "/home/data_dump",
       "level": "L2",
       "rank": [],
       "step": [],
       "tensor": {
           "scope": [],
           "list": ["Functional.linear.0.backward"]
       }
   }
   ```

2. **模型脚本**

   在模型脚本中配置工具使能代码，可参考[快速入门](mindspore_dump_quick_start.md)中的“创建模型脚本”。

3. **运行训练脚本**

   在命令行中执行以下命令：

   ```bash
   python train.py
   ```
   
4. **输出说明**

   如果API kernel级数据采集成功，会打印以下信息：

   ```bash
   The kernel data of {api_name} is dumped successfully.
   ```

   注意：如果打印该信息后，没有数据生成，参考[FAQ](#faq)进行排查。

   如果kernel dump遇到不支持的API，会打印以下信息：

   ```bash
   The kernel dump does not support the {api_name} API.
   ```

   其中{api_name}是对应溢出的 API 名称。

## dump结果文件说明

kernel dump采集成功后，会在指定的dump_path目录下生成如下文件：

```ColdFusion
├── /home/data_dump/
│   ├── step0
│   │   ├── 20241201103000    # 日期时间格式，表示2024-12-01 10:30:00
│   │   │   ├── 0             # 表示 device id
│   │   │   │   ├──{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}    # kernel 层算子数据
│   │   │  ...
│   │   ├── kernel_config_{device_id}.json    # kernel dump 在接口调用过程中生成的中间文件，一般情况下无需关注
│   │  ...     
│   ├── step1
│  ...
```

## FAQ

Q：采集结果文件为空，有可能是什么原因？

A：可通过如下步骤排查：

首先需要确认工具使用方式、配置文件内容、list填写的API名称格式是否都正确无误。
其次需要确认API是否运行在昇腾NPU上，如果是运行在其他设备上则不会存在kernel级数据。
