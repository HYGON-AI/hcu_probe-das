# 基于torch图模式（torchair）整网算子精度比对

## 简介

torchair图模式整网算子精度比对通过采集torchair图模式下模型中间算子的输入、输出数据，对比两次推理结果是否一致，从而判断模型在不同算子上的精度是否一致。torch图模式（torchair）整网算子精度比对主要支持GE数据与FX数据的比对以及GE开关融合数据的比对。

**基本概念**：
- [torchair](https://www.hiascend.com/document/detail/zh/Pytorch/720/modthirdparty/torchairuseguide/torchair_00073.html)：torch图模式的后端，用于将torch模型编译为昇腾AI处理器上的可执行程序，包括计算图转换与优化。
- GE: GE指Graph Engine（图引擎），是昇腾AI计算平台的核心组件 。GE作为计算图编译和运行的控制中心，主要提供：
    - 图优化：对神经网络计算图进行优化，提高执行效率。
    - 图编译管理：将不同框架的模型转换为统一的内部表示。
    - 图执行控制：负责优化后图的高效执行。
- FX: PyTorch FX（或称为 TorchFX，PyTorch Functional eXecution）是PyTorch的函数式执行和图转换框架，主要功能包括：
    - 将PyTorch代码转换为可优化的中间表示 (IR) 图。
    - 提供程序捕获和转换功能，支持模型重构和优化。
- 融合：图融合是指将神经网络中的多个算子合并为一个算子，以减少计算量和内存占用，一般按照规则进行融合。
- 采集：是指在模型推理过程中，将模型中间算子的输入、输出数据采集下来，保存为文件。
- 比对：是指对比两次采集到的中间算子数据是否一致，从而判断模型在不同算子上的精度是否一致。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。

**约束**

- 仅支持torch图模式（torchair）推理场景。

- 执行推理任务前需要确认torchvision版本与torch版本是否匹配。torch版本与torchvision版本的匹配关系如下：

  | torch版本 | torchvision版本 |
  | --------- | --------------- |
  | 2.3.0     | 0.18.0          |
  | 2.2.0     | 0.17.0          |
  | 2.1.0     | 0.16.0          |
  | 2.0.0     | 0.15.0          |

- 两次GE dump或两次FX dump间，请指定不同的dump数据保存路径，否则会导致数据混乱，无法区分的问题，从而影响数据比对和分析。

## GE融合模式（默认） dump数据与FX dump数据精度比对

### GE融合模式dump数据
**功能说明**
调用 `set_ge_dump_config` 接口，获取配置后的 `config` 实例，或在已有 `config` 实例上增加dump配置，配置模型compile，并执行推理。

**接口介绍**

**接口原型**

```Python
set_ge_dump_config(
        dump_path='',
        dump_mode='all',
        fusion_switch_file=None,
        dump_token=None,
        dump_layer=None,
        compiler_config=None
)
```

**参数说明**

| 参数名             | 参数描述                                                                          | 是否必选                               |
| ------------------ | --------------------------------------------------------------------------------- | -------------------------------------- |
| dump_path          | dump数据的存放路径。                                                                | 否（默认为"./"）                         |
| dump_mode          | data dump模式，用于指定dump算子输入还是输出数据。可选值有"input", "output", "all"。 | 否（默认为"all"，dump输入与输出数据）    |
| fusion_switch_file | 是否关闭融合dump功能。                                                              | 否（默认为None，开启融合）               |
| dump_token         | 指定token进行dump。格式：[1,2,5] 代表dump第1、2、5个token数据。                | 否（默认为None，dump全量数据）           |
| dump_layer         | 指定layer进行dump。格式：["Add", "Conv_1"] 代表dump Add和Conv_1两层数据。           | 否（默认为None，dump全量数据）           |
| compiler_config    | 图编译配置（CompilerConfig对象）。                                                  | 否（默认为None，返回新创建的图编译配置） |
#### 工具使用
**使用示例（已创建CompilerConfig对象）**
  ```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config  # 添加导入
  ...
  model = ...
  config = torchair.CompilerConfig()
  # 在已有config上增加dump配置
  set_ge_dump_config(dump_path="dump", compiler_config=config)
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

**使用示例（未创建CompilerConfig对象）**
  ```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config  # 添加导入
  ...
  model = ...
  # 添加获取config
  config = set_ge_dump_config(dump_path="dump")
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

#### dump结果文件介绍dump数据保存路径为 `{dump_path}/msprobe_ge_dump`。其中 `{dump_path}` 为用户通过 `set_ge_dump_config` 接口的'dump_path' 参数传入的路径，`msprobe_ge_dump`为msProbe工具自动创建的目录。

使用Ascend Extension for PyTorch 7.1.0及以上版本时，结果件目录结构为

```
├── ${dump_path}
│   ├── msprobe_ge_dump
│   |   ├── dynamo_optimized_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── dynamo_original_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── worldsize${rank_size}_global_rank${rank_id}
│   |   │   ├── ${time}
|   |   |   |   ├── ${device_id}
|   |   |   |   |   ├── ${model_name}
|   |   |   |   |   |   ├── ${model_id}
|   |   |   |   |   |   |   ├── ${token_id}
└── └── └── └── └── └── └── └── └── # bin格式数据
```

使用Ascend Extension for PyTorch 7.1.0以下版本时，结果件目录结构为

```
├── ${dump_path}
│   ├── msprobe_ge_dump
│   |   ├── dynamo_optimized_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── dynamo_original_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── ${time}
|   |   |   ├── ${device_id}
|   |   |   |   ├── ${model_name}
|   |   |   |   |   ├── ${model_id}
|   |   |   |   |   |   ├── ${token_id}
└── └── └── └── └── └── └── └── # bin格式数据
```

 ### 1.2 FX模式dump数据

调用 `set_fx_dump_config` 接口，获取配置后的 `config` 实例，或在已有 `config` 实例上增加dump配置，配置模型compile，并执行推理。

#### 接口介绍

**功能说明**
调用 `set_fx_dump_config` 接口，获取配置后的 `config` 实例，或在已有 `config` 实例上增加dump配置，配置模型compile，并执行推理。

**接口介绍**

**接口原型**
```Python
set_fx_dump_config(dump_path='', op_list=None, compiler_config=None)
```

**参数说明**

| 参数名          | 参数描述                                             | 是否必选                               |
| --------------- | ---------------------------------------------------- | -------------------------------------- |
| dump_path       | dump数据的存放路径。仅在使用Ascend Extension for PyTorch 7.0.0及以上版本时生效。 | 否（默认为"./"）                         |
| op_list         | 指定算子进行dump。格式：["Add", "Conv_1"] 代表dump Add和Conv_1两层数据。 | 否（默认为None，dump全量数据）           |
| compiler_config | 图编译配置（CompilerConfig对象）。                     | 否（默认为None，返回新创建的图编译配置） |

#### 工具使用
**使用示例（已创建CompilerConfig对象）**

```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_fx_dump_config  # 添加导入
  ...
  model = ...
  config = torchair.CompilerConfig()
  # 在已有config上增加dump配置
  set_fx_dump_config(dump_path="dump", compiler_config=config)
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

**使用示例（未创建CompilerConfig对象）**
```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_fx_dump_config  # 添加导入
  ...
  model = ...
  # 添加获取config
  config = set_fx_dump_config(dump_path="dump")
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

**使用示例（指定算子进行dump）**
```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_fx_dump_config  # 添加导入
  ...
  model = ...
  # 添加获取config
  config = set_fx_dump_config(dump_path="dump", op_list=['Add', 'Conv_1'])
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

#### dump结果文件介绍dump数据保存路径为 `{dump_path}/msprobe_fx_dump`。其中 `{dump_path}` 为用户通过 `set_fx_dump_config` 接口的'dump_path' 参数传入的路径，`msprobe_fx_dump`为msProbe工具自动创建的目录。

使用Ascend Extension for PyTorch 7.1.0及以上版本时，结果件目录结构为

```
├── ${dump_path}
│   ├── msprobe_fx_dump
│   |   ├── worldsize${rank_size}_global_rank${rank_id}
│   |   │   ├── ${model_name}
|   |   |   |   ├── ${token_id}
└── └── └── └── └── └── # npy格式数据
```

使用Ascend Extension for PyTorch 7.1.0版本时，结果件目录结构为

```
├── ${dump_path}
│   ├── msprobe_fx_dump
│   |   ├── data_dump
|   |   |   ├── ${token_id+1}
|   |   |   |   ├── gm_${time}_dump
└── └── └── └── └── └── # npy格式数据
```

使用Ascend Extension for PyTorch 7.1.0以下版本时，结果件目录结构为

```
├── . # 当前工作目录
│   ├── data_dump
|   |   ├── ${token_id+1}
|   |   |   ├── gm_${time}_dump
└── └── └── └── └── # npy格式数据
```

### compare精度比对

执行 `msprobe compare --target_path [GE dump data path] --golden-path [FX dump data path] --output [output path] --mode torchair`，在指定的 `output` 路径下输出比对结果csv文件，若不使用 `--output` 参数，则默认保存在当前目录下。

```sh
# 使用7.1.0及以上版本PTA
msprobe compare --target_path ${dump_path}/msprobe_ge_dump --golden-path ${dump_path}/msprobe_fx_dump --mode torchair
```

```sh
# 使用Ascend Extension for PyTorch 7.1.0版本
msprobe compare --target_path ${dump_path}/msprobe_ge_dump --golden-path ${dump_path}/msprobe_fx_dump/data_dump --mode torchair
```

```sh
# 使用7.1.0以下版本PTA
msprobe compare --target_path ${dump_path}/msprobe_ge_dump --golden-path data_dump --mode torchair
```

**注意**：使用Ascend Extension for PyTorch 7.1.0以下版本时，FX模式dump结果件中的token id目录的目录名比实际token id大1，因此在比对时会将token id目录名减1，作为真实token id。

## GE融合模式（默认）dump数据与GE关闭融合模式dump数据精度比对

### GE融合模式dump数据

同"[1.1 GE融合模式dump数据](#11-ge-融合模式-dump-数据)"小节。

### GE模式关闭融合dump数据

在[**GE融合模式dump数据**](#11-ge-融合模式-dump-数据)的基础上，通过`set_ge_dump_config` 接口的 `fusion_switch_file` 参数传入设置关闭算子融合的配置文件。工具使用示例如下：

- 创建设置关闭算子融合的配置文件 `fusion_switch.json`。算子融合规则的详解介绍可参见《PyTorch图模式使用(torchair)》中的"[算子融合规则配置功能](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00025.html)"章节。

  ```json
  {
    "Switch": {
      "GraphFusion": {
        "ALL": "off"
      },
      "UBFusion": {
        "ALL": "off"
      }
    }
  }
  ```

- 在推理脚本中调用 `set_ge_dump_config` 接口。

  ```py
  # 若用户已创建CompilerConfig对象
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config  # 添加导入
  ...
  model = ...
  config = torchair.CompilerConfig()
  # 在已有config上增加dump配置
  set_ge_dump_config(dump_path="dump_fusion_off",
                     fusion_switch_file="fusion_switch.json", compiler_config=config)
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

  ```py
  # 若用户未创建CompilerConfig对象
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config  # 添加导入
  ...
  model = ...
  # 添加获取config
  config = set_ge_dump_config(dump_path="dump_fusion_off", fusion_switch_file="fusion_switch.json")
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

- dump结果文件的目录结构与含义可见"[1.1.3 dump结果文件介绍](#113-dump-结果文件介绍)"小节。

### compare精度比对

执行 `msprobe compare --target_path [GE dump data path] --golden-path [fusion off GE dump data path] --output [output path] --mode torchair`，在指定的 `output` 路径下输出比对结果csv文件，若不使用 `--output` 参数，则默认保存在当前目录下。

```sh
msprobe compare --target_path ${dump_path in GE dump}/msprobe_ge_dump --golden-path ${dump_path in fusion off GE dump}/msprobe_ge_dump --mode torchair
```

## 结果查看

精度比对结果的字段含义、判定标准与颜色标记等信息，现已收录在文末附录《精度比对结果参数说明》中，可直接在本页查阅，无需再跳转其他文件。

## 附录

### (定向客户提供) 将dump数据转化为指定信息以压缩数据量
- dump过程中生成的数据量可能占用大量磁盘空间，可以在dump过程中启用后台进程，将完整的数据提取为指定的信息。以下参考脚本将数据转化为最大最小值，并删除原数据。
  ```py
  #!/bin/env python3
  import os
  import time
  import argparse
  
  surfix = "_min_max"  # Converted data save surfix
  
  # Define how single data is converted
  def convert_data_to_info(data):
      return [data.min(), data.max()]
  
  def convert(data_path):
      import numpy as np
      from components.utils.acc_cmp import parse_torchair_dump_data
  
      npz_surfix, npy_surfix = "{}.npz".format(surfix), "{}.npy".format(surfix)
      for cur_path, dirs, files in os.walk(data_path):
          for file in files:
              if file.endswith(npy_surfix):  # already converted FX data
                  continue
  
              cur = os.path.join(cur_path, file)
              if file.endswith(".npy"):  # FX saved npy data
                  file_name = os.path.splitext(cur)[0]
                  np.save(file_name + surfix, convert_data_to_info(np.load(cur)))
                  os.remove(cur)
                  print("Converted: {} -> {}{}".format(cur, file_name, npy_surfix))
              elif not file.endswith(npz_surfix) and not file.endswith(".txt") and not file.endswith(".swp"):
                  inputs, outputs = parse_torchair_dump_data(cur)
                  inputs = [convert_data_to_info(ii) for ii in inputs]
                  outputs = [convert_data_to_info(ii) for ii in outputs]
  
                  np.savez(cur + npz_surfix, inputs=inputs, outputs=outputs)
                  os.remove(cur)
                  print("Converted: {} -> {}{}".format(cur, cur, npz_surfix))
  
  if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("data_path", help="GE or FX data dump path")
      args = parser.parse_args()
      while True:
          convert(args.data_path)
          time.sleep(0.5)
          print("Wmsiting...")
  ```

  在dump过程中后台执行该脚本，将dump数据转化为info数据，以减少内存占用。

  ```sh
  # 将msprobe_ge_dump下的GE dump数据转化为info
  python3 convert.py msprobe_ge_dump
  ```

  ```sh
  # 使用Ascend Extension for PyTorch 7.1.0及以上版本时，将msprobe_fx_dump下的FX dump数据转化为info
  python3 convert.py msprobe_fx_dump
  ```

  ```sh
  # 使用Ascend Extension for PyTorch 7.1.0以下版本时，将data_dump下的FX dump数据转化为info
  python3 convert.py data_dump
  ```

### 比对结果文件格式torchair场景下的精度比对结果以CSV文件格式输出，包含以下主要列：

#### 基本信息
- **API Name**: 算子或API名称。
- **Stack Info**: 堆栈信息，用于定位代码位置。
- **Data Name**: 数据名称，格式为 [NPU真实数据名，Bench真实数据名]。

#### 真实数据模式指标
当dump数据模式为真实数据时，包含以下指标：

| 指标名称 | 含义 | 正常范围 |
|---------|------|----------|
| Cosine | 余弦相似度，衡量两个向量的方向相似性 | 0.99-1.0 |
| EucDist | 欧氏距离，衡量两个向量的绝对距离 | 越小越好 |
| MaxAbsErr | 最大绝对误差 | 越小越好 |
| MaxRelativeErr | 最大相对误差 | 一般 < 0.01 |
| One Thousandth Err Ratio | 相对误差小于千分之一的比例 | 越高越好 |
| Five Thousandths Err Ratio | 相对误差小于千分之五的比例 | 越高越好 |
| Requires_grad Consistent | 计算梯度是否一致 | True |

#### 统计数据模式指标
当dump数据模式为统计数据时，包含以下指标：

| 指标名称 | 含义 |
|---------|------|
| Max diff | 最大值差异 |
| Min diff | 最小值差异 |
| Mean diff | 平均值差异 |
| L2norm diff | L2范数差异 |
| MaxRelativeErr | 最大相对误差 |
| MinRelativeErr | 最小相对误差 |
| MeanRelativeErr | 平均相对误差 |
| NormRelativeErr | 范数相对误差 |

#### MD5模式指标
当dump数据模式为MD5时，包含以下指标：

| 指标名称 | 含义 |
|---------|------|
| NPU MD5 | NPU数据CRC-32值 |
| BENCH MD5 | 标杆数据CRC-32值 |

#### 结果判定信息
- **Result**: 比对结果（PASS/FAIL）
- **Accuracy Reached or Not**: 计算精度是否达标（Yes/No）
- **Err_message**: 错误信息提示

### 结果判定标准

#### 真实数据模式判定
- **PASS**: Cosine ≥ 0.99且MaxRelativeErr < 0.01
- **FAIL**: Cosine < 0.99或MaxRelativeErr ≥ 0.01

#### 统计数据模式判定
- **PASS**: 各项差异指标在可接受范围内
- **FAIL**: 存在显著差异

#### MD5模式判定
- **PASS**: NPU MD5 == BENCH MD5
- **FAIL**: NPU MD5 != BENCH MD5

### 颜色标记说明

当开启高亮颜色标记功能时：
- **红色**: 表示精度异常，需要重点关注。
- **黄色**: 表示精度可疑，需要进一步分析。
- **绿色**: 表示精度正常。

### 特殊值处理

- **N/A**: 表示无法计算该比对指标值。
- **NaN**: 表示计算结果为非数字，通常由于数据中存在NaN值。
- **inf**: 表示计算结果为无穷大，通常由于除零操作。

当dump数据中存在0或NaN时，比对结果中最大相对误差可能出现inf或NaN的情况，属于正常现象。

### 结果文件位置

比对结果CSV文件默认保存在当前目录，或通过 --output参数指定的目录中。
