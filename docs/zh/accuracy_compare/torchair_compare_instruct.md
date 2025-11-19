# 基于 torch 图模式（torchair）整网算子精度比对

- 在跑推理之前需要确认 torchvision 版本与 torch 版本是否匹配。torch 版本与 torchvision 版本的匹配关系如下：

  | torch 版本 | torchvision 版本 | 
  |---------|---------------|
  | 2.3.0   | 0.18.0        |
  | 2.2.0   | 0.17.0        |
  | 2.1.0   | 0.16.0        | 
  | 2.0.0   | 0.15.1        | 

- 两次 GE dump 或两次 FX dump 间，请指定不同的 dump 数据保存路径，否则会导致数据混乱，无法区分的问题，从而影响数据比对、分析。

## 1. GE 融合模式（默认） dump 数据与 FX dump 数据精度比对

- **GE**: Graph Engine，基于昇腾 AI 软件栈对不同的机器学习框架提供统一的IR接口，对接上层网络模型框架。
- **FX**：功能类似于 PyTorch 框架的 FX 工具包，用于消除动态图和静态图之间的 gap，使我们对于 nn.Module 的各种操作变得更加简单。

### 1.1 GE 融合模式 dump 数据

调用 `set_ge_dump_config` 接口，获取配置后的 `config` 实例，或在已有 `config` 实例上增加 dump 配置，配置模型 compile，并执行推理。

#### 1.1.1 接口介绍

**接口原型**：

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

**参数列表**：

  | 参数名                | 参数描述                                               | 是否必选                |
  |--------------------|----------------------------------------------------|---------------------|
  | dump_path          | dump数据的存放路径                                        | 否(默认为"./")                   |
  | dump_mode         | data dump模式，用于指定dump算子输入还是输出数据。可选值有"input", "output", "all" | 否(默认为"all"，dump输入与输出数据)                   |
  | fusion_switch_file | 是否关闭融合dump功能                                       | 否(默认为None，开启融合)    | 
  | dump_token         | 指定token进行dump。格式：[1,2,5] 代表dump第1、2、5个token数据      | 否(默认为None，dump全量数据) | 
  | dump_layer         | 指定layer进行dump。格式：["Add", "Conv_1"] 代表dump Add和Conv_1两层数据 | 否(默认为None，dump全量数据) | 
  | compiler_config    | 图编译配置（CompilerConfig对象） | 否(默认为None，返回新创建的图编译配置) | 

#### 1.1.2 工具使用

  ```py
  # 若用户已创建 CompilerConfig 对象

  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config  # 添加导入
  ...
  model = ...
  config = torchair.CompilerConfig()
  # 在已有 config 上增加 dump 配置
  set_ge_dump_config(dump_path="dump", compiler_config=config)
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

  ```py
  # 若用户未创建 CompilerConfig 对象

  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config  # 添加导入
  ...
  model = ...
  # 添加获取 config
  config = set_ge_dump_config(dump_path="dump")
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

#### 1.1.3 dump 结果文件介绍

dump 数据保存路径为 `{dump_path}/msit_ge_dump`。其中 `{dump_path}` 为用户通过 `set_ge_dump_config` 接口的'dump_path' 参数传入的路径，`msit_ge_dump`为msProbe工具自动创建的目录。

使用 7.1.0 以上版本 PTA 时，结果件目录结构为

```
├── ${dump_path}
│   ├── msit_ge_dump
│   |   ├── dynamo_optimized_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── dynamo_original_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── worldsize${rank_size}_global_rank${rank_id}
│   |   │   ├── ${time}
|   |   |   |   ├── ${device_id}
|   |   |   |   |   ├── ${model_name}
|   |   |   |   |   |   ├── ${model_id}
|   |   |   |   |   |   |   ├── ${token_id}
└── └── └── └── └── └── └── └── └── # bin 格式数据
```

使用 7.1.0 及以下版本 PTA 时，结果件目录结构为

```
├── ${dump_path}
│   ├── msit_ge_dump
│   |   ├── dynamo_optimized_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── dynamo_original_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── ${time}
|   |   |   ├── ${device_id}
|   |   |   |   ├── ${model_name}
|   |   |   |   |   ├── ${model_id}
|   |   |   |   |   |   ├── ${token_id}
└── └── └── └── └── └── └── └── # bin 格式数据
```

 ### 1.2 FX 模式 dump 数据

调用 `set_fx_dump_config` 接口，获取配置后的 `config` 实例，或在已有 `config` 实例上增加 dump 配置，配置模型 compile，并执行推理。

#### 1.2.1 接口介绍

**接口原型**：

```Python
set_fx_dump_config(dump_path='', compiler_config=None)
```

**参数列表**：

  | 参数名                | 参数描述                                               | 是否必选                |
  |--------------------|----------------------------------------------------|---------------------|
  | dump_path          | dump数据的存放路径。仅在使用7.0.0以上版本的PTA时生效 | 否(默认为"./")                   |
  | compiler_config    | 图编译配置（CompilerConfig对象） | 否(默认为None，返回新创建的图编译配置) | 

#### 1.2.2 工具使用

  ```py
  # 若用户已创建 CompilerConfig 对象

  import torch, torch_npu, torchair
  from msprobe.pytorch import set_fx_dump_config  # 添加导入
  ...
  model = ...
  config = torchair.CompilerConfig()
  # 在已有 config 上增加 dump 配置
  set_fx_dump_config(dump_path="dump", compiler_config=config)
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

  ```py
  # 若用户未创建 CompilerConfig 对象

  import torch, torch_npu, torchair
  from msprobe.pytorch import set_fx_dump_config  # 添加导入
  ...
  model = ...
  # 添加获取 config
  config = set_fx_dump_config(dump_path="dump")
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

#### 1.2.3 dump 结果文件介绍

dump 数据保存路径为 `{dump_path}/msit_fx_dump`。其中 `{dump_path}` 为用户通过 `set_fx_dump_config` 接口的'dump_path' 参数传入的路径，`msit_fx_dump`为msProbe工具自动创建的目录。

使用 7.1.0 以上版本 PTA 时，结果件目录结构为

```
├── ${dump_path}
│   ├── msit_fx_dump
│   |   ├── worldsize${rank_size}_global_rank${rank_id}
│   |   │   ├── ${model_name}
|   |   |   |   ├── ${token_id}
└── └── └── └── └── └── # npy 格式数据
```

使用 7.1.0 版本 PTA 时，结果件目录结构为

```
├── ${dump_path}
│   ├── msit_fx_dump
│   |   ├── data_dump
|   |   |   ├── ${token_id+1}
|   |   |   |   ├── gm_${time}_dump
└── └── └── └── └── └── # npy 格式数据
```

使用 7.1.0 以下版本 PTA 时，结果件目录结构为

```
├── . # 当前工作目录
│   ├── data_dump
|   |   ├── ${token_id+1}
|   |   |   ├── gm_${time}_dump
└── └── └── └── └── # npy 格式数据
```

### 1.3 compare 精度比对

执行 `msprobe compare --target_path [GE dump data path] --golden-path [FX dump data path] --output [output path] --mode torchair`，在指定的 `output` 路径下输出比对结果 csv 文件，若不使用 `--output` 参数，则默认保存在当前目录下。

```sh
# 使用 7.1.0 以上版本 PTA
msprobe compare --target_path ${dump_path}/msprobe_ge_dump --golden-path ${dump_path}/msprobe_fx_dump --mode torchair
```

```sh
# 使用 7.1.0 版本 PTA
msprobe compare --target_path ${dump_path}/msprobe_ge_dump --golden-path ${dump_path}/msprobe_fx_dump/data_dump --mode torchair
```

```sh
# 使用 7.1.0 以下版本 PTA
msprobe compare --target_path ${dump_path}/msprobe_ge_dump --golden-path data_dump --mode torchair
```

**注意**：使用 7.1.0 及以下版本 PTA 时，FX 模式 dump 结果件中的 token id 目录的目录名比实际 token id 大 1，因此在比对时会将 token id 目录名减 1，作为真实 token id。

## 2. GE融合模式（默认）dump 数据与GE关闭融合模式 dump 数据精度比对

### 2.1 GE 融合模式 dump 数据

同"[1.1 GE 融合模式 dump 数据](#11-ge-融合模式-dump-数据)"小节。

### 2.2 GE 模式关闭融合 dump 数据

在[**GE 融合模式 dump 数据**](#11-ge-融合模式-dump-数据)的基础上，通过`set_ge_dump_config` 接口的 `fusion_switch_file` 参数传入设置关闭算子融合的配置文件。工具使用示例如下：

- 创建设置关闭算子融合的配置文件 `fusion_switch.json`。算子融合规则的详解介绍可参见《PyTorch图模式使用(TorchAir)》中的"[算子融合规则配置功能](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00025.html)"章节。

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
  # 若用户已创建 CompilerConfig 对象

  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config  # 添加导入
  ...
  model = ...
  config = torchair.CompilerConfig()
  # 在已有 config 上增加 dump 配置
  set_ge_dump_config(dump_path="dump_fusion_off",
                                   fusion_switch_file="fusion_switch.json", compiler_config=config)
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

  ```py
  # 若用户未创建 CompilerConfig 对象

  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config  # 添加导入
  ...
  model = ...
  # 添加获取 config
  config = set_ge_dump_config(dump_path="dump_fusion_off", fusion_switch_file="fusion_switch.json")
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

- dump 结果文件的目录结构与含义可见"[1.1.3 dump 结果文件介绍](#113-dump-结果文件介绍)"小节。

### 2.3 compare 精度比对

执行 `msprobe compare --target_path [GE dump data path] --golden-path [fusion off GE dump data path] --output [output path] --mode torchair`，在指定的 `output` 路径下输出比对结果 csv 文件，若不使用 `--output` 参数，则默认保存在当前目录下。

```sh
msprobe compare --target_path ${dump_path in GE dump}/msprobe_ge_dump --golden-path ${dump_path in fusion off GE dump}/msprobe_ge_dump --mode torchair
```

## 3. 结果查看

精度比对结果的字段含义、判定标准与颜色标记等信息，现已收录在文末附录《精度比对结果参数说明》中，可直接在本页查阅，无需再跳转其他文件。

## 4. 附录

### (定向客户提供) 将 dump 数据转化为指定信息以压缩数据量
- dump 过程中生成的数据量可能占用大量磁盘空间，可以在 dump 过程中启用后台进程，将完整的数据提取为指定的信息。以下参考脚本将数据转化为最大最小值，并删除原数据
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

  在 dump 过程中后台执行该脚本，将 dump 数据转化为 info 数据，以减少内存占用

  ```sh
  # 将 msprobe_ge_dump 下的 GE dump 数据转化为 info
  python3 convert.py msprobe_ge_dump
  ```

  ```sh
  # 使用 7.1.0 及以上版本 PTA时，将 msprobe_fx_dump 下的 FX dump 数据转化为 info
  python3 convert.py msprobe_fx_dump
  ```

  ```sh
  # 使用 7.1.0 以下 PTA时，将 data_dump 下的 FX dump 数据转化为 info
  python3 convert.py data_dump
  ```
## 附录：精度比对结果参数说明

### 1. 比对结果文件格式

TorchAir 场景下的精度比对结果以 CSV 文件格式输出，包含以下主要列：

#### 1.1 基本信息
- **API Name**: 算子或 API 名称
- **Stack Info**: 堆栈信息，用于定位代码位置
- **Data Name**: 数据名称，格式为 [NPU真实数据名，Bench真实数据名]

#### 1.2 真实数据模式指标
当 dump 数据模式为真实数据时，包含以下指标：

| 指标名称 | 含义 | 正常范围 |
|---------|------|----------|
| Cosine | 余弦相似度，衡量两个向量的方向相似性 | 0.99-1.0 |
| EucDist | 欧氏距离，衡量两个向量的绝对距离 | 越小越好 |
| MaxAbsErr | 最大绝对误差 | 越小越好 |
| MaxRelativeErr | 最大相对误差 | 一般 < 0.01 |
| One Thousandth Err Ratio | 相对误差小于千分之一的比例 | 越高越好 |
| Five Thousandths Err Ratio | 相对误差小于千分之五的比例 | 越高越好 |
| Requires_grad Consistent | 计算梯度是否一致 | True |

#### 1.3 统计数据模式指标
当 dump 数据模式为统计数据时，包含以下指标：

| 指标名称 | 含义 |
|---------|------|
| Max diff | 最大值差异 |
| Min diff | 最小值差异 |
| Mean diff | 平均值差异 |
| L2norm diff | L2 范数差异 |
| MaxRelativeErr | 最大相对误差 |
| MinRelativeErr | 最小相对误差 |
| MeanRelativeErr | 平均相对误差 |
| NormRelativeErr | 范数相对误差 |

#### 1.4 MD5 模式指标
当 dump 数据模式为 MD5 时，包含以下指标：

| 指标名称 | 含义 |
|---------|------|
| NPU MD5 | NPU 数据 CRC-32 值 |
| BENCH MD5 | 标杆数据 CRC-32 值 |

#### 1.5 结果判定信息
- **Result**: 比对结果（PASS/FAIL）
- **Accuracy Reached or Not**: 计算精度是否达标（Yes/No）
- **Err_message**: 错误信息提示

### 2. 结果判定标准

#### 2.1 真实数据模式判定
- **PASS**: Cosine ≥ 0.99 且 MaxRelativeErr < 0.01
- **FAIL**: Cosine < 0.99 或 MaxRelativeErr ≥ 0.01

#### 2.2 统计数据模式判定
- **PASS**: 各项差异指标在可接受范围内
- **FAIL**: 存在显著差异

#### 2.3 MD5 模式判定
- **PASS**: NPU MD5 == BENCH MD5
- **FAIL**: NPU MD5 != BENCH MD5

### 3. 颜色标记说明

当开启高亮颜色标记功能时：
- **红色**: 表示精度异常，需要重点关注
- **黄色**: 表示精度可疑，需要进一步分析
- **绿色**: 表示精度正常

### 4. 特殊值处理

- **N/A**: 表示无法计算该比对指标值
- **NaN**: 表示计算结果为非数字，通常由于数据中存在 NaN 值
- **inf**: 表示计算结果为无穷大，通常由于除零操作

当 dump 数据中存在 0 或 NaN 时，比对结果中最大相对误差可能出现 inf 或 NaN 的情况，属于正常现象。

### 5. 结果文件位置

比对结果 CSV 文件默认保存在当前目录，或通过 --output 参数指定的目录中。
