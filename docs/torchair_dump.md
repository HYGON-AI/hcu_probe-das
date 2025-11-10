# 基于 torch 图模式（torchair）推理场景

- 在跑推理之前需要确认 torchvision 版本与 torch 版本是否匹配。torch 版本与 torchvision 版本的匹配关系如下：
  | torch 版本 | torchvision 版本 |
  | ---------- | ---------------- |
  | 2.3.0      | 0.18.0           |
  | 2.2.0      | 0.17.0           |
  | 2.1.0      | 0.16.0           |
  | 2.0.0      | 0.15.1           |

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

| 参数名             | 参数描述                                                                          | 是否必选                               |
| ------------------ | --------------------------------------------------------------------------------- | -------------------------------------- |
| dump_path          | dump数据的存放路径                                                                | 否(默认为"./")                         |
| dump_mode          | data dump模式，用于指定dump算子输入还是输出数据。可选值有"input", "output", "all" | 否(默认为"all"，dump输入与输出数据)    |
| fusion_switch_file | 是否关闭融合dump功能                                                              | 否(默认为None，开启融合)               |
| dump_token         | 指定token进行dump。格式：[1,2,5] 代表dump第1、2、5个token数据                     | 否(默认为None，dump全量数据)           |
| dump_layer         | 指定layer进行dump。格式：["Add", "Conv_1"] 代表dump Add和Conv_1两层数据           | 否(默认为None，dump全量数据)           |
| compiler_config    | 图编译配置（CompilerConfig对象）                                                  | 否(默认为None，返回新创建的图编译配置) |

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

dump 数据保存路径为 `{dump_path}/msit_ge_dump`。其中 `{dump_path}` 为用户通过 `set_ge_dump_config` 接口的'dump_path' 参数传入的路径，`msit_ge_dump`为 msit 工具自动创建的目录。

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

| 参数名          | 参数描述                                             | 是否必选                               |
| --------------- | ---------------------------------------------------- | -------------------------------------- |
| dump_path       | dump数据的存放路径。仅在使用7.0.0以上版本的PTA时生效 | 否(默认为"./")                         |
| compiler_config | 图编译配置（CompilerConfig对象）                     | 否(默认为None，返回新创建的图编译配置) |

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

dump 数据保存路径为 `{dump_path}/msit_fx_dump`。其中 `{dump_path}` 为用户通过 `set_fx_dump_config` 接口的'dump_path' 参数传入的路径，`msit_fx_dump`为 msit 工具自动创建的目录。

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

### 2. GE 模式关闭融合 dump 数据

在[**GE 融合模式 dump 数据**](#11-ge-融合模式-dump-数据)的基础上，通过 `set_ge_dump_config` 接口的 `fusion_switch_file` 参数传入设置关闭算子融合的配置文件。工具使用示例如下：

- 创建设置关闭算子融合的配置文件 `fusion_switch.json`。算子融合规则的详解介绍可参见《PyTorch图模式使用(TorchAir)》中的“[算子融合规则配置功能](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00025.html)”章节。

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
- dump 结果文件的目录结构与含义可见“[1.1.3 dump 结果文件介绍](#113-dump-结果文件介绍)”小节。

