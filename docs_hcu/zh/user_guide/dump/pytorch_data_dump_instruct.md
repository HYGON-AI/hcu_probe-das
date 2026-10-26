# PyTorch场景精度数据采集

## 数据采集功能介绍

### 功能说明

根据需求选择合适的功能(task)类型，task配置项详细说明请参见[config.json介绍](./config_json_introduct.md#参数介绍)。

| 调试需求   | 推荐配置        | 特点                                                                                                                                                                                     |
| ------ | --------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 初步精度分析 | task="statistics"      | 资源占用低，快速获取统计信息                                                                                                                                                                         |
| 深度精度分析 | task="tensor"          | 采集完整数据，支持详细分析                                                                                                                                                                          |
| 确定性问题分析 | task="statistics"<br>summary_mode="md5"          | 采集统计信息和tensor的CRC-32校验值，快速分析确定性问题                                                                                                                                                      |

### 使用示例

#### dump采集基础示例

dump采集基础示例如下，示例代码中的高亮显示为需要在用户脚本中添加的代码。

<details>
<summary><b>单击展开示例</b></summary>

```diff
+# 导入工具数据采集接口，尽量在迭代训练文件导包后的位置执行seed_all和实例化PrecisionDebugger
+from hcuprobe.pytorch import seed_all, PrecisionDebugger
+# 在模型训练开始前固定随机性
+seed_all()
+# PrecisionDebugger实例化，加载dump配置文件
+debugger = PrecisionDebugger(config_path="./config.json")

 # 模型、损失函数的定义及初始化等操作
 # ...

 # 数据集迭代的位置一般为模型训练开始的位置
 for data, label in data_loader:
+    debugger.start(model)  # 开启数据dump

     # 如下是模型每个step执行的逻辑
     output = model(data)
     # ...
     # 计算梯度
     loss.backward()
     # 更新参数
     optimizer.step()

+    debugger.stop()  # 关闭数据dump，可继续开启数据dump，采集数据会记录在同一个step中
+    debugger.step()  # 结束数据dump，若继续开启数据dump，采集数据将记录在下一个step中
```

</details><br>

示例接口介绍：

- [seed_all](#seed_all)：用于固定网络中的随机性和开启确定性计算。
- [PrecisionDebugger](#precisiondebugger)：通过加载dump配置文件的方式来确定dump操作的详细配置。
- [start](#start)：启动精度数据采集。
- [stop](#stop)：停止精度数据采集。
- [step](#step)：结束一个step的数据采集，完成所有数据落盘并更新dump参数。

更多接口介绍请参见[接口介绍](#接口介绍)章节。

**注意事项**

- `PrecisionDebugger` 提供 dump 动态启停能力，通过配置文件中的 `dump_enable` 字段进行控制：在同一训练/推理任务中可按需动态开启或关闭 dump。详细说明请参见 [config.json介绍](./config_json_introduct.md) 中的 `dump_enable` 参数说明。

- 本工具提供固定的 API 支持列表，若需要删除或增加 dump 的 API，可以在 [support_wrap_ops.yaml](../../../../python/hcuprobe/pytorch/dump/api_dump/support_wrap_ops.yaml) 文件内手动修改，如下示例：

  ```yaml
  functional:  # functional为算子类别，找到对应的类别，在该类别下按照下列格式删除或添加API
    - conv1d
    - conv2d
    - conv3d
  ```

#### 采集函数模块化数据

采集函数模块化数据示例如下，示例代码中的高亮显示为需要在用户脚本中添加的代码。

<details>
<summary><b>单击展开示例</b></summary>

```diff
 # 根据需要import包
 import torch
 import torch.nn as nn
 import torch.nn.functional as F

+# 导入工具的数据采集接口
+from hcuprobe.pytorch import PrecisionDebugger, module_dump, module_dump_end

+# 在模型训练开始前实例化PrecisionDebugger
+debugger = PrecisionDebugger(config_path='./config.json')

 # 定义网络
 class ModuleOP(nn.Module):
     def __init__(self) -> None:
         super().__init__()
         self.linear_1 = nn.Linear(in_features=8, out_features=4)
         self.linear_2 = nn.Linear(in_features=4, out_features=2)

     def forward(self, x):
         x1 = self.linear_1(x)
         x2 = self.linear_2(x1)
         r1 = F.relu(x2)
         return r1

 if __name__ == "__main__":
     module = ModuleOP()

+    debugger.start()  # 开启数据dump

     x = torch.randn(10, 8)
     # ...                              # start和module_dump接口之间的数据正常dump
+    module_dump(module, "MyModuleOP")  # 开启模块级精度数据dump
     out = module(x)  # module内部的child modules或API将不会被dump
+    module_dump_end()  # 关闭模块级精度数据dump
     loss = out.sum()  # module_dump_end和stop接口之间的数据正常dump
     loss.backward()

+    debugger.stop()  # 关闭数据dump，可继续开启数据dump，采集数据会记录在同一个step中
+    debugger.step()  # 结束数据dump，若继续开启数据dump，采集数据将记录在下一个step中
```

</details><br>

#### 跨文件采集数据

跨文件采集数据示例如下，示例代码中的高亮显示为需要在用户脚本中添加的代码。

<details>
<summary><b>单击展开示例</b></summary>

为了确保所有API都被工具封装，PrecisionDebugger的实例化通常放在训练工程的入口位置，但有的时候，模型定义会在另一个文件中。

假设有两个文件，`train.py`（训练工程入口文件）和`module.py`（模型定义文件），为了采集module.py中定义的ModuleOP模块中某些子模块或API的前反向数据，需要在`train.py`和`module.py`文件中分别导入PrecisionDebugger并进行如下配置。

train.py文件：

```diff
 # 根据需要import包
 import torch
 from module import ModuleOP

+# 导入工具的数据采集接口
+from hcuprobe.pytorch import PrecisionDebugger

+# 将PrecisionDebugger的实例化放在文件的开始位置，即导包后的位置，确保所有API都被封装
+debugger = PrecisionDebugger(config_path='./config.json')

 if __name__ == "__main__":
     module = ModuleOP()

     x = torch.randn(10, 8)
     out = module(x)
     loss = out.sum()
     loss.backward()
```

module.py文件：

```diff
 import torch
 import torch.nn as nn
 import torch.nn.functional as F

+from hcuprobe.pytorch import PrecisionDebugger

 # 定义网络
 class ModuleOP(nn.Module):
     def __init__(self) -> None:
         super().__init__()
         self.linear_1 = nn.Linear(in_features=8, out_features=4)
         self.linear_2 = nn.Linear(in_features=4, out_features=2)

     def forward(self, x):
+        PrecisionDebugger.start()
         x1 = self.linear_1(x)
+        PrecisionDebugger.stop()
         x2 = self.linear_2(x1)
         r1 = F.relu(x2)
         return r1

```

</details><br>

#### 快速验证
可通过 Claude Code Skill 根据用户自动插入所需的采集代码，快速完成采集功能的接入与验证。使用说明详见：
```text
.claude/skills/hcu_probe-pytorch-dump/SKILL.md
```


## 接口介绍

### PrecisionDebugger

**功能说明**

通过加载dump配置文件的方式来确定dump操作的详细配置。此接口的初始化需与采集目标在同一个进程中，否则将无法采集目标数据。

**函数原型**

```Python
debugger = PrecisionDebugger(config_path=None, task=None, dump_path=None, level=None, step=None)
```

**参数说明**

- config_path：可选参数，str类型，指定dump配置文件路径，参数示例："./config.json"。未配置该路径时，默认使用[config.json](../../../../python/msprobe/config.json)
  文件的默认配置，配置选项含义可见[config.json介绍](./config_json_introduct.md)。
- 其他参数均在config.json文件中可配，详细配置可见[config.json介绍](./config_json_introduct.md)。
- 此接口的参数均不是必要的（均不配置的情况下默认采集所有 rank 和 step 的 L1 级别的统计数据），且优先级高于config.json
  文件中的配置，但可配置的参数相比config.json较少。

**返回值说明**

debugger是PrecisionDebugger类的实例对象，该实例对象提供**start**、**stop**和**step**等函数开启和结束数据采集的功能。

**调用示例**

请参见[dump采集基础示例](#dump采集基础示例)。

### start

**功能说明**

启动精度数据采集。在模型初始化之后的位置添加。需要与**stop**和**step**等函数一起搭配使用。

**函数原型**

```Python
debugger.start(model=None, token_range=None, rank_id=None, scheduled_tokens=None)
```

**参数说明**

- model：指定需要采集Module级数据的模型，支持传入torch.nn.Module、list[torch.nn.Module]或Tuple[torch.nn.Module]类型，默认未配置。

  level配置为"L0"、"mix"或token_range不为None时，必须在该接口中配置该参数。

  对于复杂模型，如果仅需要监测一部分（如model.A，model.A extends torch.nn.Module），传入需要监测的部分（如model.A）即可。

  注意：传入的当前层不会被dump，工具只会dump传入层的子层级。如传入了model.A，A本身不会被dump，而是会dump A.x、A.x.xx等。

- token_range：指定推理模型采集时的token循环始末范围，支持传入[int, int]类型，代表[start, end]，范围包含边界，默认未配置。

- rank_id: 指定自定义的rank ID，支持传入大于等于0的整数。默认未配置，则工具基于torch.distributed.get_rank接口获取rank ID；
  配置此参数后，dump的结果中，rank文件夹名称中的{ID}将使用该参数所配置的值。

  注意：通常情况下，用户无需手动配置rank_id参数，工具默认通过torch.distributed.get_rank接口（下面简称get_rank接口）可自动获取多卡多进程的唯一rank ID；
  然而，在某些特殊场景下，get_rank接口可能无法正确获取唯一的rank ID。例如，在推理框架sglang的DP推理场景中，各DP worker之间是独立的分布式集群，导致get_rank接口返回重复的rank ID，进而引发dump结果中rank文件夹同名覆盖的问题，造成dump数据丢失。

  针对此类特殊场景，可通过配置rank_id参数为rank文件夹命名，但需要保证rank_id在各个进程中唯一。该值通常可在模型脚本或训练推理框架中获取，例如推理框架sglang中的self.gpu_id，其在每个进程中均保持唯一性。

  配置示例：`debugger.start(rank_id=self.gpu_id)`

- scheduled_tokens: 指定当前调度每个请求占用的token数，dict类型。key是str，表示请求id，value是int，value>=0，表示该请求占用的token数。配置示例：`debugger.start(scheduled_tokens={"chatcmpl-req-0": 10, "chatcmpl-req-1": 15})`。
  配合[request_id配置](./config_json_introduct.md#request_id参数配置说明)可以实现指定请求id采集数据。

**返回值说明**

无

**调用示例**

请参见[dump采集基础示例](#dump采集基础示例)。

### stop

**功能说明**

停止精度数据采集。在**start**接口之后的任意位置添加。

若**stop**函数添加在反向计算代码（如loss.backward）之后，则会采集**start**和该函数之间的前反向数据。

若**stop**函数添加在反向计算代码之前，则需要将**step**函数添加到反向计算代码之后，才能采集**start**
和该函数之间的前反向数据。

**注意事项**

**stop**函数必须调用，否则可能导致精度数据落盘不全。

**函数原型**

```Python
debugger.stop()
```

**返回值说明**

无

**调用示例**

请参见[dump采集基础示例](#dump采集基础示例)。

### step

**功能说明**

结束一个step的数据采集，完成所有数据落盘并更新dump参数。在一个step结束的位置添加，且必须在**stop**函数之后的位置调用。

**注意事项**

该函数需要配合**start**和**stop**函数使用，尽量添加在反向计算代码（如loss.backward）之后，否则可能会导致反向数据丢失。

**函数原型**

```Python
debugger.step()
```

**返回值说明**

无

**调用示例**

请参见[dump采集基础示例](#dump采集基础示例)。

### module_dump

**功能说明**
开启模块级精度数据dump。该接口为函数模块化接口，即只会dump输入的模块数据，不会dump子模块和模块内API的数据。

**注意事项**

该函数需要配合start、stop和step等接口使用。

**函数原型**

```Python
module_dump(module, module_name)
```

**参数说明**

- module：网络中实例化好的nn.Module类对象。torch.nn.Module类型，必选。
- module_name：用户自定义该module名称，主要用于dump数据的命名。str类型，必选。

**返回值说明**

无

**调用示例**

请参见[采集函数模块化数据](#采集函数模块化数据)。

### module_dump_end

**功能说明**

结束模块级精度数据dump。在module_dump和module_dump_end之间的API或者Module，除了传入的module，其他数据均不会被dump。
在执行完module_dump_end后数据恢复正常的dump模式。

**函数原型**

```Python
module_dump_end()
```

**返回值说明**

无

**调用示例**

请参见[采集函数模块化数据](#采集函数模块化数据)。

### save

**功能说明**

单点保存网络执行过程中前反向数值，并以统计值数据或张量文件落盘。

**函数原型**

```python
save(variable, name, save_backward=True)
```

**参数说明**

| 参数名称      | 可选/必选 | 参数含义                                                     |
| ------------- | --------- | ------------------------------------------------------------ |
| variable      | 必选      | 需要保存的变量，支持数据类型：dict、list、tuple、torch.tensor、int、float、str。 |
| name          | 必选      | 指定的名称，支持数据类型：str。                              |
| save_backward | 可选      | 是否保存反向数据，支持数据类型：boolean。                    |

**返回值说明**

无

**调用示例**

请参见[单点保存工具使用介绍](./debugger_save_instruct.md)。

### set_init_step

**功能说明**

设置起始step数，step数默认从0开始计数，使用该接口后step从指定值开始计数。该函数需要写在训练迭代的循环开始前，不能写在循环内。

**函数原型**

```Python
debugger.set_init_step(step)
```

**参数说明**

step：指定的起始step数。

**返回值说明**

无

**调用示例**

无

### register_custom_api

**功能说明**

注册用户自定义的API到工具用于API级数据dump。

**函数原型**

```Python
debugger.register_custom_api(module, api, api_prefix)
```

**参数说明**

以torch.matmul api为例

- module：必选，API所属的包，即传入torch。
- api：必选，API的名称，str类型，即传入"matmul"。
- api_prefix：可选，[dump.json](#dumpjson文件说明)中api名的前缀，默认为包名的字符串格式，即"torch"。

**返回值说明**

无

**调用示例**

```diff
+# 需要导入自定义算子的模块
+import module_a
+from msprobe.pytorch import seed_all, PrecisionDebugger
+seed_all()
+debugger = PrecisionDebugger(config_path="./config.json")
+# 假设自定义算子的调用方式是module_a.compute(args)，注册方式如下。
+debugger.register_custom_api(module = module_a, api = "compute")

 # 模型、损失函数的定义及初始化等操作
 # ...

 # 数据集迭代的位置一般为模型训练开始的位置
 for data, label in data_loader:
+    debugger.start(model)  # 开启数据dump

     # 如下是模型每个step执行的逻辑
     output = model(data)
     # ...
     # 计算梯度
     loss.backward()
     # 更新参数
     optimizer.step()

+    debugger.stop()  # 关闭数据dump，可继续开启数据dump，采集数据会记录在同一个step中
+    debugger.step()  # 结束数据dump，若继续开启数据dump，采集数据将记录在下一个step中

+# 取消该API的dump，根据实际情况按需使用。
+debugger.restore_custom_api(module_a, "compute")
```

### restore_custom_api

**功能说明**

恢复用户原有的自定义的API，取消该API的dump。

**函数原型**

```Python
debugger.restore_custom_api(module, api)
```

**参数说明**

以torch.matmul api为例

- module：必选，API所属的包，即传入torch。
- api：必选，API的名称，str类型，即传入"matmul"。

**返回值说明**

无

**调用示例**

请参见[register_custom_api](#register_custom_api)。

### seed_all

**功能说明**

用于控制模型网络中的随机性，支持固定随机种子以及启用确定性计算，以确保实验的可复现性。

**函数原型**

```python
seed_all(seed=1234, mode=False, rm_dropout=False, is_enhanced=False)
```

**参数说明**

- seed：随机性种子。参数示例：seed=1000。默认为1234，非必选。
- mode：确定性计算模式。可配置True或False。参数示例：mode=True。默认为False。非必选（注意：确定性计算会导致API执行性能降低，建议在发现模型多次执行结果不同的情况下开启）。
- rm_dropout：控制dropout失效的开关。可配置True或False，默认为False，非必选。参数示例：rm_dropout=True。
  该参数设置为True后，工具会自动将`torch.nn.functional.dropout`、`torch.nn.functional.dropout2d`、`torch.nn.functional.dropout3d`、`torch.nn.Dropout`、`torch.nn.Dropout2d`、`torch.nn.Dropout3d`
  的接口参数p置为0，以避免因随机dropout造成的网络随机性。注意：通过rm_dropout控制dropout失效需要在初始化dropout实例前调用才能生效。
- is_enhanced：增强随机性固定的开关。可配置True或False，默认为False，非必选。参数示例：is_enhanced=True。开启该功能后，将进一步固定PyTorch、NumPy以及Python内置随机数生成器的状态。在同一个进程或不同进程中多次执行相同的随机性API，每次生成的随机值都完全相同。这有助于在更复杂的随机场景下实现严格的可复现性。

**返回值说明**

无

**调用示例**

```diff
 import torch
+from msprobe.pytorch import seed_all

 # seed_all仅固定随机种子和开启确定性计算。
+seed_all(seed=1234, mode=True)
 num1 = torch.mean(torch.randn(2,2))
 print(num1)  # tensor(-0.0866)
 num2 = torch.mean(torch.randn(2,2))
 print(num2)  # tensor(0.2038)
```

上述脚本重复执行多次，每次执行结果num1和num2是能对应的，但该脚本内执行两次相同的随机性API，生成的结果是不一样的，即num1不等于num2。

```diff
 import torch
+from msprobe.pytorch import seed_all

 # seed_all固定随机种子和开启确定性计算，并开启随机数固定增强功能。
+seed_all(seed=1234, mode=True, is_enhanced=True)
 num1 = torch.mean(torch.randn(2,2))
 print(num1)  # tensor(-0.0866)
 num2 = torch.mean(torch.randn(2,2))
 print(num2)  # tensor(-0.0866)
```

上述脚本中执行两次相同的随机性API，生成的结果能完全相同，即使该脚本多次重复执行也能确保两个值完全相同。

**其他说明**

当前工具dump功能默认不会固定随机性，若希望每次采集的数据保持一致，建议在dump数据前调用seed_all接口。

seed_all函数可固定随机数的范围如下表：

| API                                      | 固定随机数                                      |
|------------------------------------------|--------------------------------------------|
| os.environ['PYTHONHASHSEED'] = str(seed) | 禁止Python中的hash随机化                          |
| os.environ['HCCL_DETERMINISTIC'] = True  | 固定通信算子计算的确定性                               |
| np.random.seed(seed)                     | 设置numpy中随机生成器的种子                           |
| torch.manual_seed(seed)                  | 设置当前CPU的随机种子                               |
| torch.cuda.manual_seed(seed)             | 设置当前GPU的随机种子                               |
| torch.cuda.manual_seed_all(seed)         | 设置所有GPU的随机种子                               |
| torch_npu.npu.manual_seed(seed)          | 设置当前NPU的随机种子                               |
| torch_npu.npu.manual_seed_all(seed)      | 设置所有NPU的随机种子                               |
| torch.use_deterministic_algorithms(True) | CUDA/CANN使能确定性计算（注意mode为True时才会调用该方法开启确定性） |
| torch.backends.cudnn.enable=False        | 关闭cuDNN                                    |
| torch.backends.cudnn.benchmark=False     | cuDNN确定性地选择算法                              |
| torch.backends.cudnn.deterministic=True  | cuDNN仅使用确定性的卷积算法                           |
| torch.nn.functional.dropout              | 将dropout的接口参数p置为0                          |
| torch.nn.functional.dropout2d            | 将dropout2d的接口参数p置为0                        |
| torch.nn.functional.dropout3d            | 将dropout3d的接口参数p置为0                        |
| torch.nn.Dropout                         | 将Dropout的接口参数p置为0                          |
| torch.nn.Dropout2d                       | 将Dropout2d的接口参数p置为0                        |
| torch.nn.Dropout3d                       | 将Dropout3d的接口参数p置为0                        |

需要保证CPU或GPU以及NPU的模型输入完全一致，dump数据的比对才有意义，seed_all并不能保证模型输入完全一致，如下表所示场景需要保证输入的一致性。

| 场景          | 固定方法       |
|-------------|------------|
| 数据集的shuffle | 关闭shuffle。 |

关闭shuffle示例如下：

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)
```

### random_save

**功能说明**

寻找模型脚本中所有随机性API的调用位置，并将结果输出到csv表格，帮助用户快速定位模型中的随机数生成点。

**函数原型**

```python
random_save(output_path="./output")
```

**参数说明**

output_path：指定随机性API的保存路径。可选参数，str类型，默认值为"./output"，即默认保存在当前路径的output目录下。参数示例：output_path="./output"。

**返回值说明**

无

**调用示例**

请参见[采集随机性API](#采集随机性api)。

## 补充说明

dump "statistics"模式的性能膨胀大小与"tensor"模式采集的数据量大小，可以参考[dump基线](../../baseline/pytorch_data_dump_perf_baseline.md)。
