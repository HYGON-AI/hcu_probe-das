# PyTorch场景精度数据采集

## 数据采集功能介绍

### 功能说明

根据需求选择合适的功能(task)类型，task配置项详细说明请参见[config.json介绍](./config_json_introduct.md#参数介绍)。

| 调试需求   | 推荐配置        | 特点                                                                                                                                                                                     |
| ------ | --------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 初步精度分析 | task="statistics"      | 资源占用低，快速获取统计信息                                                                                                                                                                         |
| 深度精度分析 | task="tensor"          | 采集完整数据，支持详细分析                                                                                                                                                                          |
| 确定性问题分析 | task="statistics"<br>summary_mode="md5"          | 采集统计信息和tensor的CRC-32校验值，快速分析确定性问题                                                                                                                                                      |
| 轻量确定性差异定位 | task="statistics"<br>summary_mode="xor"          | 仅采集XOR校验值；如需加速采集，可安装编译时包含`--include-mod=xor_checksum`的工具包，安装方法请参见[安装指南-源码安装](../../install_guide/msprobe_install_guide.md#23-源码安装)中的“xor_checksum加速算子”相关指导，优先使用C++加速算子执行，可带来数倍性能提升 |
| NaN/Inf检测 | task="nan_check"      | 通过寄存器状态检测API运行中的NaN/Inf                                                                                                                                                                |

### 使用示例

#### dump采集基础示例

dump采集基础示例如下，示例代码中的高亮显示为需要在用户脚本中添加的代码。

<details>
<summary><b>单击展开示例</b></summary>

```diff
+# 导入工具数据采集接口，尽量在迭代训练文件导包后的位置执行seed_all和实例化PrecisionDebugger
+from msprobe.pytorch import seed_all, PrecisionDebugger
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

- 本工具提供固定的 API 支持列表，若需要删除或增加 dump 的 API，可以在 [support_wrap_ops.yaml](../../../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml) 文件内手动修改，如下示例：

  ```yaml
  functional:  # functional为算子类别，找到对应的类别，在该类别下按照下列格式删除或添加API
    - conv1d
    - conv2d
    - conv3d
  ```

  删除API的场景：部分模型代码逻辑会存在API原生类型校验，工具执行dump操作时，对模型的API封装可能与模型的原生API类型不一致，此时可能引发校验失败，详见《FAQ》中“[异常情况](../../support/faq.md#异常情况)”的第10条。

- 如果需要采集自定义API的精度数据，需要先向工具注册该API，注册方式请参见[批量注册用户自定义的API](#批量注册用户自定义的api)或[register_custom_api](#register_custom_api)。

- 对于本工具支持 dump 的 API（参见 [support_wrap_ops.yaml](../../../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml) 文件），支持对 API 输入或输出配置忽略规则进行过滤，具体操作请参见[API输入输出过滤](#api输入输出过滤)。

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
+from msprobe.pytorch import PrecisionDebugger, module_dump, module_dump_end

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
+from msprobe.pytorch import PrecisionDebugger

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

+from msprobe.pytorch import PrecisionDebugger

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