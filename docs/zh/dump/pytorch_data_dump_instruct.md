# PyTorch场景精度数据采集

## 简介

msProbe工具主要通过在训练脚本内添加dump接口、启动训练的方式采集精度数据。

dump "statistics"模式的性能膨胀大小"与"tensor"模式采集的数据量大小，可以参考[dump基线](../baseline/pytorch_data_dump_perf_baseline.md)。

本工具提供固定的 API 支持列表，若需要删除或增加 dump 的 API，可以在[support_wrap_ops.yaml](../../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml)文件内手动修改，如下示例：

```yaml
functional:  # functional为算子类别，找到对应的类别，在该类别下按照下列格式删除或添加API
  - conv1d
  - conv2d
  - conv3d
```

删除API的场景：部分模型代码逻辑会存在API原生类型校验，工具执行dump操作时，对模型的API封装可能与模型的原生API类型不一致，此时可能引发校验失败，详见《FAQ》中“[异常情况](../faq.md#异常情况)”的第10和11条。

加工具后loss/gnorm发生变化：可能是工具中的item操作引入同步，pt/ms框架的hook机制等原因导致的，详见《FAQ》中“[工具导致计算结果变化](../faq.md#模型计算结果改变原因分析)”。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。

**约束**

仅支持PyTorch框架。

## 快速入门

这个示例定义了一个 nn.Module 类型的简单网络，使用原型函数 PrecisionDebugger 进行数据采集。

```python
# 根据需要import包
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入工具的数据采集接口
from msprobe.pytorch import PrecisionDebugger, seed_all

# 在模型训练开始前固定随机性
seed_all()

# 在模型训练开始前实例化PrecisionDebugger
debugger = PrecisionDebugger()

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
    
    # 开启数据 dump
    debugger.start(model=module)
    x = torch.randn(10, 8)
    out = module(x)
    loss = out.sum()
    loss.backward()

    # 关闭数据 dump
    debugger.stop()
```

## 接口介绍

### PrecisionDebugger

**功能说明**：通过加载 dump 配置文件的方式来确定 dump 操作的详细配置。

**原型**：

```Python
PrecisionDebugger(config_path=None, task=None, dump_path=None, level=None, step=None)
```

1. config_path：指定 dump 配置文件路径；
2. 其他参数均在 config.json 文件中可配，详细配置可见 [config.json 介绍](./config_json_introduct.md)。

此接口的参数均不是必要（均不配置的情况下默认采集所有 rank 和 step 的 L1 级别的统计数据），且优先级高于 config.json 文件中的配置，但可配置的参数相比 config.json 较少。

注：此接口的初始化需与采集目标在同一个进程中，否则将无法采集目标数据。

### start

**功能说明**：启动精度数据采集。在模型初始化之后的位置添加。需要与 stop 函数一起添加在 for 循环内。

**原型**：

```Python
debugger.start(model=None, token_range=None)
```

1. model：指定需要采集 Module 级数据的模型，支持传入 torch.nn.Module、list[torch.nn.Module]或Tuple[torch.nn.Module] 类型，默认未配置。
level 配置为"L0"|"mix"或token_range不为None时，必须在该接口或 **PrecisionDebugger** 接口中配置该参数。
本接口中的 model 比 PrecisionDebugger 中 model 参数优先级更高，会覆盖 PrecisionDebugger 中的 model 参数。
<br>对于复杂模型，如果仅需要监控一部分(如model.A，model.A extends torch.nn.Module)，传入需要监控的部分(如model.A)即可。  
注意：传入的当前层不会被dump，工具只会dump传入层的子层级。如传入了model.A，A本身不会被dump，而是会dump A.x, A.x.xx等。
2. token_range：指定推理模型采集时的token循环始末范围，支持传入[int, int]类型，代表[start, end]，范围包含边界，默认未配置。

### stop

**功能说明**：停止精度数据采集。在 **start** 函数之后的任意位置添加。
若 **stop** 函数添加在反向计算代码（如loss.backward）之后，则会采集 **start** 和该函数之间的前反向数据。
若 **stop** 函数添加在反向计算代码之前，则需要将 [**step**](#step) 函数添加到反向计算代码之后，才能采集 **start** 和该函数之间的前反向数据。
使用示例可参见 [快速入门](#快速入门) 和 [采集完整的前反向数据](#采集完整的前反向数据)。

**注意**：**stop** 函数必须调用，否则可能导致精度数据落盘不全。

**原型**：

```Python
debugger.stop()
```

### step

**功能说明**：结束一个 step 的数据采集，完成所有数据落盘并更新 dump 参数。在一个 step 结束的位置添加，且必须在 **stop** 函数之后的位置调用。
该函数需要配合 **start** 和 **stop** 函数使用，尽量添加在反向计算代码（如loss.backward）之后，否则可能会导致反向数据丢失。使用示例可参见[采集完整的前反向数据](#采集完整的前反向数据)。

**原型**：

```Python
debugger.step()
```

### module_dump

**功能说明**：开启模块级精度数据dump。该接口为函数模块化接口，即只会dump输入的模块数据，不会dump子模块和模块内API的数据。
需要配合start、stop和step等接口使用。使用示例可参考[采集函数模块化数据](#采集函数模块化数据)

**原型**：

```Python
module_dump(module, module_name)
```
**参数说明**:

1. module: 网络中实例化好的nn.Module类对象。数据类型：torch.nn.Module。必选。
2. module_name：用户自定义该module名称，主要用于dump数据的命名。数据类型：str。必选。

### module_dump_end

**功能说明**：结束模块级精度数据dump。在module_dump和module_dump_end之间的API或者Module，除了传入的module，其他数据均不会被dump。
在执行完module_dump_end后数据恢复正常的dump模式。

**原型**：

```Python
module_dump_end()
```

### seed_all

**功能说明**：对于网络中随机性的固定和确定性计算，msProbe工具提供了seed_all接口用于固定网络中的随机性和开启确定性计算。

**原型**：
```python
seed_all(seed=1234, mode=False, rm_dropout=False)
```

**参数说明**:

1. seed: 随机性种子。参数示例: seed=1000。默认值：1234。非必选
2. mode：确定性计算模式。可配置True或False。参数示例：mode=True。默认为False。非必选（注意：确定性计算会导致API执行性能降低，建议在发现模型多次执行结果不同的情况下开启）
3. rm_dropout：控制dropout失效的开关。可配置 True 或 False，默认值：False，非必选。参数示例：rm_dropout=True。
该参数设置为 True 后， 工具会自动将 `torch.nn.functional.dropout`、`torch.nn.functional.dropout2d`、`torch.nn.functional.dropout3d`、`torch.nn.Dropout`、`torch.nn.Dropout2d`、`torch.nn.Dropout3d`
的接口参数 p 置为0，以避免因随机dropout造成的网络随机性。 注意：通过rm_dropout控制dropout失效或生效需要在初始化dropout实例前调用才能生效。

当前工具 dump 默认不会固定随机性和使 dropout 失效，若希望每次采集的数据保持一致，建议在 dump 数据前调用 seed_all 接口。

seed_all 函数可固定随机数的范围如下表。

| API                                      | 固定随机数                  |
| ---------------------------------------- | --------------------------- |
| os.environ['PYTHONHASHSEED'] = str(seed) | 禁止 Python 中的 hash 随机化 |
| os.environ['HCCL_DETERMINISTIC'] = True  | 固定通信算子计算的确定性       |
| random.seed(seed)                        | 设置 random 随机生成器的种子  |
| np.random.seed(seed)                     | 设置 numpy 中随机生成器的种子 |
| torch.manual_seed(seed)                  | 设置当前 CPU 的随机种子       |
| torch.cuda.manual_seed(seed)             | 设置当前 GPU 的随机种子       |
| torch.cuda.manual_seed_all(seed)         | 设置所有 GPU 的随机种子       |
| torch_npu.npu.manual_seed(seed)          | 设置当前 NPU 的随机种子       |
| torch_npu.npu.manual_seed_all(seed)      | 设置所有 NPU 的随机种子       |
| torch.use_deterministic_algorithms(True) | CUDA/CANN 使能确定性计算（注意mode为True时才会调用该方法开启确定性     |
| torch.backends.cudnn.enable=False        | 关闭 cuDNN                   |
| torch.backends.cudnn.benchmark=False     | cuDNN 确定性地选择算法       |
| torch.backends.cudnn.deterministic=True  | cuDNN 仅使用确定性的卷积算法 |
| torch.nn.functional.dropout              | 将 dropout 的接口参数 p 置为0 |
| torch.nn.functional.dropout2d            | 将 dropout2d 的接口参数 p 置为0 |
| torch.nn.functional.dropout3d            | 将 dropout3d 的接口参数 p 置为0 |
| torch.nn.Dropout                         | 将 Dropout 的接口参数 p 置为0 |
| torch.nn.Dropout2d                       | 将 Dropout2d 的接口参数 p 置为0 |
| torch.nn.Dropout3d                       | 将 Dropout3d 的接口参数 p 置为0 |

需要保证 CPU 或 GPU 以及 NPU 的模型输入完全一致，dump 数据的比对才有意义，seed_all 并不能保证模型输入完全一致，如下表所示场景需要保证输入的一致性。

| 场景            | 固定方法      |
| --------------- | ------------- |
| 数据集的 shuffle | 关闭 shuffle。 |

关闭 shuffle 示例：

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)
```

### save

**功能说明**：单点保存网络执行过程中正反向数值，并以统计值/张量文件落盘。

**原型**：
```python
save(variable, name, save_backward=True)
```

**参数说明**:
| 参数名称        | 参数含义          |        支持数据类型    |   是否必选|
| ----------     | ------------------| ------------------- | ------------------- |
| variable       | 需要保存的变量     |dict, list, tuple, torch.tensor, int, float, str |  是  |
| name           | 指定的名称         | str                 | 是  |
| save_backward  | 是否保存反向数据   | boolean             | 否 |

具体使用样例可参考：[单点保存工具使用介绍](./debugger_save_instruct.md)。

### set_init_step

**功能说明**：设置起始step数，step数默认从0开始计数，使用该接口后step从指定值开始计数。该函数需要写在训练迭代的循环开始前，不能写在循环内。

**原型**：

```Python
debugger.set_init_step(step)
```

**参数说明**:

1.step: 指定的起始step数。

### register_custom_api

**功能说明**：注册用户自定义的api到工具用于 L1 dump 。

**原型**：

```Python
debugger.register_custom_api(module, api_name, api_prefix)
```
**参数说明**:

以 torch.matmul api 为例

1.module: api 所属的包，即传入 torch。

2.api_name: api 名，string类型，即传入 "matmul"。

3.api_prefix:  [dump.json](#dumpjson文件说明)中 api 名的前缀，可选，默认为包名的字符串格式, 即 "torch"。

### restore_custom_api

**功能说明**：恢复用户原有的自定义的api，取消 dump 。

**原型**：

```Python
debugger.restore_custom_api(module, api_name)
```
**参数说明**:

以 torch.matmul api 为例

1.module: api 所属的包，即传入 torch。

2.api_name: api 名，string类型，即传入 "matmul"。


## 示例代码


### 采集完整的前反向数据

```Python
from msprobe.pytorch import PrecisionDebugger, seed_all
# 在模型训练开始前固定随机性
seed_all()
# 请勿将PrecisionDebugger的初始化流程插入到循环代码中
debugger = PrecisionDebugger(config_path="./config.json", dump_path="./dump_path")
# 模型、损失函数的定义及初始化等操作
# ...
# 数据集迭代的位置一般为模型训练开始的位置
for data, label in data_loader:
    debugger.start()  # 开启数据dump
    # 如下是模型每个step执行的逻辑
    output = model(data)
    #...
    loss.backward()
    debugger.stop()  # 关闭数据dump
    debugger.step()  # 结束一个step的dump
```

### 采集函数模块化数据

```Python
# 根据需要import包
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入工具的数据采集接口
from msprobe.pytorch import PrecisionDebugger, module_dump, module_dump_end

# 在模型训练开始前实例化PrecisionDebugger
debugger = PrecisionDebugger(config_path='./config.json')

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
    # 开启数据dump
    debugger.start()
    x = torch.randn(10, 8)
    # ...                              # start和module_dump接口之间的数据正常dump
    module_dump(module, "MyModuleOP")  # 开启模块级精度数据dump
    out = module(x)                    # module内部的child modules或API将不会被dump
    module_dump_end()                  # 关闭模块级精度数据dump
    loss = out.sum()                   # module_dump_end和stop接口之间的数据正常dump
    loss.backward()
    # 关闭数据dump
    debugger.stop()
```

### 跨文件采集数据
为了确保所有API都被工具封装，PrecisionDebugger的实例化通常放在训练工程的入口位置，但有的时候，模型定义会在另一个文件中。 假设有两个文件，train.py（为训练工程入口）module.py（为模型定义文件），为了采集module.py中定义的ModuleOP模块中某些子模块或API的前反向数据，需要在train.py和module.py文件中分别导入PrecisionDebugger并进行如下配置。

train.py文件：

```Python
# 根据需要import包
import torch
from module import ModuleOP

# 导入工具的数据采集接口
from msprobe.pytorch import PrecisionDebugger

# 将PrecisionDebugger的实例化放在文件的开始位置，即导包后的位置，确保所有API都被封装
debugger = PrecisionDebugger(config_path='./config.json')

if __name__ == "__main__":
    module = ModuleOP()

    x = torch.randn(10, 8)
    out = module(x)
    loss = out.sum()
    loss.backward()
```

module.py文件：

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

from msprobe.pytorch import PrecisionDebugger

# 定义网络
class ModuleOP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=8, out_features=4)
        self.linear_2 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        PrecisionDebugger.start()
        x1 = self.linear_1(x)
        PrecisionDebugger.stop()
        x2 = self.linear_2(x1)
        r1 = F.relu(x2)
        return r1

```

### 推理模型采集指定token_range

```Python
from vllm import LLM, SamplingParams
from msprobe.pytorch import PrecisionDebugger, seed_all
# 在模型训练开始前固定随机性
seed_all()
# 请勿将PrecisionDebugger的初始化流程插入到循环代码中
debugger = PrecisionDebugger(config_path="./config.json", dump_path="./dump_path")
# 模型定义及初始化等操作
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temprature=0.8, top_p=0.95)
llm = LLM(model='...')
model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
# 开启数据dump, 指定采集推理模型逐字符循环推理中的第1~3次
debugger.start(model=model, token_range=[1,3])  
# 推理模型生成的逻辑
output = llm.generate(prompts, sampling_params=sampling_params)
# 关闭数据dump并落盘
debugger.stop()
debugger.step()
```

## dump结果文件介绍

### dump目录说明

训练结束后，工具将 dump 的数据保存在 dump_path 参数指定的目录下。目录结构示例如下：

```lua
├── dump_path
│   ├── step0
│   |   ├── rank0
│   |   │   ├── dump_tensor_data
|   |   |   |    ├── Tensor.permute.1.forward.pt
|   |   |   |    ├── Functional.linear.5.backward.output.pt    # 命名格式为{api_type}.{api_name}.{API调用次数}.{forward/backward}.{input/output}.{参数序号}, 其中，“参数序号”表示该API的第n个输入或输出，例如1，则为第一个参数，若该参数为list格式，则根据list继续排序，例如1.1，表示该API的第1个参数的第1个元素。
|   |   |   |    ...
|   |   |   |    ├── Module.conv1.Conv2d.forward.0.input.0.pt          # 命名格式为{Module}.{module_name}.{class_name}.{forward/backward}.{调用次数}.{input/output}.{参数序号}, 其中，“参数序号”表示该Module的第n个参数，例如1，则为第一个参数，若该参数为list格式，则根据list继续排序，例如1.1，表示该Module的第1个参数的第1个元素。
|   |   |   |    ├── Module.conv1.Conv2d.forward.0.parameters.bias.pt  # 模块参数数据：命名格式为{Module}.{module_name}.{class_name}.forward.{调用次数}.parameters.{parameter_name}。
|   |   |   |    └── Module.conv1.Conv2d.parameters_grad.weight.pt     # 模块参数梯度数据：命名格式为{Module}.{module_name}.{class_name}.parameters_grad.{parameter_name}。因为同一模块的参数使用同一梯度进行更新，所以参数梯度文件名不包含调用次数。
|   |   |   |                                                          # 当dump时传入的model参数为List[torch.nn.Module]或Tuple[torch.nn.Module]时，模块级数据的命名中包含该模块在列表中的索引index，命名格式为{Module}.{index}.*，*表示以上三种模块级数据的命名格式，例如：Module.0.conv1.Conv2d.forward.0.input.0.pt。
│   |   |   ├── dump.json
│   |   |   ├── stack.json
│   |   |   ├── dump_error_info.log
│   |   |   └── construct.json
│   |   ├── rank1
|   |   |   ├── dump_tensor_data
|   |   |   |   └── ...
│   |   |   ├── dump.json
│   |   |   ├── stack.json
│   |   |   ├── dump_error_info.log
|   |   |   └── construct.json
│   |   ├── ...
│   |   |
|   |   └── rank7
│   ├── step1
│   |   ├── ...
│   ├── step2
```
* `rank`：设备 ID，每张卡的数据保存在对应的 `rank{ID}` 目录下。非分布式场景下没有 rank ID，目录名称为 rank。
* `dump_tensor_data`：保存采集到的张量数据。
* `dump.json`： 保存API或Module前反向数据的统计量信息。包含dump数据的API名称或Module名称，各数据的dtype、 shape、max、min、mean、L2norm（L2范数，平方根）统计信息以及当配置summary_mode="md5"时的CRC-32数据。具体介绍可参考[dump.json文件说明](#dumpjson文件说明)。
* `dump_error_info.log`: 仅在dump工具报错时拥有此记录日志，用于记录dump错误日志。
* `stack.json`：API/Module的调用栈信息。
* `construct.json`：分层分级结构，level为L1时，construct.json内容为空。


dump 过程中，pt 文件在对应算子或者模块被执行后就会落盘，而 json 文件则需要在正常执行 PrecisionDebugger.stop() 后才会写入完整数据，异常的程序终止会保存终止前被执行算子的相关 npy 文件，可能会导致 json 文件中数据丢失。

其中 rank 为设备上各卡的 ID，每张卡上 dump 的数据会生成对应 dump 目录。非分布式场景下没有 rank ID，目录名称为 rank。

pt 文件保存的前缀和 PyTorch 对应关系如下：

| 前缀          | Torch模块             |
|-------------|---------------------|
| Tensor      | torch.Tensor        |
| Torch       | torch               |
| Functional  | torch.nn.functional |
| NPU         | NPU 亲和算子            |
| VF          | torch._VF           |
| Aten        | torch.ops.aten      |
| Distributed | torch.distributed   |
| MindSpeed   | mindspeed.ops       |

### dump.json文件说明

#### L0级别

L0级别的dump.json文件包括模块的前反向的输入输出，以及模块的参数和参数梯度。以PyTorch的Conv2d模块为例，网络中模块调用代码为:  
`output = self.conv2(input) # self.conv2 = torch.nn.Conv2d(64, 128, 5, padding=2, bias=True)`  

dump.json文件中包含以下数据名称：  

- `Module.conv2.Conv2d.forward.0`：模块的前向数据，其中input_args为模块的输入数据（位置参数），input_kwargs为模块的输入数据（关键字参数），output为模块的输出数据，parameters为模块的参数数据，包括权重（weight）和偏置（bias）。  
- `Module.conv2.Conv2d.parameters_grad`：模块的参数梯度数据，包括权重（weight）和偏置（bias）的梯度。  
- `Module.conv2.Conv2d.backward.0`：模块的反向数据，其中input为模块反向的输入梯度（对应前向输出的梯度），output为模块的反向输出梯度（对应前向输入的梯度）。

**说明**：当dump时传入的model参数为List[torch.nn.Module]或Tuple[torch.nn.Module]时，模块级数据的命名中包含该模块在列表中的索引index，命名格式为`{Module}.{index}.*`，*表示以上三种模块级数据的命名格式，例如：`Module.0.conv1.Conv2d.forward.0`。     

```json
{
 "task": "tensor",
 "level": "L0",
 "framework": "pytorch",
 "dump_data_dir": "/dump/path",
 "data": {
  "Module.conv2.Conv2d.forward.0": {
   "input_args": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      16,
      14,
      14
     ],
     "Max": 1.638758659362793,
     "Min": 0.0,
     "Mean": 0.2544615864753723,
     "Norm": 70.50277709960938,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.input.0.pt"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      32,
      10,
      10
     ],
     "Max": 1.6815717220306396,
     "Min": -1.5120246410369873,
     "Mean": -0.025344856083393097,
     "Norm": 149.65576171875,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.output.0.pt"
    }
   ],
   "parameters": {
    "weight": {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      5,
      5
     ],
     "Max": 0.05992485210299492,
     "Min": -0.05999220535159111,
     "Mean": -0.0006165213999338448,
     "Norm": 3.421217441558838,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.weight.pt"
    },
    "bias": {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32
     ],
     "Max": 0.05744686722755432,
     "Min": -0.04894155263900757,
     "Mean": 0.006410328671336174,
     "Norm": 0.17263513803482056,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.bias.pt"
    }
   }
  },
  "Module.conv2.Conv2d.parameters_grad": {
   "weight": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      5,
      5
     ],
     "Max": 0.018550323322415352,
     "Min": -0.008627401664853096,
     "Mean": 0.0006675920449197292,
     "Norm": 0.26084786653518677,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.parameters_grad.weight.pt"
    }
   ],
   "bias": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32
     ],
     "Max": 0.014914230443537235,
     "Min": -0.006656786892563105,
     "Mean": 0.002657240955159068,
     "Norm": 0.029451673850417137,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.parameters_grad.bias.pt"
    }
   ]
  },
  "Module.conv2.Conv2d.backward.0": {
   "input": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      32,
      10,
      10
     ],
     "Max": 0.0015069986693561077,
     "Min": -0.001139344065450132,
     "Mean": 3.3215508210560074e-06,
     "Norm": 0.020567523315548897,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.backward.0.input.0.pt"
    }
   ],
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      16,
      14,
      14
     ],
     "Max": 0.0007466732058674097,
     "Min": -0.00044813455315306783,
     "Mean": 6.814070275140693e-06,
     "Norm": 0.01474067009985447,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.backward.0.output.0.pt"
    }
   ]
  }
 }
}
```

#### L1级别

L1级别的dump.json文件包括API的前反向的输入输出。以PyTorch的relu函数为例，网络中API调用代码为:  
`output = torch.nn.functional.relu(input)`  

dump.json文件中包含以下数据名称：  

- `Functional.relu.0.forward`：API的前向数据，其中input_args为API的输入数据（位置参数），input_kwargs为API的输入数据（关键字参数），output为API的输出数据。  
- `Functional.relu.0.backward`：API的反向数据，其中input为API的反向输入梯度（对应前向输出的梯度），output为API的反向输出梯度（对应前向输入的梯度）。

```json
{
 "task": "tensor",
 "level": "L1",
 "framework": "pytorch",
 "dump_data_dir":"/dump/path",
 "data": {
  "Functional.relu.0.forward": {
   "input_args": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 1.3864083290100098,
     "Min": -1.3364859819412231,
     "Mean": 0.03711778670549393,
     "Norm": 236.20692443847656,
     "requires_grad": true,
     "data_name": "Functional.relu.0.forward.input.0.pt"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 1.3864083290100098,
     "Min": 0.0,
     "Mean": 0.16849493980407715,
     "Norm": 175.23345947265625,
     "requires_grad": true,
     "data_name": "Functional.relu.0.forward.output.0.pt"
    }
   ]
  },
  "Functional.relu.0.backward": {
   "input": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 0.0001815402356442064,
     "Min": -0.00013352684618439525,
     "Mean": 0.00011915402356442064,
     "Norm": 0.007598237134516239,
     "requires_grad": false,
     "data_name": "Functional.relu.0.backward.input.0.pt"
    }
   ],
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 0.0001815402356442064,
     "Min": -0.00012117840378778055,
     "Mean": 2.0098118724831693e-08,
     "Norm": 0.006532244384288788,
     "requires_grad": false,
     "data_name": "Functional.relu.0.backward.output.0.pt"
    }
   ]
  }
 }
}  
```

#### mix级别

mix级别的dump.json文件同时包括L0和L1级别的dump数据，文件格式与上述示例相同。  
