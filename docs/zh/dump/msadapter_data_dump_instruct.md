# MSAdapter场景精度数据采集

MSAdapter 是一款 MindSpore 生态适配工具，可以将 PyTorch 训练脚本高效迁移至 MindSpore 框架执行，以实现在不改变原有 PyTorch 用户开发习惯的情况下，使得 PyTorch 代码能在昇腾上获得高效性能。

msprobe 工具主要通过在训练脚本内添加 dump 接口、启动训练的方式采集精度数据。

**注意**：

- 为了正确识别 MSAdapter 场景，在导入 msprobe 工具前，需完成 torch 模块的导入。

- 因 MindSpore 框架自动微分机制的限制，dump 数据中可能会缺少原地操作模块/API 及其上一个模块/API 的反向数据。

本工具提供固定的 API 支持列表，若需要删除或增加 dump 的 API，可以在 msprobe/pytorch/hook_module/support_wrap_ops.yaml 文件内手动修改，如下示例：

```yaml
functional:  # functional为算子类别，找到对应的类别，在该类别下按照下列格式删除或添加API
  - conv1d
  - conv2d
  - conv3d
```

删除 API 的场景：部分模型代码逻辑会存在 API 原生类型校验，工具执行dump操作时，对封装后的模型 API 可能与模型的原生 API 类型不一致，此时可能引发校验失败，详见《[FAQ](../faq.md#异常情况)》“异常情况”的第10和11条。

## 工具安装

请参见《[msprobe 工具安装指南](../msprobe_install_guide.md)》。

## 接口介绍

### msprobe.mindspore.PrecisionDebugger

**功能说明**：通过加载 dump 配置文件的方式来确定 dump 操作的详细配置。

**原型**：

```Python
PrecisionDebugger(config_path=None, task=None, dump_path=None, level=None, step=None)
```

**参数说明**:

1. config_path：指定 dump 配置文件路径，string 类型。参数示例："./config.json"。未配置该路径时，默认使用 [config.json](../../../python/msprobe/config.json) 文件的默认配置，配置选项含义可见 [config.json 介绍](./config_json_introduct.md)。

2. 其他参数与 [config.json](../../../python/msprobe/config.json) 文件中的同名配置字段含义相同，具体可见 [config.json 介绍](./config_json_introduct.md)。当参数值非None时，优先级高于 [config.json](../../../python/msprobe/config.json) 文件中的同名配置。

#### start

**功能说明**：启动精度数据采集。需要与 [**stop**](#stop) 接口一起添加在训练迭代的 for 循环内。

**原型**：

```Python
start(model=None)
```

**参数说明**:

1. model：指定需要采集 Module 级数据的模型，支持传入 torch.nn.Module、list[torch.nn.Module]或Tuple[torch.nn.Module] 类型，默认未配置。level 配置为 "L0" 或 "mix" 时，必须在该接口中配置该参数。API级别（"L1" level）dump 时，传入 model 可以采集 model 内包含 primitive op 对象在内的所有 API 数据，若不传入 model 参数，则只采集非 primitive op 的 API 数据。

#### stop

**功能说明**：停止精度数据采集。在 **start** 接口调用之后的任意位置添加。若 **stop** 接口添加在反向计算代码之后，则会采集 **start** 和该接口之间的前反向数据。
若 **stop** 接口添加在反向计算代码之前，则需要将 [**step**](#step) 接口添加到反向计算代码之后，才能采集 **start** 和该接口之间的前反向数据。

**注意**：**stop** 接口必须调用，否则可能导致精度数据落盘不全。

**原型**：

```Python
stop()
```

#### step

**功能说明**：进行训练 step 数的自增，完成当前 step 所有数据的落盘并更新 dump 参数。在一个 step 训练结束的位置添加，且必须在 **stop** 接口之后的位置调用。该接口需要配合 **start** 和 **stop** 函数使用，尽量添加在反向计算代码之后，否则可能会导致反向数据丢失。

**原型**：

```Python
step()
```

#### forward_backward_dump_end

**功能说明**：停止精度数据采集。与 **stop** 接口功能相同，该函数在将来会被移除，建议使用 **stop** 接口。

**原型**：

```Python
forward_backward_dump_end()
```

#### save

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

### msprobe.mindspore.seed_all

**功能说明**：用于固定网络中的随机性和开启确定性计算。

**原型**：
```python
seed_all(seed=1234, mode=False, rm_dropout=True)
```

**参数说明**:

1. seed: 随机性种子，默认值：1234，非必选。参数示例: seed=1000。该参数用于 random、numpy.random, mindspore.common.Initializer、mindspore.nn.probability.distribution的随机数生成以及 Python 中 str、bytes、datetime 对象的 hash 算法。

2. mode：确定性计算使能，可配置 True 或 False，默认值：False，非必选。参数示例：mode=True。该参数设置为 True 后，将会开启算子确定性运行模式与归约类通信算子（AllReduce、ReduceScatter、Reduce）的确定性计算。注意：确定性计算会导致 API 执行性能降低，建议在发现模型多次执行结果不同的情况下开启。

3. rm_dropout：控制 dropout 失效的开关。可配置 True 或 False，默认值：True，非必选。参数示例：rm_dropout=True。该参数设置为 True 后，将会使 mindspore.ops.Dropout，mindspore.ops.Dropout2D，mindspore.ops.Dropout3D，mindspore.mint.nn.Dropout和mindspore.mint.nn.functional.dropout 失效，以避免因随机 dropout 造成的网络随机性。建议在采集数据前调用。

**注意**：通过 rm_dropout 控制 dropout 失效或生效需要在初始化 Dropout 实例前调用才能生效。

## 示例代码

以下为添加了 msprobe 工具 dump 接口的示例训练脚本。

```python
import mindspore as ms
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入工具的数据采集接口
from msprobe.mindspore import PrecisionDebugger

# 在模型训练开始前实例化PrecisionDebugger
debugger = PrecisionDebugger(config_path='./config.json')


# 定义网络
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=8, out_features=4)
        self.linear2 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        logits = F.relu(x2)
        return logits


net = Net()


def train_step(inputs):
    return net(inputs)


if __name__ == "__main__":
    data = (torch.randn(10, 8), torch.randn(10, 8), torch.randn(10, 8))
    grad_fn = ms.value_and_grad(train_step, grad_position=0)

    for inputs in data:
        # 开启数据 dump
        debugger.start(model=net)

        out, grad = grad_fn(inputs)

        # 停止数据 dump
        debugger.stop()
        # 更新 step 信息
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
|   |   |   |    ├── Tensor.permute.1.forward.npy
|   |   |   |    ├── Functional.linear.5.backward.output.npy    # 命名格式为{api_type}.{api_name}.{API调用次数}.{forward/backward}.{input/output}.{参数序号}, 其中，“参数序号”表示该API的第n个输入或输出，例如1，则为第一个参数，若该参数为list格式，则根据list继续排序，例如1.1，表示该API的第1个参数的第1个元素。
|   |   |   |    ...
|   |   |   |    ├── Module.conv1.Conv2d.forward.0.input.0.npy          # 命名格式为{Module}.{module_name}.{class_name}.{forward/backward}.{调用次数}.{input/output}.{参数序号}, 其中，“参数序号”表示该Module的第n个参数，例如1，则为第一个参数，若该参数为list格式，则根据list继续排序，例如1.1，表示该Module的第1个参数的第1个元素。
|   |   |   |    ├── Module.conv1.Conv2D.forward.0.parameters.bias.npy  # 模块参数数据：命名格式为{Module}.{module_name}.{class_name}.forward.{调用次数}.parameters.{parameter_name}。
|   |   |   |    └── Module.conv1.Conv2D.parameters_grad.weight.npy     # 模块参数梯度数据：命名格式为{Module}.{module_name}.{class_name}.parameters_grad.{parameter_name}。因为同一模块的参数使用同一梯度进行更新，所以参数梯度文件名不包含调用次数。
|   |   |   |                                                          # 当dump时传入的model参数为List[torch.nn.Module]或Tuple[torch.nn.Module]时，模块级数据的命名中包含该模块在列表中的索引index，命名格式为{Module}.{index}.*，*表示以上三种模块级数据的命名格式，例如：Module.0.conv1.Conv2d.forward.0.input.0.npy。
│   |   |   ├── dump.json
│   |   |   ├── stack.json
│   |   |   └── construct.json
│   |   ├── rank1
|   |   |   ├── dump_tensor_data
|   |   |   |   └── ...
│   |   |   ├── dump.json
│   |   |   ├── stack.json
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
* `dump.json`： 保存 API 或 Module 前反向数据的统计量信息。包含 dump 数据的 API 名称或 Module 名称，各数据的 dtype、 shape、max、min、mean、L2norm（L2范数，平方根）统计信息以及当配置 summary_mode="md5" 时的 CRC-32 数据。具体介绍可参考[dump.json文件说明](#dumpjson文件说明)。
* `stack.json`：API/Module 的调用栈信息。
* `construct.json`：分层分级结构，level 为 L1 时，construct.json 内容为空。


当 task 为 tensor 时，dump 过程中，npy 文件在对应算子或者模块被执行后就会落盘，而 json 文件则需要在正常执行 PrecisionDebugger.stop() 后才会写入完整数据。因此如果程序异常终止，终止前被执行算子的相关 npy 文件得以保存，但 json 文件中的数据可能丢失。

其中 rank 为设备上各卡的 ID，每张卡上 dump 的数据会生成对应 dump 目录。非分布式场景下没有 rank ID，目录名称为 rank。

npy 文件名的前缀含义如下：

| 前缀        | 含义                          |
| ----------- | ---------------------------- |
| Tensor      | torch.Tensor API数据          |
| Torch       | torch API数据                 |
| Functional  | torch.nn.functional API数据   |
| NPU         | NPU 亲和API数据               |
| Distributed | torch.distributed API数据     |
| Jit         | 被 "jit" 装饰的模块或函数数据   |
| Module      | torch.nn.Module 类（模块）数据 |

### dump.json文件说明

#### L0级别

L0 级别的 dump.json 文件包括模块的前反向的输入输出，以及模块的参数和参数梯度。以 Conv2d 模块为例，网络中模块调用代码为:  
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
 "framework": "mindtorch",
 "dump_data_dir": "/dump/path",
 "data": {
  "Module.conv2.Conv2d.forward.0": {
   "input_args": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
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
     "data_name": "Module.conv2.Conv2d.forward.0.input.0.npy"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
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
     "data_name": "Module.conv2.Conv2d.forward.0.output.0.npy"
    }
   ],
   "parameters": {
    "weight": {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
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
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.weight.npy"
    },
    "bias": {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      32
     ],
     "Max": 0.05744686722755432,
     "Min": -0.04894155263900757,
     "Mean": 0.006410328671336174,
     "Norm": 0.17263513803482056,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.bias.npy"
    }
   }
  },
  "Module.conv2.Conv2d.parameters_grad": {
   "weight": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
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
     "data_name": "Module.conv2.Conv2d.parameters_grad.weight.npy"
    }
   ],
   "bias": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      32
     ],
     "Max": 0.014914230443537235,
     "Min": -0.006656786892563105,
     "Mean": 0.002657240955159068,
     "Norm": 0.029451673850417137,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.parameters_grad.bias.npy"
    }
   ]
  },
  "Module.conv2.Conv2d.backward.0": {
   "input": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
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
     "data_name": "Module.conv2.Conv2d.backward.0.input.0.npy"
    }
   ],
   "output": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
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
     "data_name": "Module.conv2.Conv2d.backward.0.output.0.npy"
    }
   ]
  }
 }
}
```

#### L1级别

L1级别的dump.json文件包括API的前反向的输入输出。以 relu API 为例，网络中 API 调用代码为:  
`output = torch.nn.functional.relu(input)`  

dump.json文件中包含以下数据名称：  

- `Functional.relu.0.forward`：API的前向数据，其中input_args为API的输入数据（位置参数），input_kwargs为API的输入数据（关键字参数），output为API的输出数据。  
- `Functional.relu.0.backward`：API的反向数据，其中input为API的反向输入梯度（对应前向输出的梯度），output为API的反向输出梯度（对应前向输入的梯度）。

```json
{
 "task": "tensor",
 "level": "L1",
 "framework": "mindtorch",
 "dump_data_dir":"/dump/path",
 "data": {
  "Functional.relu.0.forward": {
   "input_args": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
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
     "data_name": "Functional.relu.0.forward.input.0.npy"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
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
     "data_name": "Functional.relu.0.forward.output.0.npy"
    }
   ]
  },
  "Functional.relu.0.backward": {
   "input": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
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
     "data_name": "Functional.relu.0.backward.input.0.npy"
    }
   ],
   "output": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
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
     "data_name": "Functional.relu.0.backward.output.0.npy"
    }
   ]
  }
 }
}  
```

#### mix级别

mix级别的dump.json文件同时包括L0和L1级别的dump数据，文件格式与上述示例相同。  