---
name: pytorch-integration
description: 把 hcuprobe 精度数据 dump 集成进 PyTorch 训练/推理脚本。当用户需要采集算子级（L1）、模块级（L0）精度数据、编写 config.json、按 step 定点 dump、或排查"结果目录空/字段缺失"等问题时使用。keywords: hcuprobe, PrecisionDebugger, dump, 精度数据采集, config.json, L0, L1, module_dump, seed_all
---

# PyTorch 精度数据采集集成

本 skill 说明如何用 `hcuprobe` 在 PyTorch 脚本中接入精度数据采集，覆盖 `scripts/` 下四个可直接运行的 demo。

## 核心模型

hcuprobe 的采集流程只有三个动作，所有 demo 都是这三步的不同变体：

1. **构造** `PrecisionDebugger(config_path=...)` —— 进程内单例，重复构造返回同一实例。
2. **圈定范围** —— `debugger.start()` / `debugger.stop()` 之间的 forward/backward 才会被采集。
3. **推进 step** —— `debugger.step()` 结束当前 step，把缓冲的数据落盘到 `dump_path/stepN/`。

`start()` 和 `stop()` 是成对的书挡。漏掉 `stop()` 数据不落盘；漏掉 `step()` 则多个 step 的数据会挤在同一个目录里。

## 最小可运行模板

`scripts/quick_start.py` 是最小完整例子：单次 forward + backward + 一次 step。

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from hcuprobe.pytorch import PrecisionDebugger, seed_all

class ModuleOP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=8, out_features=4)

    def forward(self, x):
        x1 = self.linear(x)
        return F.relu(x1)

if __name__ == "__main__":
    seed_all()                                  # 固定随机种子，保证可复现
    os.chdir(Path(__file__).resolve().parent)   # 让相对 dump_path 落在脚本目录
    debugger = PrecisionDebugger(config_path=str(Path(__file__).resolve().parent / "config.json"))
    model = ModuleOP()
    debugger.start(model=model)                 # 圈定采集范围
    x = torch.randn(10, 8)
    out = model(x)
    loss = out.sum()
    loss.backward()
    debugger.stop()                             # 停止采集
    debugger.step()                             # 落盘当前 step
```

**必须 `os.chdir` 到脚本目录**。`config.json` 里的 `"dump_path": "results"` 是相对路径，不切目录时结果会散落在启动时的工作目录下。四个 demo 都做了这一步。

## 四种集成模式

### 模式 1：单次前反向（`quick_start.py`）

用于快速验证工具是否接通。一次 forward/backward 配一次 step，产出 `results/step0/proc<PID>/`。

### 模式 2：训练循环（`basic_dump.py`）

把书挡放进 dataloader 循环内部，每个 batch 一次 `start/stop/step`：

```python
model.train()
for step, (x, y) in enumerate(dataloader):
    x = x.to(device).contiguous(memory_format=torch.channels_last)
    y = y.to(device).contiguous(memory_format=torch.channels_last)
    debugger.start(model=model)
    out = model(x)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    debugger.stop()
    debugger.step()
```

`step()` 的调用次数决定产出多少个 `stepN` 目录——它自增，不需要手工传序号。demo 里 batch_size=4、数据量 12，循环 3 圈，于是 `results/` 下正好是 `step0`、`step1`、`step2`。

书挡的范围覆盖 forward、loss 计算、`backward` 和 `optimizer.step()`，也就是优化器更新产生的算子也会被采集。想让这部分不进 dump，把 `debugger.stop()` 提到 `optimizer.step()` 之前。

### 模式 3：模块级采集（`module_dump.py`）

不需要 `config.json` 的 `level` 配成 L0，用 `module_dump()` 直接圈定某个子模块，采集它的输入输出：

```python
from hcuprobe.pytorch import PrecisionDebugger, module_dump, module_dump_end

debugger = PrecisionDebugger(config_path=...)
debugger.start()                      # 这里不需要传 model
model = ModuleOP()
x = torch.randn(10, 8)
module_dump(model, "MyModuleOP")      # 第一个参数必须是 nn.Module，第二个必须是 str
out = model(x)
module_dump_end()                     # 与 module_dump 成对
loss = out.sum()
loss.backward()
debugger.stop()
debugger.step()
```

这里采集到的数据 key 是模块级命名，与模式 1/2 的算子级命名不同：

| 模式 | dump.json 中的 key 形式 |
| --- | --- |
| 模式 1 / 2（L1 算子级） | `Functional.linear.0.forward`、`Functional.relu.0.forward` |
| 模式 3（模块级） | `Module.MyModuleOP.ModuleOP.forward.0` |

`module_dump` 内部通过 `register_module_hook(..., recursive=False)` 只挂传进去的那一个模块，因此 `"MyModuleOP"` 这个字符串会成为 key 里的名字段——它不必等于类名。

两个易错点，对应源码里的显式校验：

- `module_dump` 的第一个参数不是 `torch.nn.Module` 会抛 `MsprobeException(INVALID_PARAM_ERROR)`；第二个参数不是 `str` 同样抛错。
- 必须先 `PrecisionDebugger(...)` 再 `module_dump()`，否则 `_get_debugger_instance()` 抛 `INTERFACE_USAGE_ERROR: PrecisionDebugger must be instantiated before using module_dump interfaces`。

### 模式 4：在模型内部埋点（`train.py` + `model.py`）

没有全局的书挡循环，而是把 `start/stop` 写进 `forward` 里，只采集指定的一段：

```python
# model.py
class ModuleOP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=8, out_features=4)
        self.linear_2 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        PrecisionDebugger.start()
        x1 = self.linear_1(x)
        PrecisionDebugger.stop()
        x2 = self.linear_2(x)
        r1 = F.relu(x2)
        return r1
```

```python
# train.py
debugger = PrecisionDebugger(config_path=...)
model = ModuleOP()          # 只 import，不调 start()
x = torch.randn(10, 8)
out = model(x)
loss = out.sum()
loss.backward()
```

这一段跑出来的结果印证了埋点范围生效：`linear_2` 和 `relu` 都不在书挡内，所以 `results/step0/proc195179/dump.json` 里只有 `Functional.linear.0.forward` 和 `Functional.linear.0.backward` 两个 key，没有 `linear_2`、没有 `relu`。

这里是 `PrecisionDebugger.start()` 的类方法形态（内部走单例），与模式 1-3 的实例调用等价。

## config.json

demo 用的是最简配置，够跑通 L1 统计量采集：

```json
{
    "task": "statistics",
    "dump_path": "results",
    "rank": [],
    "step": [],
    "level": "L1",
    "async_dump": false,
    "extra_info": true,

    "statistics": {
        "scope": [],
        "list": [],
        "tensor_list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

常用字段：

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `task` | `statistics` / `tensor` / `acc_check` / `nan_check` / `structure` | `statistics` 只采统计量；`tensor` 额外复刻真实数据，体积大 |
| `dump_path` | str | 必选。相对路径的行为取决于 cwd，见上文 `os.chdir` |
| `level` | `L0` / `L1` / `L2` / `mix` / `debug` | `L1` 为默认，算子级；`L0` 模块级；`mix` = L0+L1 |
| `rank` | list | 单卡训练**必须为 `[]`**，不能指定 rank |
| `step` | list | 默认采所有 step。只想采第 2 步写 `[2]`，范围写 `"4-6"` |
| `async_dump` | bool | 训练结束后统一落盘，降同步开销；`task=tensor` 时必须配 `list` |
| `extra_info` | bool | 是否产出 `stack.json` / `construct.json`，默认 true |

`statistics` 段里的 `scope` 可进一步收窄（L1 填 API 名如 `Tensor.add.0.forward`，L0 填模块名），`data_mode` 控制采 input/output/backward，`summary_mode` 可选 `statistics` / `md5` / `xor`。

## 输出结构

```
results/
└── step0/
    └── proc193962/          # proc + PID，多进程/多卡时每个进程一个目录
        ├── dump.json        # 精度数据本体
        ├── stack.json       # 调用栈，key 为自增序号
        └── construct.json   # 模型结构
```

`dump.json` 顶层是 `task` / `level` / `framework` / `dump_data_dir` / `data`，`data` 下按算子名分组，每组含 `input_args`、`input_kwargs`、`output`（backward 侧为 `input` / `output`）。tensor 条目记录 `type`、`dtype`、`shape`、`Max`、`Min`、`Mean`、`Norm`、`requires_grad`；非 tensor 参数（如 `upsample_nearest` 的 `size`）记录为 `{"type": "int", "value": 64}`。

`stack.json` 把每个算子映射到一条从框架内部到用户脚本的调用栈，排查"这个算子是谁调起来的"时看它。

`construct.json` 在本 demo 中为 `{}`——这几个脚本都是直接在 Python 里搭的网络，没有走图编译，所以没有结构信息可采。吃 `{}` 是正常的，不代表采集失败。

## 排错

**结果目录为空**：先确认 `os.chdir` 是否生效、`dump_path` 落在哪。再确认 `stop()` 和 `step()` 都调了——只调 `start()` 不会产出任何文件。

**dump.json 里有节点但参数不全**：`input_kwargs` 只记录非 tensor 的关键字参数；`risk_level`（默认 `FOCUS`）会过滤掉 reshape/transpose/permute/to/view 这类形状变换 API，需要全量时配 `"risk_level": "ALL"`。

**`stepN` 目录数量不对**：`debugger.step()` 的次数就是步数。中途想跳步用配置里的 `step` 字段，不要在代码里调两次 `step()`。

**多卡下只有一张卡有数据**：`rank` 没配时默认采所有卡；配了但值超过实际运行的 rank 范围，那张卡就静默为空——注意单卡训练必须写 `[]`。

**`module_dump` 报 `INTERFACE_USAGE_ERROR`**：`PrecisionDebugger` 实例还没构造。先构造再调。
