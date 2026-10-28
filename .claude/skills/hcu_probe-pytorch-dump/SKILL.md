---
name: hcu_probe-pytorch-dump
description: 将 HCU Probe / hcuprobe 的 PyTorch 精度数据采集能力集成到用户的训练或推理代码中。根据代码结构和采集需求生成或放置 config.json，并通过 PrecisionDebugger、module_dump 和 module_dump_end 等接口完成代码插桩。适用于用户要求在 PyTorch 脚本中 dump 精度数据、采集训练过程中的基础数据或函数模块级数据、集成精度调试接口等场景。
---

> ⚠️ **复验状态(2026-10)**:本 skill 未纳入本轮(2026-10 Megatron / verl)实跑验证。已在新版本复验的是 `hcu_probe-verl-fsdp-v08-integration`、`hcu_probe-verl-megatron-v08-integration`、`hcu_probe-megatron-integration`,见 `docs_hcu/zh/quick_start/`。

# PyTorch Precision Dump Integration

## 1. 任务目标

将 hcuProbe  的 PyTorch 精度数据采集能力集成到用户现有的训练或推理工程中，尽可能减少对原有业务逻辑的侵入，并确保插桩位置、配置路径和数据采集范围符合用户需求。

执行过程中应遵循以下原则：

- **最小化代码修改**：仅修改完成数据采集所必需的代码，尽量保持原有模型结构、训练逻辑、优化器调用和数据处理流程不变。
- **先确认需求，再修改代码**：缺少必要信息时，优先向用户收集信息，不得自行假设采集模式或工程入口。
- **遵循实际工程结构**：根据训练入口、模型定义位置和模块调用关系判断插桩方式。
- **参考已有实现**：优先使用项目中已有的配置文件、示例代码和 API 用法，避免臆造接口或参数。
- **明确验证边界**：区分代码静态检查、运行验证和实际数据产物检查，不得将未执行的验证描述为已通过。

## 2. 工作流程

必须按以下顺序执行各阶段。对于尚未获得的信息，可以暂停当前阶段并向用户询问；不得在必要信息缺失时直接生成最终插桩方案。

### 阶段 0：收集必要信息

开始修改前，确认以下信息：

1. **训练或推理工程目录**：用户代码所在的实际目录。
2. **主入口代码**：训练或推理入口文件的路径及相关代码。若代码较长，可先查看入口文件，再根据调用关系读取相关文件。
3. **采集类型**：
   - 基础数据采集：按配置采集运行过程中的精度数据。
   - 函数模块数据采集：通过模块级接口控制特定模块的数据采集范围。
4. **验证需求**：询问用户是否需要实际运行验证；若需要，确认可用的运行环境和启动方式。

如果上述信息不完整，应将缺失项合并为一次简洁的询问，避免反复提问。

请在每次用户调用之前都询问清楚

### 阶段 1：分析工程结构

读取入口代码及必要的关联文件，确定以下信息：

- 训练或推理的实际入口文件。
- 模型定义文件及模块调用关系。
- `PrecisionDebugger` 的实例化位置。
- 模型前向、反向传播和参数更新等关键执行位置。
- 是否涉及多文件导入、多个模型、多个训练阶段或其他可能影响采集范围的结构。
- 当前工程是否已经存在精度采集代码或配置文件。

根据实际代码结构，将工程归类为以下两种主要情况：

- **单文件工程**：入口代码和需要插桩的模型逻辑位于同一文件。
- **跨文件工程**：入口文件与模型定义、目标模块或相关执行逻辑分布在不同文件中。

如果工程结构不符合上述简单分类，应先梳理实际调用关系，再确定插桩方案，不得强行套用示例。

### 阶段 2：准备 config.json

根据用户选择的采集类型，查找项目提供的对应配置模板。

默认配置模板位置为：

`config/config.json`

执行要求：

1. 先检查模板是否存在，并读取其实际内容。
2. 根据用户工程和采集需求，确定配置文件应放置的位置。
3. 优先将配置文件放置在用户指定的工程目录中；如果该目录不适合存放配置文件，应明确说明并选择合理位置。
4. 如果目标位置已经存在 `config.json`，先读取并检查，避免直接覆盖用户已有配置。
5. 尽量复用项目模板，仅修改确实需要调整的字段。
6. 记录配置文件的实际路径，并确保后续插桩代码中的 `config_path` 与之对应。

注意：

- 不得凭空编造配置字段、枚举值或默认参数。
- 不得假定配置文件一定相对于当前 Python 文件解析；应结合工具的实际实现确认路径解析规则。
- 如果模板不存在或配置格式不明确，应先查找项目文档和已有示例；仍无法确认时，向用户说明问题，不得自行猜测。

### 阶段 3：生成插桩方案并修改代码

根据采集类型和工程结构选择对应方案。

#### 3.1 基础数据采集：单文件工程

参考以下基本模式，将接口放置在合适的位置：

```python
from hcuprobe.pytorch import seed_all, PrecisionDebugger

# 在需要的位置固定随机性
seed_all()

# 使用实际配置文件路径
debugger = PrecisionDebugger(config_path="./config.json")

# 模型、损失函数及优化器初始化
# ...

for data, label in data_loader:
    debugger.start(model)

    output = model(data)
    # ...
    loss.backward()
    optimizer.step()

    debugger.stop()
    debugger.step()
```

插桩要求：

- 在模型执行前后合理设置采集边界。
- 根据实际训练逻辑确认前向、反向和参数更新是否处于预期采集范围。
- 不得机械地将示例代码原样复制到所有训练工程。
- `seed_all()` 可能影响随机数状态，应根据项目要求确认是否需要使用，以及其适当的调用位置。
- 通过工具实现和实际示例确认 `start()`、`stop()`、`step()` 的具体语义后，再确定最终插桩位置。

#### 3.2 函数模块数据采集：单文件工程

通过 `PrecisionDebugger` 和模块级接口，对指定模块进行数据采集。

参考模式：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from hcuprobe.pytorch import (
    PrecisionDebugger,
    module_dump,
    module_dump_end,
)

debugger = PrecisionDebugger(config_path="./config.json")


class ModuleOP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(8, 4)
        self.linear_2 = nn.Linear(4, 2)

    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = self.linear_2(x1)
        return F.relu(x2)


if __name__ == "__main__":
    module = ModuleOP()

    debugger.start()

    x = torch.randn(10, 8)
    # 普通采集范围内的操作

    module_dump(module, "MyModuleOP")
    out = module(x)
    module_dump_end()

    loss = out.sum()
    loss.backward()

    debugger.stop()
    debugger.step()
```

插桩要求：

- `module_dump()` 的目标必须是实际存在且符合工具要求的模块。
- 模块标识应与用户的采集目标一致，避免随意使用示例名称。
- 确认模块级采集与普通采集的范围关系，避免将用户期望采集的操作排除在采集范围之外。
- 如果需要采集反向传播数据，应根据工具支持情况确认模块级采集的覆盖范围。
- 对于多个目标模块，应分别分析采集边界及接口的配对关系。

#### 3.3 基础数据采集：跨文件工程

对于入口文件与模型定义分离的工程，应分别分析各文件的职责和导入顺序。

例如：

- `train.py`：工程入口，负责初始化采集器和启动训练。
- `module.py`：定义模型及相关计算逻辑。
- `config.json`：保存采集配置。

入口文件示意：

```python
import torch
from module import ModuleOP
from hcuprobe.pytorch import PrecisionDebugger

debugger = PrecisionDebugger(config_path="./config.json")

if __name__ == "__main__":
    module = ModuleOP()

    x = torch.randn(10, 8)
    out = module(x)
    loss = out.sum()
    loss.backward()
```

模型定义文件示意：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from hcuprobe.pytorch import PrecisionDebugger


class ModuleOP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(8, 4)
        self.linear_2 = nn.Linear(4, 2)

    def forward(self, x):
        PrecisionDebugger.start()
        x1 = self.linear_1(x)
        PrecisionDebugger.stop()

        x2 = self.linear_2(x1)
        return F.relu(x2)
```

注意：

- 上述代码仅用于说明跨文件插桩的基本结构，不代表所有版本的 API 都支持相同的调用方式。
- 修改前必须核对当前项目中 `PrecisionDebugger.start()` 和 `PrecisionDebugger.stop()` 的真实定义、调用要求及其与实例方法的关系。
- 必须确认实例化、导入和接口调用顺序符合工具要求，尤其是需要在模型导入或 API 执行之前完成封装的场景。
- 如果实际目标是采集特定模块的数据，应优先依据真实需求确定插桩位置，而不是将所有操作都放进 `forward()`。
- 修改跨文件代码时，应尽量避免重复实例化、循环导入和不必要的全局状态。

#### 3.4 其他工程结构

对于推理脚本、分布式训练、多进程启动、多个模型或动态模型调用等场景：

1. 先梳理实际执行路径。
2. 查阅工具中与该场景对应的接口说明或已有测试。
3. 明确采集器的初始化位置、生命周期和进程行为。
4. 在不确定接口行为时先向用户说明限制，不得直接套用单文件或跨文件示例。

### 阶段 4：验证代码及采集产物

插桩完成后，先向用户确认是否需要验证。

如果用户选择不验证，应明确说明目前仅完成代码修改，尚未确认运行行为和采集结果。

如果用户选择验证，按以下顺序执行。

#### 4.1 静态检查

- 检查 Python 语法。
- 检查新增导入是否有效。
- 检查 `config_path` 是否指向实际文件。
- 检查 `start()`、`stop()`、`step()` 及模块级接口的调用位置。
- 检查是否存在重复初始化、遗漏结束接口或明显的控制流问题。
- 使用 `git diff` 确认修改仅涉及预期文件。

#### 4.2 运行验证

在用户授权且环境可用的前提下：

1. 使用用户提供的启动命令或项目文档中的正确命令。
2. 优先运行最小化测试，避免不必要的完整训练。
3. 检查配置解析、接口调用和模型执行是否正常。
4. 若运行需要 GPU、HCU 或其他特定设备，应确认对应环境可用。
5. 如果运行失败，区分环境问题、配置问题、接口问题和插桩问题，并根据错误信息进行必要修复。

不得在未经用户同意的情况下启动高成本、长时间或可能影响共享资源的训练任务。

#### 4.3 检查采集产物

如果工具成功运行，进一步检查：

- 是否生成预期的采集输出目录或文件。
- 采集结果是否与用户指定的采集类型相符。
- 采集范围是否覆盖目标操作或模块。
- 是否存在空结果、缺失数据或异常终止。

具体产物路径和文件名应以实际配置、工具实现及运行日志为准，不得预设不存在的产物结构。

#### 4.4 向用户反馈结果

完成后简要说明：

- 修改了哪些文件。
- 新增了哪些采集接口。
- 配置文件的实际路径。
- 采用的采集类型及采集范围。
- 执行了哪些验证。
- 验证是否通过；如未验证，应明确说明。
- 如有未解决的问题，列出具体错误和后续建议。

## 3. 通用约束

- 优先检查项目内的真实实现、官方文档和已有测试，再决定 API 用法。
- 不得擅自更改模型计算逻辑、张量形状、数据类型、损失函数或优化器行为。
- 不得覆盖用户已有配置或无关文件。
- 不得将示例中的相对路径、模块名称和模型结构直接视为用户工程的真实值。
- 对不确定的接口行为、版本差异和运行限制，应先查证或明确说明。
- 所有修改均应便于审查，并能够通过差异对比定位新增内容。