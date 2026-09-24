# 配置与产物参考

## config.json 参数说明

hcu_probe 的数据采集通过 `config.json` 配置文件控制，在 `PrecisionDebugger(config_path="./config.json")` 初始化时传入。以下仅列出 PyTorch 场景适用的参数。

### 通用参数

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|---|---|---|---|---|
| `task` | str | 否 | `"statistics"` | dump 任务类型：`"statistics"`（仅统计信息）、`"tensor"`（统计信息 + 真实数据）、`"structure"`（仅模型结构） |
| `dump_path` | str | **是** | — | dump 数据输出目录的**绝对路径** |
| `rank` | list[int\|str] | 否 | `[]`（全部卡） | 指定采集某张卡的数据，如 `[0, 1, "4-6"]`。单卡训练时必须为 `[]` |
| `step` | list[int\|str] | 否 | `[]`（全部 step） | 指定采集某些 step，如 `[0, "4-6"]` |
| `level` | str | 否 | `"L1"` | dump 级别：`"L0"`（模块级）、`"L1"`（API 级，默认）、`"mix"`（L0 + L1） |
| `async_dump` | bool | 否 | `false` | 异步 dump 开关。开启后数据在当前 step 结束后统一落盘，训练过程中不触发同步。tensor 模式开启异步时必须配置 `tensor.list` 参数 |
| `dump_enable` | bool | 否 | — | 动态启停 dump 开关。配置后在运行中修改 json 可动态开关采集 |
| `extra_info` | bool | 否 | `true` | 是否输出 `stack.json` 和 `construct.json`。设为 `false` 可加快采集速度 |
| `precision` | str | 否 | `"low"` | 统计值计算精度：`"high"` 用 float32（更精确但可能显存溢出）、`"low"` 用原始类型 |
| `risk_level` | str | 否 | `"FOCUS"` | API 风险级别过滤，仅 L1 / mix 生效：`"ALL"`（全部）、`"CORE"`（核心高风险的融合计算/通信/精密计算 API）、`"FOCUS"`（核心 + 关注，排除 reshape/transpose/permute/to/view 等形状变换类 API） |

---

### task=statistics 专属参数

在 config.json 中增加 `"statistics": { ... }` 字段：

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|---|---|---|---|---|
| `scope` | list[str] | 否 | `[]` | dump 范围，配置两个模块名或 API 名锁定区间，要求列表长度为 2 |
| `list` | list[str] | 否 | `[]` | 自定义采集的算子列表。可配置 API 全称（如 `Tensor.permute.1.forward`）、关键字（如 `relu`，匹配所有含 relu 的 API）、模块名（mix 级别展开该模块） |
| `tensor_list` | list[str] | 否 | `[]` | 指定额外采集完整 tensor 真实数据的算子列表（关键字匹配） |
| `data_mode` | list[str] | 否 | `["all"]` | 数据过滤：`"all"`、`"forward"`、`"backward"`、`"input"`、`"output"`；除 `"all"` 外可自由组合，如 `["forward", "input"]` |
| `summary_mode` | str | 否 | `"statistics"` | 输出模式：`"statistics"`（仅统计信息）、`"md5"`（含 CRC-32 校验值）、`"xor"`（仅 XOR 校验值，不输出统计信息） |

配置示例：

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
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

---

### task=tensor 专属参数

在 config.json 中增加 `"tensor": { ... }` 字段：

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|---|---|---|---|---|
| `scope` | list[str] | 否 | `[]` | 同 statistics |
| `list` | list[str] | 否 | `[]` | 同 statistics |
| `data_mode` | list[str] | 否 | `["all"]` | 同 statistics |
| `summary_mode` | str | 否 | `"statistics"` | `"statistics"` 或 `"md5"` |
| `bench_path` | str | 否 | — | 预置的 md5 基线数据路径。配置后 dump 会实时对比 md5 差异，仅 dump 差异节点的真实数据 |
| `diff_nums` | int | 否 | `1` | 最大差异次数。第 N 次差异出现后停止差异分析。设为 `-1` 表示持续检测直到训练结束 |

配置示例：

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "async_dump": false,

    "tensor": {
        "scope": [],
        "list": [],
        "data_mode": ["all"],
        "bench_path": "/home/bench_data_dump",
        "summary_mode": "md5",
        "diff_nums": 5
    }
}
```

---

### task=structure 专属参数

structure 模式仅采集模型结构，用于网络结构的一致性比对。无额外特殊配置：

```json
{
    "task": "structure",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "mix"
}
```

> **使用场景**：模型迁移过程中确保迁移前后模型结构一致，或排查精度差异前先确认是否由结构差异引起。

---

### 推荐配置组合

| 调试需求 | task | level | summary_mode | 特点 |
|---|---|---|---|---|
| 初步精度分析 | `"statistics"` | `"mix"` | `"statistics"` | 资源占用低，快速获取统计信息 |
| 深度精度分析 | `"tensor"` | `"mix"` | `"statistics"` | 采集完整真实数据，支持详细逐元素对比 |
| 确定性问题分析 | `"statistics"` | `"L1"` | `"md5"` | 采集统计信息 + CRC-32 校验，快速定位确定性问题 |
| 仅结构比对 | `"structure"` | `"mix"` | — | 只采集结构，不采集训练数据，速度最快 |

### list 参数配置说明

`list` 支持三种配置方式：

1. **API 全称** — dump 该具体 API 的数据：
   ```json
   "list": ["Tensor.permute.1.forward", "Torch.relu.3.backward"]
   ```

2. **关键字** — dump 所有名称中包含该关键字的 API：
   ```json
   "list": ["relu"]
   ```
   level=mix 时还会展开包含该关键字的模块。

3. **模块名**（仅 mix 级别）— 展开该模块，dump 其所有内部 API：
   ```json
   "list": ["Module.model.language_model.encoder.layers.0.mlp.ParallelMlp.forward.0"]
   ```

### scope 参数配置说明

`scope` 须配置两个完整的模块名或 API 名，锁定区间范围；长度为 2：

```json
"scope": ["Module.conv1.Conv2d.forward.0", "Module.fc2.Linear.forward.0"]
```

level 为 L0 时配置模块名；L1 时配置 API 名；mix 时可配置模块名或 API 名。

---

## dump 产物目录结构

### 目录树

```
{dump_path}/
└── step0/
    ├── rank0/
    │   ├── dump.json             # 各层/API 的输入输出张量统计信息
    │   ├── stack.json            # 各层/API 的 Python 调用栈（extra_info=true 时生成）
    │   ├── construct.json        # 模型层级结构（level=L0 或 mix 时非空，L1 时为空文件）
    │   └── dump_tensor_data/     # 真实 tensor 数据文件（task=tensor 时生成）
    │       ├── Module.conv1.Conv2d.forward.0.input.0.pt
    │       ├── Module.conv1.Conv2d.forward.0.output.0.pt
    │       ├── Module.conv1.Conv2d.backward.0.output.0.pt
    │       └── ...
    ├── rank1/
    │   ├── dump.json
    │   ├── stack.json
    │   ├── construct.json
    │   └── dump_tensor_data/
    │       └── ...
    └── ...
```

### 文件说明

| 文件 | 内容 | 何时生成 |
|---|---|---|
| `dump.json` | 每层 / API 的输入输出张量统计信息（max / min / mean / norm / shape / dtype） | 始终生成 |
| `stack.json` | 每层 / API 被调用时的 Python 完整调用栈信息 | `extra_info=true`（默认）时生成 |
| `construct.json` | 模型父子节点关系，用于重构完整的模型层级结构 | `level=L0` 或 `level=mix` 时生成；`level=L1` 时为空文件 |
| `dump_tensor_data/` | 真实 tensor 数据文件（.pt 或 .npy 格式） | `task=tensor` 时生成 |

### construct.json 格式

`construct.json` 是一个 JSON 对象，key 为节点标识符，value 为其父节点标识符（根节点为 `null`）：

```json
{
  "Module.conv1.Conv2d.forward.0": null,
  "Module.bn1.BatchNorm2d.forward.0": null,
  "Module.layer1.Sequential.forward.0": null,
  "Module.layer1.0.Bottleneck.forward.0": "Module.layer1.Sequential.forward.0",
  "Module.layer1.0.conv1.Conv2d.forward.0": "Module.layer1.0.Bottleneck.forward.0",
  "Module.layer1.0.bn1.BatchNorm2d.forward.0": "Module.layer1.0.Bottleneck.forward.0",
  "Module.layer1.0.relu.ReLU.forward.0": "Module.layer1.0.Bottleneck.forward.0",
  ...
}
```

通过父节点关系，可以还原出完整的树状模型结构。叶子节点是实际的计算算子（Conv2d、Linear、ReLU 等），非叶子节点是容器模块（Sequential、Bottleneck 等）。

---

## L0 / L1 命名规则

### L0 命名：Module 级别

格式：

```
Module.{module_name}.{class_name}.{forward/backward}.{调用次数}
```

| 组成部分 | 含义 | 示例 |
|---|---|---|
| `Module` | 固定前缀 | `Module` |
| `module_name` | 通过 `named_modules()` 获取的模块名，用 `.` 分隔层级 | `layer1.0.conv1` |
| `class_name` | 模块的类名 | `Conv2d`, `Linear`, `Bottleneck` |
| `forward` / `backward` / `parameters_grad` | 前向 / 反向 / 参数梯度 | `forward`, `backward` |
| 调用次数 | 该模块第几次被调用（从 0 开始） | `0`, `1`, `2` |

示例：

- `Module.conv1.Conv2d.forward.0` — 名为 `conv1` 的 Conv2d 模块第 1 次前向计算
- `Module.layer1.0.relu.ReLU.forward.1` — layer1 下第 0 个 Bottleneck 中 `relu` 的 ReLU 模块第 2 次前向
- `Module.fc.Linear.parameters_grad.0` — fc 中 Linear 模块的参数梯度

**多模型场景**（如 `list[nn.Module]`），模块名前加索引：`Module.0.conv1.Conv2d.forward.0`

### L0 前向/反向的构造关系

前向传播过程中，模型按深度优先顺序遍历并计算每一个叶子节点。对应到 `construct.json` 中：

- 前向节点以 `forward.{次数}` 结尾
- 反向节点以 `backward.{次数}` 结尾
- 反向顺序与前向相反（从最后一个前向节点开始回溯）

### L1 命名：API 级别

按 API 所在模块分类，格式：

```
{类别}.{api_name}.{调用次数}.{forward/backward}.{输入/输出}.{索引}
```

| 类别 | 对应的 Python 模块 |
|---|---|
| `Functional` | `torch.nn.functional` 中的函数 |
| `Torch` | `torch` 模块中的函数 |
| `Tensor` | `torch.Tensor` 类的方法 |
| `Distributed` | 分布式通信相关 API |

示例：

- `Functional.conv2d.1.forward.input.0` — `functional.conv2d` 第 2 次调用的前向第 0 个输入
- `Tensor.permute.3.forward.output.0` — `Tensor.permute` 第 4 次调用的前向输出
- `Torch.matmul.0.backward.input.1` — `torch.matmul` 第 1 次调用的反向第 1 个输入
- `Distributed.all_gather.2.forward.input.0` — 分布式通信 `all_gather` 第 3 次调用

### L0 vs L1 选择建议

| level | 采集粒度 | 适用场景 | 数据量 |
|---|---|---|---|
| `L0` | 模块级（nn.Module 子类） | 大模型场景，模块结构可能在框架迁移中发生变化 | 小 |
| `L1` | API 级（每个 tensor 算子） | 算子级精度排查，精确定位到具体 API | 大 |
| `mix` | L0 + L1 | **推荐**，同时采集模块和 API 两个粒度的数据 | 较大 |

> L0 模式下，即使两个模型的 API 调用链因框架适配而不同（算子融合 / 切分策略不同），只要模块边界一致，仍可按模块粒度进行精度比对。
