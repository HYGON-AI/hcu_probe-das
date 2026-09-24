# overflow — 整网首个溢出节点分析

扫描 dump 数据,跨 rank 关联通信节点,定位训练中**最先出现 NaN/Inf 的算子**。输出 `anomaly_analyze_{ns}.json`。

## 分析能力

| 阶段 | 范围 |
|---|---|
| `_pre_analyze` | 第一个通信节点**之前**的异常算子 |
| `_analyze` | 跨通信节点的异常关联(整网级,通过 ProcessGroup 跨 rank 连接) |
| `_post_analyze` | 最后一个通信节点**之后**的异常算子 |

输出按 rank 分组,每个 rank 给出有限个异常节点(首个为主)。

## Python API

```python
from dcu_probe.tools import overflow

overflow.run(
    input_path="./dump/step_0",
    output_path="./overflow_out",
)
```

## CLI

```bash
dcu-probe overflow_check \
    -i ./dump/step_0 \
    -o ./overflow_out
```

## 输入要求

input_path 必须是**单个 step** 的目录,结构如下:

```
./dump/step_0/
├── rank_0/
│   ├── dump.json
│   ├── construct.json
│   └── stack.json
├── rank_1/
│   └── ...
└── ...
```

且 `dump.json` 内 `framework == "pytorch"`,否则 raise 不支持。

## 输出结构

```json
{
  "rank_0": [
    {
      "op_name": "Module.layer4.0.conv2.forward",
      "node_info": { ... 上下文 stack + 数据统计 ... }
    }
  ],
  "rank_1": [ ... ]
}
```

📖 **产物 4 字段逐项详解 + 实战阅读流程 + 按异常 module 的根因排查表见**:[`overflow_result_analysis.md`](overflow_result_analysis.md)

## DCU 上的状态

✅ **2026-05-12 scnet docker 端到端跑通**:
- 节点 f09r1n05 / 容器 `dcu_probe_lab` / 镜像 `triton-distributed:backup-20260324`
- 输入:llama2-7B L0 dump,8 卡 × step0(33MB dump.json + 5MB construct.json + 1.8MB stack.json per rank)
- 分析时长:18 秒(scan + 跨 rank 通信节点连接)
- 输出:`Cannot find any anomaly node, no need to generate analyze file.`(by design — 无 anomaly 时不创建产物)

DCU 上工作确认:
1. ✅ `framework == "pytorch"` 字段识别正常
2. ✅ RCCL 通信算子被正确识别(`_pre_analyze` / `_analyze` / `_post_analyze` 三阶段全过)
3. ✅ 多 rank 关联(`_connect_comm_nodes`)在 DCU 8 卡拓扑下工作

## 已知坑(DCU docker 环境)

1. **input_path 必须 writable**:上游 mindstudio-probe 的 `check_path_writability()` 对 input 路径也要求 writable。如果 dump 数据在 `:ro` mount 路径(像 `-v /public:/public:ro`),需要先 `cp -r` 到可写目录再调用。
2. **container 依赖**:overflow_check 自身只需 numpy/psutil,但同一 whl 内 import 链可能拉到 tensorboard / openpyxl 等。容器无外网时从主机 conda env `cp -r` 到 `/usr/local/lib/python3.10/dist-packages/`。

## 与其他工具的关系

- 配合 `dump` 用:训练时启用 L0 dump → 出现 NaN 后跑 `overflow` 定位首个出错节点
- 也可在 `graph_viz` 中用 `-oc / --overflow_check` 把溢出标注画到可视化图上

## 完整能力参考(上游 PyTorch 文档)

dcu_probe 的 overflow_check 实现完全沿用上游 mindstudio-probe v26.0.0-alpha.1 的代码。以下细节本文档没展开,需要时查上游文档:

- **异常计算节点判定规则**:当某计算节点 input 的 Max/Min **不含 inf/nan**,但 output **含 inf/nan** → 认为该节点是溢出起点
- **异常通信节点判定**(分两类):
  - **有向节点**(`send` / `recv` / `scatter` / `gather` / `broadcast` / `reduce`):src input 异常 → 起点在 src 之前的计算节点;src input 正常 output 异常 → 通信节点本身有问题
  - **无向节点**(`all_gather` / `all_reduce` / `reduce_scatter` / `all_to_all`):input 异常 → 起点在 src 之前;input 正常 output 异常 → 通信节点本身有问题
- **顺序判定**:有向算子 src 早于 dst;无向算子认为异常同时发生;计算节点按 dump 顺序排序
- `anomaly_analyze_{ts}.json` 完整字段(`op_name` / `data_info` / `construct_info` / `stack_info`)

上游文档地址:

```
https://gitcode.com/Ascend/msprobe/blob/26.0.0-alpha.1/docs/zh/overflow_check/overflow_check_instruct.md
```

> ⚠️ **不要参考同目录下 `mindspore_overflow_check_instruct.md`** — MindSpore 专属,通信算子分类、判定规则、产物格式都跟 PyTorch 路径不一样。
