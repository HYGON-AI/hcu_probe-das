# data2db — dump / monitor 数据导入 SQLite + **趋势可视化入口**

把 dump 数据或 monitor v2 CSV 解析成 `.trend.db`(SQLite 数据库),供 tensorboard `TREND ANALYZER` 插件做趋势可视化。

> 趋势可视化没有独立的 `trend_visualize` 子命令,**入口就是 `data2db` 命令 + tensorboard 插件**的组合。

## 趋势可视化能做什么

从 3 个维度观测精度数据分布,定位长稳训练中的精度漂移:

| 维度 | 热力图视角 | 折线图视角 |
|---|---|---|
| **Step**(迭代步数) | 单个 step 下所有数据(X=Rank, Y=Module Name) | 单张量在 Step 维度的趋势 |
| **Rank**(节点) | 单个 rank 下所有数据(X=Step, Y=Module Name) | 单张量在 Rank 维度的趋势 |
| **Module Name**(张量目标) | 单张量在所有 step×rank 上的分布(X=Step, Y=Rank) | 单张量在 Module Name 维度的趋势 |

可选指标(dump 数据场景):
- `forward` / `backward` / `recompute` / `parameters_grad`

可选指标(monitor v2 数据场景,自动提取):
- `actv` / `actv_grad` / `exp_avg` / `exp_avg_sq` / `grad_unreduced` / `grad_reduced` / `param_origin` / `param_updated`

可选统计量:
- dump 数据:`norm` / `mix`(最大值) / `mean` / `min`
- monitor 数据:根据 CSV 列内容自动提取

## Python API

```python
from dcu_probe.tools import data2db

# 自动识别 dump 或 monitor 数据
data2db.run(
    db="./trend_db_out",
    data="./dcu_probe_runs/dcu_probe_dump",
    format="auto",       # 或 "dump" / "monitor" 显式指定
)
```

完整签名:

```python
def run(
    *,
    db: str,                       # 输出目录(SQLite 文件存盘)
    data: str,                     # 输入 dump 或 monitor 数据目录
    format: str = "auto",          # auto / dump / monitor
    mapping: str = None,           # 可选 mapping.json,做模型层名映射
    micro_step: bool = True,       # 启用微步计数(一个 step 拆多个微迭代)
    process_num: int = 1,          # 仅对 monitor 数据并行加速,dump 不支持多进程
) -> str
```

## CLI

```bash
# 1. dump 数据 → trend db
dcu-probe data2db --data ./dcu_probe_dump --db ./trend_out

# 2. monitor v2 数据 → trend db(支持多进程加速)
dcu-probe data2db --data ./monitor_v2_runs --db ./trend_out --format monitor --process_num 4

# 3. 浏览(dcu_probe whl 不内置浏览侧插件,见下文"浏览侧"段)
```

## 输出产物

| 场景 | 输出文件 |
|---|---|
| dump 数据 | `<db>/dump_data.trend.db` |
| monitor 数据 | `<db>/monitor_data.trend.db` |
| auto 模式 + 数据目录混合 | 同时生成两个 `.trend.db` |

## 输入要求

- **dump 数据**:`level = "L0"` 或 `"mix"`(L1 仅 API,无 module 信息,不够趋势可视化用)
- **monitor v2 数据**:输出格式必须是 `csv`(monitor v2 默认就是 csv,无需额外配置)
- 落盘数据范围 `[-1e9, 1e9]`,超出会被截断,`inf` → `1e9+1`,`-inf` → `-1e9-1`

## 浏览侧(dcu_probe 不内置 — 设计决策)

`data2db` **后端命令 dcu_probe whl 已支持**(生成 `.trend.db` SQLite),但**浏览侧的 `TREND ANALYZER` tensorboard 插件 dcu_probe whl 不内置**,这是有意识的架构决策。

### 上游两种 tensorboard 插件的发布形态对比

| 插件 | 发布形态 | 装它的代价 |
|---|---|---|
| `tb_graph_ascend`(graph_viz 浏览用) | **独立 PyPI 包**(`tb-graph-ascend 8.3.2`,1.4 MB 纯浏览端) | 不依赖 mindstudio-probe 主包,跟 dcu_probe 不冲突 ✅ |
| `trend_analyzer`(data2db 浏览用) | **`mindstudio-probe 26.0.0` 主包内嵌的顶层目录** | 必须装 2.1 MB 上游 msprobe 主包,内含 dump/compare/precheck/monitor 全套 — 跟 dcu_probe 重叠 90%+ ❌ |

dcu_probe 的定位是**独立 fork,替代 mindstudio-probe**;让 dcu_probe 用户为"看 trend"再装一遍上游主包,会变相把上游主包拉回依赖链,违背独立 fork 愿景。所以**当前 dcu_probe 不 ship 浏览侧 trend 插件**。

> ⚠️ **不要用**上游官方文档 link 的 `tb_trend_plugin-0.1.0-py3-none-any.whl`(放在 msprobe 仓库 `docs/zh/resources/` 下)。**经实测是残缺产物**:只 9.4 KB / 2 个孤儿 .py(`markdown_db.py` + `unable_mapping.py`,实际是开发者临时迁移脚本),entry point 指向 wheel 内根本不存在的 `tb_monitor_plugin.backend:MonVis`。装它 tensorboard 启动会 plugin 注册失败。

### 用户解锁趋势可视化的两条路

| 路径 | 命令 | 取舍 |
|---|---|---|
| **路径 1(简单,但拉回上游主包)** | 独立 venv:`pip install mindstudio-probe==26.0.0 tensorboard`;把 dcu_probe 生成的 `.trend.db` 拷到该 venv 的 logdir;`tensorboard --logdir <dir>`;浏览器 `#TrendVis` 入口 | 完整 UI(热力图 + 折线图)可用;代价:浏览机器装上游 msprobe 全套(2.1 MB) |
| **路径 2(脱困,纯 SQLite 读取)** | `sqlite3 dump_data.trend.db ".schema"` 看表结构;自定义 SQL 提取数据;matplotlib / seaborn 自渲染 | 适合 CI 脚本化分析、自定义指标;失去交互式 UI |

### 未来可能的演化

如果用户需求增长,可后续把 `trend_analyzer/` fork 进 dcu_probe whl(走 8 批 namespace 清理同样模式:保留上游字字一致代码 + 改顶层目录名 + 自己注册 tensorboard plugin entry point),或单独发个 `dcu-probe-tb-plugin` PyPI 包(对称 tb-graph-ascend 模式)。**当前不优先**。

## Megatron 模型并行可视化(上游能力)

Megatron 框架的张量并行 / 流水线并行 / 虚拟流水线并行 / 数据并行切分场景下,上游 mindstudio-probe 提供"模型并行可视化",识别每个 rank 上的模型层分布。该能力通过 `data2db` 输出的 `.trend.db` 加载到 tensorboard 后展示,前置条件:

- 仅支持 Megatron 框架
- dump 时需要在配置里写明 TP / PP / VPP / DP 切分参数(`construct.json` 内含)

## DCU 上的状态

| 维度 | 状态 |
|---|---|
| 后端 `data2db` 命令(生成 `.trend.db`) | 🟡 未在 DCU 独立实测,但代码层跨平台(纯 SQLite 操作,无 device 调用),DCU 上预期 OK |
| 浏览侧 trend 插件(`TREND ANALYZER` tensorboard tab) | ❌ **dcu_probe 不提供**(见上文"浏览侧"段) |
| dump 数据 → `data2db` 消费 | L0 dump 已实测(2026-05-12 scnet docker),数据格式跨平台,理论可消费 |
| monitor v2 数据 → `data2db` 消费 | monitor v2 训练侧已实测产 8 rank csv + loss_log,数据格式跨平台,理论可消费 |

后端实测(`dcu-probe data2db --data <dump> --db <out>` 产 `.trend.db` 文件)**待补做**。

## 与其他工具的关系

- `data2db` 把"原始 dump / monitor"压缩成"趋势 db",**没有精度判定 / diff 功能**,只做数据汇聚
- 想看具体算子的精度差异 → 用 `compare`(xlsx 表格)
- 想看模型结构 / 单 step 的层间精度 → 用 `graph_viz`(`.vis.db`,层级图)
- 想看跨 step / 跨 rank 的趋势 → 用 `data2db` + tensorboard `TREND ANALYZER`(热力图 + 折线图)

## 完整能力参考(上游 PyTorch 文档)

dcu_probe 的 data2db 实现完全沿用上游 mindstudio-probe v26.0.0-alpha.1 的代码。趋势可视化的完整 UI 操作细节本文档没展开,需要时查上游文档:

- TREND ANALYZER 浏览器界面三区(选择工具栏 / 热力图 / 折线图)操作详解
- 5 个标签类型(`default` / `layer` / `index` / `module` / `function`)的筛选规则
- 维度切换(Step / Rank / Module Name)对应的热力图 X/Y 轴含义
- 热力图模式(渐变色 vs 分段色)+ 热力条调整
- 折线图多点对比 + 图例切换 + 坐标轴缩放
- `mapping.json` 模型层名映射格式(精简名称 / step 间名称对齐)
- **Megatron 模型并行可视化**:TP / PP / VPP / DP 切分场景下,各 rank 上模型层分布的可视化(仅支持 Megatron)
- 微步计数 (`--micro_step`) 在梯度累积场景下的拆分逻辑
- TensorBoard 服务多种网络场景下的连接方式(直连 / vpn / vscode 端口转发 / 本地 PC 加载)

上游文档地址:

```
https://gitcode.com/Ascend/msprobe/blob/26.0.0-alpha.1/docs/zh/accuracy_compare/trend_visualization_instruct.md
```

> ⚠️ **不要参考同目录下 `mindspore_visualization_instruct.md`** — MindSpore 专属,数据维度的具体语义、tensorboard 插件名都不一样。
