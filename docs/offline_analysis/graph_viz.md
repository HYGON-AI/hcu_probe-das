# graph_viz — dump 图可视化(分级可视化)

把 dump 数据构建成层级图,加载到 tensorboard 浏览;或者两端 dump 比对成"差异图",定位首个精度发散点。

## 两种模式

| 模式 | 入参 | 输出 |
|---|---|---|
| **单源构图** | 只给 `target_path` | `build_{ts}.vis.db`,tensorboard 加载后能逐层下钻 |
| **双源 diff** | `target_path` + `golden_path` | `compare_{ts}.vis.db`,每个节点带 npu vs bench 的差异色彩标注 + 精度比对指标 |

## 完整能力清单(上游)

| 能力 | 参数 | 说明 |
|---|---|---|
| 重建模型层级结构 | 默认 | 输入 dump 须 `level=L0` 或 `mix`,`construct.json` 非空 |
| 多 rank 批量构图 | 自动识别 dump 目录格式 | 同时关联各卡通信节点,展示卡间数据传递 |
| 多 step 批量构图 | 自动识别 | step1 / step2 / ... 一起构 |
| 双源结构差异比对 | `-gp <golden>` | 含跨套件比对(DeepSpeed vs Megatron) |
| 双源精度数据比对 | `-gp <golden>` | 三种 dump 类型:`statistics` / `tensor` / `md5` |
| **跨套件比对** | `-lm` / `-lm mapping.yaml` | DCU↔NV / 不同框架 module 名映射 |
| **模糊匹配** | `-fm` | 节点名容差匹配,适合层级结构有微差的场景 |
| **溢出标注** | `-oc` | 每节点带 medium / high / critical 溢出等级标记 |
| 不同切分策略图合并 | `--rank_size --tp --pp --vpp --order` | 多机张量并行 / 流水线并行场景 |
| tensor 数据日志 | `-tensor_log` | 单模块/API 的 tensor 数据打印 |
| 进度日志 | `-progress_log` | 任务详细进度 |

## Python API

```python
from dcu_probe.tools import graph_viz

# 单源
graph_viz.run(target_path="./dcu_dump/step0", output_path="./vis_out")

# 双源 diff
graph_viz.run(
    target_path="./dcu_dump/step0",
    golden_path="./h20_dump/step0",
    output_path="./vis_diff_out",
    overflow_check=True,   # 把溢出节点标红
)
```

完整签名:

```python
def run(
    *,
    target_path: str,
    output_path: str,
    golden_path: str = None,           # 双源 diff
    layer_mapping: str = None,         # 手动点点匹配文件
    overflow_check: bool = False,      # 图上叠加溢出标注
    fuzzy_match: bool = False,
    rank_size: Sequence[int] = None,   # 并行配置(多机合并图)
    tp: Sequence[int] = None,
    pp: Sequence[int] = None,
    vpp: Sequence[int] = None,
    order: Sequence[str] = None,       # 并行维度顺序,默认 ['tp-cp-ep-dp-pp']
    ...
) -> str
```

## CLI

```bash
# 单源
dcu-probe graph_visualize -tp ./dump -o ./vis_out

# 双源
dcu-probe graph_visualize -tp ./dcu_dump -gp ./h20_dump -o ./vis_diff -oc

# tensorboard 浏览
tensorboard --logdir ./vis_out --bind_all
```

## DCU 上的关注点

### construct.json 模块名兼容

可视化依赖 `construct.json` 还原模型图。dcu_megatron 通过 patch 改了一些模块实现(reuse-fp32-param / EDGC / DualPipeV 调度模块等),可能与原生 Megatron 在 NV 上的 construct.json 模块名不完全一致。

**应对**:第一次跨平台比对时,在生成的图上观察自动匹配率。匹配率低时用 `layer_mapping` 写一份 DCU↔NV 模块名映射 YAML,常见映射可沉淀为模板复用。

### 通信算子可视化

`reduce_scatter_to_sequence_parallel_region` / `reduce_from_tensor_model_parallel_region` 等通信算子在 dcu_megatron 里被 `torch._dynamo.disable` 包装。验证点:这些算子是否能被 dump hook 拦截到并出现在可视化图上。

### DCU 上的状态(后端构图 + 浏览侧 双实证)

✅ **2026-05-12 后端构图 — scnet docker 跑通**:
- 节点 f09r1n05 / 容器 `dcu_probe_lab`
- 输入:llama2-7B L0 dump 单源 8 卡 × step0
- 处理时长:~20 秒(rank 0-7 并行 export)
- 产物:`build_20260512164944.vis.db` SQLite 文件 **608 MB**

✅ **2026-05-13 浏览侧 — Mac 本地 tb_graph_ascend 跑通**(详见下文"浏览侧"段):
- scp scnet 新跑 `build_20260513114445.vis.db`(llama2-7B L0 dump,580 MB)到 Mac
- pip install `tb-graph-ascend 8.3.2` + `tensorboard 2.20.0`
- 浏览器 `http://localhost:6006/#graph_ascend` 完整渲染:Module.module.Float16Module.forward.X 节点 + dtype/shape/Max/Min/Mean/Norm/requires_grad 全字段

DCU 上 construct.json 模块名格式与上游 mindstudio-probe 兼容,无需 layer_mapping 即可自动 export。跨 NV 比对仍待补做(需 H20 标杆 dump)。

## 输出产物

| 模式 | 文件名格式 |
|---|---|
| 单源构图 | `build_{timestamp}.vis.db` |
| 双源 diff | `compare_{timestamp}.vis.db` |

控制台日志:匹配成功/失败的节点数统计 + 各 rank export 进度。

## 浏览侧:tb_graph_ascend tensorboard 插件

`graph_visualize` 命令**dcu_probe whl 已支持**(生成 `.vis.db`),浏览器看节点搜索 / 精度筛选 / 层级下钻 / 跳转展开等交互能力**需要单独装 `tb_graph_ascend` 插件**(独立 PyPI 包,不在 dcu_probe whl 里)。

### 安装(任意干净 Python 环境)

```bash
python -m venv /tmp/venv_tb_graph
source /tmp/venv_tb_graph/bin/activate
pip install tb-graph-ascend tensorboard
```

不需要装 mindstudio-probe 全套或 dcu_probe whl —— **浏览侧零硬件依赖**,可在任意 Mac / Linux / Windows 机器跑。

### 启动浏览

```bash
# vis.db 可直接放 logdir,无需 events file
tensorboard --logdir <dir_containing_vis_db> --port 6006
# 浏览器打开 http://localhost:6006/#graph_ascend
```

### ✅ 2026-05-13 Mac 浏览侧实证(本仓库)

| 维度 | 数据 |
|---|---|
| Mac 工具栈 | `tb-graph-ascend 8.3.2`(2026-02-04 release)+ `tensorboard 2.20.0` |
| 输入 | scnet 跑出的 `build_20260513114445.vis.db`(llama2-7B L0 dump,**580 MB**),scp 到 `~/Downloads/vis_db_review/` |
| 启动 | `tensorboard --logdir ~/Downloads/vis_db_review --port 6006` |
| `plugins_listing` 接口 | 17 个标准插件 disabled + `graph_ascend: enabled=True` ✅ |
| GRAPH_ASCEND Tab | 完整工作 |

界面要素全部正常渲染:

- **左侧设置面板**:语言切换、调试侧缩略图、目录(`vis_db_review`)、文件(`build_20260513114445`)、Step、Rank、MicroStep
- **中间图形区**:`Module.module.Float16Module.forward.15-19` 等连续节点正常下钻(BF16 训练 wrapper),双击可层级展开
- **顶部图例**:模块/算子、不可展开节点、游离 API 集合、融合算子集合
- **底部节点详情表**:目标节点(例如 `Module.module.Float16Module.forward.60`)输入/输出 tensor 的 `type` / `dtype` / `shape` / `Max` / `Min` / `Mean` / `Norm` / `requires_grad`
- **dtype 识别**:`torch.bfloat16` / `torch.float32` / `torch.int64` / `torch.bool` 全部正常显示
- **shape 识别**:`[1, 4096]` / `[4096, 1, 4096]` 等 llama2-7B seq=4096 / hidden=4096 形状

→ **DCU 训练数据 + tb_graph_ascend 浏览插件完整兼容,跨平台开箱即用**。

### 命名说明(`Ascend` 残留)

`tb_graph_ascend` 插件名 + URL Tab `#graph_ascend` + 视图标题 `GRAPH_ASCEND` 都沿用上游 mindstudio-probe 项目对**华为昇腾(NPU)**的品牌命名。但**插件功能跟硬件解耦**(纯 SQLite 读取 + Vue/D3 前端渲染,源码内无 NPU 硬编码),DCU 训练数据完全 work,无需修改。

跟我们 dcu_probe whl 沿用 `from msprobe.xxx` 内部 import 路径是同性质决策:**功能可用 ≠ 命名要重做**。

### 如果未来希望去掉 Ascend 字样

| 项 | 数据 |
|---|---|
| 上游源码 | `https://gitcode.com/Ascend/mstt/tree/master/plugins/tensorboard-plugins/tb_graph_ascend` |
| License | **Mulan PSL v2 / BSD**(允许 fork + 改名 + 重发布,需保留版权信息) |
| Author | Ascend Team(华为) |
| 当前版本 | 8.3.2(2026-02-04) |

改名思路(若决定做):fork → 改 plugin entry point 名(`tb_graph_ascend` → `tb_graph_dcu_probe`)+ 改 URL Tab + 改视图标题 → 重打包发布。**代价**:每次上游 release 需 rebase 改名 patch,short-term 不建议。**当前直接用上游 PyPI 包即可**。

## 与其他工具的关系

- 上游 `overflow_check`(本仓库 `tools.overflow`)输出 anomaly node 后,在 graph_viz 里用 `-oc` 直观叠加显示
- 跟 `compare` 互补:compare 出 xlsx 表格,graph_viz 出图形化逐层下钻视图
- 跟 `data2db` 互补:graph_viz 看**单 step 模型结构 + 精度**,data2db 看**跨 step / 跨 rank 的趋势**

## 完整能力参考(上游 PyTorch 文档)

dcu_probe 的 graph_visualize 实现完全沿用上游 mindstudio-probe v26.0.0-alpha.1 的 PyTorch 路径代码。以下细节本文档没展开,需要时查上游 PyTorch 场景文档:

- 完整 7 个工具特性(重建层级 / 结构差异 / 精度数据比对 / 溢出检测 / 多卡批量构图 / 节点搜索筛选 / 跨套件跨框架比对 / 切分策略图合并)
- 节点命名格式 `{Module}.{module_name}.{class_name}.{forward/backward}.{调用次数}`
- 跨套件比对(DeepSpeed vs Megatron / MindSpeed vs LLamaFactory)实战 mapping yaml 例
- `layer_mapping` 自定义映射文件配置规范
- 三种 dump 类型(`statistics` / `tensor` / `md5`)在图比对中的差异
- 模糊匹配 vs 默认匹配的具体算法说明
- 不同切分策略下的图合并:`rank_size` / `tp` / `pp` / `vpp` / `order` 参数配置案例
- 浏览器交互(节点搜索、精度筛选、溢出筛选、自动跳转展开)
- TensorBoard 服务端口配置 / 多种网络场景下的连接方式(直连 / vpn / vscode 端口转发 / 本地 PC 加载)
- 溢出标注 3 级 (`medium` / `high` / `critical`) 判定规则
- 单 rank / 多 rank / 多 step 三种 target_path 目录结构格式

上游文档地址:

```
https://gitcode.com/Ascend/msprobe/blob/26.0.0-alpha.1/docs/zh/accuracy_compare/pytorch_visualization_instruct.md
```

> ⚠️ **不要参考同目录下 `mindspore_visualization_instruct.md`** — MindSpore 专属,模块名命名、framework 字段、tensorboard 插件名称都不一样。
