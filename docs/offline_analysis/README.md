# dcu_probe 离线分析工具

`dcu_probe.tools.*` 是 dump 数据**离线**分析工具集(训练之外另跑),用于比对 / 溢出定位 / 可视化 / 精度预检等场景。跨平台、纯 Python,与训练框架无关。

> 训练时的 dump / monitor 接入见 [`../integrations/`](../integrations/)。

## 工具一览

| 工具 | 用途 | DCU 实证 | 文档 |
|---|---|---|---|
| `compare` | dump 数据两端比对(DCU vs H20 / 任意双端) | ✅ scnet docker llama2-7B dump 跑通(2026-05-12,8×24MB xlsx) | [compare.md](compare.md) |
| `overflow` | 整网首个溢出节点定位(扫 NaN/Inf,跨 rank 关联) | ✅ scnet docker 跑通(2026-05-12,8 卡 dump,无 anomaly 正常返回) | [overflow.md](overflow.md) |
| `graph_viz` | dump 图可视化(单源构图 / 双源 diff,tensorboard 加载) | ✅ scnet docker 跑通(2026-05-12,608MB vis.db) | [graph_viz.md](graph_viz.md) |
| `config_check` | 训练环境快照采集与比对 | ✅ scnet docker 跑通(2026-05-12,产物 zip 落盘);DCU 特有环境变量清单仍待沉淀 | [config_check.md](config_check.md) |
| `precheck` | API 精度预检三件套(acc_check / multi_acc_check / api_precision_compare) | ✅ scnet node119 BW200 跑通(2026-04-13) | [precheck.md](precheck.md) |
| `merge_result` | 合并多卡 compare 结果 | 🟡 跨平台理论可用,无独立实证 | — |
| `parse` | dump 数据解析 | 🟡 跨平台理论可用,无独立实证 | — |
| `data2db` | dump / monitor 数据导入 SQLite,**趋势可视化入口** | 🟡 跨平台理论可用,无独立实证 | [data2db.md](data2db.md) |

## 运行环境(标准:docker 容器)

dcu_probe.tools.* 工具的**标准运行环境是 docker `dcu_probe_lab` 容器**,**不在主机 conda env**。

| 维度 | docker 容器 | 主机 conda env(`megatron_train`) |
|---|---|---|
| 适用 | ✅ 推荐 | ❌ 不推荐 |
| 实证 | ✅ 4 个核心工具(compare / overflow / graph_viz / config_check)scnet docker 端到端跑通(2026-05-12) | ⚠️ 仅 `import` 路径验证,未在真实 dump 数据上跑过工作流 |
| deps 状态 | 通过 `_cp_all_deps.sh` 从主机一次性 cp(tensorboard / openpyxl / et_xmlfile 等) | 海光为训练装的版本(pandas 2.2.3 / protobuf 5.29.4)与 dcu_probe pyproject 约束(`<2.1` / `<=3.20.2`)版本冲突 |
| 隔离性 | 与训练侧物理隔离,改 deps 不影响训练 | 跟训练共用 deps,版本冲突风险 |

环境就绪步骤(假设已 salloc 到计算节点 + 启动容器):

```bash
# 容器镜像 / 启动脚本(在 ~/dcu_megatron_clean/docker_dcu_probe_lab.sh)
镜像: triton-distributed:backup-20260324
容器: dcu_probe_lab
启动: bash ~/dcu_megatron_clean/docker_dcu_probe_lab.sh remote <node> setup
装 whl: bash ~/dcu_megatron_clean/docker_dcu_probe_lab.sh remote <node> inside
补 deps: docker exec dcu_probe_lab bash /tmp/_cp_all_deps.sh
```

> **训练侧**(`integrations/megatron/dump` / `monitor` / `DcuProbeCallback`)仍在主机 conda env 跑,**本节规则仅适用本目录(离线分析侧)**。

## 两种入口

### 1. Python API

```python
from dcu_probe.tools import compare, overflow, graph_viz, config_check, precheck

compare.run(target_path="./dcu", golden_path="./h20", output_path="./diff")
overflow.run(input_path="./dump/step_0", output_path="./out")
```

### 2. CLI

```bash
dcu-probe compare -tp ./dcu -gp ./h20 -o ./diff
dcu-probe overflow_check -i ./dump/step_0 -o ./out
dcu-probe graph_visualize -tp ./dump -o ./vis_out
dcu-probe acc_check -api_info ./dump.json -o ./out
# ... 完整子命令列表见各工具文档
```

## 浏览侧 tensorboard 插件(whl 不含,需另装)

`graph_viz` 和 `data2db` 工具**生成 `.vis.db` / `.trend.db` 都已在 dcu_probe whl 内**。但要在 tensorboard 浏览器界面看图,需要**另装两个上游 tensorboard 插件**:

| 插件 | 服务 | 用途 |
|---|---|---|
| `tb_graph_ascend` | tensorboard 左上 `GRAPHS` 标签 | 分级可视化构图浏览(`.vis.db`) |
| `trend_analyzer` | tensorboard 左上 `TREND ANALYZER` 标签 | 趋势可视化浏览(`.trend.db`) |

部署方式:在**分析机**(本地 PC 或浏览专用服务器)从上游 mindstudio-probe 编译带 `--include-mod tb_graph_ascend,trend_analyzer` 的 wheel + 装到 tensorboard 同 venv;dcu_probe 生成的 db 文件 rsync 过去后用 tensorboard 加载。

## 趋势可视化(澄清)

之前一度判断"趋势可视化上游 v26.0.0-alpha.1 未实装" — **这是错的**。趋势可视化**没有独立 CLI 子命令**,入口就是 `data2db` 命令 + tensorboard `TREND ANALYZER` 插件的组合。`data2db` 已在 dcu_probe whl 内,详见 [data2db.md](data2db.md)。
