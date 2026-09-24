# HCU Probe 推理精度调试指南

hcu_probe 在推理场景下提供**数据采集（dump）→ 精度比对（compare）→ TensorBoard 可视化（visualize）**三级能力，覆盖 SGLang、vLLM 两种主流推理框架。

## 完整链路

```
推理代码接入 PrecisionDebugger → dump 数据落盘 → hcu-probe compare 比对 → hcu-probe graph_visualize 构图 → TensorBoard 浏览
```

## 文档导航

| 文档 | 内容 |
|---|---|
| [SGLang 推理精度调试](./sglang.md) | 两种接入方式（手动 / 原生），PD 分离，dump → compare → visualize 全流程 |
| [vLLM 推理精度调试](./vllm.md) | GPUModelRunner 接入 dump → compare → visualize 全流程 |
| [配置与产物参考](./config.md) | config.json 参数全表、dump 目录结构、L0 / L1 命名规则 |

## 前置条件

- 安装 `hcu_probe`：参见 [安装指南](../../README.md#安装)
- 推理环境已安装对应框架（SGLang / vLLM）和 HCU 版 PyTorch（`torch` 不在 hcu_probe 依赖中，需自行安装）
- **离线分析工具**（`hcu-probe compare`、`hcu-probe graph_visualize`、`hcu-probe data2db`）建议在 docker `hcu_probe_lab` 容器中运行，参见 [工具运行环境](../offline_analysis/README.md#运行环境标准docker-容器)

## 两个子系统

| 子系统 | 用途 | 对应命令 / 接口 |
|---|---|---|
| **Dump** | 采集模型运行时张量统计数据 / 真实 tensor 数据 | `hcu_probe.pytorch.PrecisionDebugger` |
| **离线分析** | dump 数据比对 + 可视化构图 + 趋势分析 | `hcu-probe compare`、`hcu-probe graph_visualize`、`hcu-probe data2db` |
