# 上游同步说明（hcuprobe ← msProbe）

hcuprobe 是华为 [`mindstudio-probe`（msProbe）](https://gitcode.com/Ascend/msprobe) 的 HCU 平台适配 fork。本文记录**当前对齐的上游版本**、**哪些子树与上游一致、哪些是 HCU 侧自维护**，以及**如何重新同步**。升级上游时更新本文。

## 当前同步状态

| 项 | 值 |
|---|---|
| 本包 名称 / 版本 | `hcuprobe` / `26.2.0+hcu.0.2.0` |
| 上游项目 / 版本 | 华为 msProbe（mindstudio-probe）`26.2.0` |
| 上游仓 | `https://gitcode.com/Ascend/msprobe`（分支 `26.2.0`） |
| 对齐上游 commit | `ac322b6f`（2026-09-27） |
| 本仓基线 commit | `2c7776d5`（历史整理后哈希不同，树内容与上游对齐点完全一致） |
| 命名空间 | 上游 `msprobe.*` → 本包 `hcuprobe.*` |
| CLI 命令 | 上游 `msprobe` → 本包 `hcuprobe`（`hcuprobe.cli:main`） |

## 子树对齐映射

| `src/hcuprobe/` 子树 | 与上游关系 |
|---|---|
| `core/` `pytorch/` `visualization/` `overflow_check/` `msaccucmp/` | **与上游内核一致**（仅命名空间 `msprobe→hcuprobe` + HCU 侧适配改动） |
| `integrations/` | **HCU 侧自维护**的多框架训练接入翻译层（上游无对应，接入操作见各框架 skill） |
| `tools/` + `cli.py` | HCU 侧离线分析工具封装与 `hcuprobe` CLI 入口 |

### 上游子树内的 HCU 适配点（同步上游时需保留）

| 文件 | 改动 | 原因 |
|---|---|---|
| `infer/offline/compare/msquickcmp/cmp_process.py` | 导入 `acl` 失败时 `logger.error` → `logger.debug` | `acl` 是昇腾 CANN 模块，HCU 上必然缺失；该模块在 CLI 启动时即被导入，原报错会让每条 `hcuprobe` 命令都打印 "Please verify that the CANN environment is properly configured." |

verl 训推一致性比对的 HCU 修正（statistics 模式 squeeze 容差、剔除 `Triton.*` 条目）不改 `core/`，由 `integrations/verl/compare.py` 提供。

## License 与署名

- **代码**：上游采用[木兰宽松许可证 第2版（Mulan PSL v2）](https://gitcode.com/Ascend/msprobe)。
- **文档**：上游 `docs/` 采用 **CC-BY 4.0**。⚠️ **若从上游文档复制内容到本仓，CC-BY 4.0 要求保留对 msProbe 的署名**——复制型文档须注明来源；如需完全独立品牌的说明，应改为 HCU 侧原创撰写（不复制上游文本）。

## 进一步参考文档（框架接入之外的功能）

本仓各能力**优先看 HCU 化的自有文档**（用 `hcuprobe` CLI、含 HCU 实证状态）：

| 功能 | 本仓文档 |
|---|---|
| 两端精度比对 `compare` | [`docs/offline_analysis/compare.md`](offline_analysis/compare.md) |
| 溢出/首个 NaN 定位 `overflow_check` | [`docs/offline_analysis/overflow.md`](offline_analysis/overflow.md) |
| API 精度预检 `acc_check` / `multi_acc_check` / `api_precision_compare` | [`docs/offline_analysis/precheck.md`](offline_analysis/precheck.md) |
| 训练环境快照比对 `config_check` | [`docs/offline_analysis/config_check.md`](offline_analysis/config_check.md) |
| dump 图可视化 `graph_visualize` | [`docs/offline_analysis/graph_viz.md`](offline_analysis/graph_viz.md) |
| dump/monitor 入库 SQLite `data2db` | [`docs/offline_analysis/data2db.md`](offline_analysis/data2db.md) |
| 推理侧 dump（vllm / sglang） | [`docs/infer/`](infer/) |
| 常见问题 | [`docs/FAQ.md`](FAQ.md) |

**本仓尚未覆盖的上游功能**（如 `checkpoint_compare`、`response_anomaly`、monitor 深入、accuracy_checker/accuracy_compare 细节等）可参考上游同版本 user_guide：`https://gitcode.com/Ascend/msprobe`（`docs/zh/user_guide/`）。

> ⚠️ **对照上游文档时注意 HCU 映射**：上游用 `msprobe` 命令 → 本仓是 `hcuprobe`；上游含 NPU/昇腾专属能力（MindSpore、torchair、atb、aclgraph 等）**在 HCU 上不适用**，只取 PyTorch 相关章节。

## 重新同步上游（升级版本时）

```bash
# 1. 拉上游参考 clone 最新(diff 用,不进 git)
cd upstream_msprobe && git pull   # 或另 clone gitcode.com/Ascend/msprobe

# 2. 选定要对齐的上游 commit/tag,与本仓 src/hcuprobe/{core,pytorch,visualization,overflow_check,msaccucmp} 逐子树 diff
#    只并入内核改动;integrations/ 与 tools/ 是 HCU 自维护,不覆盖

# 3. 保持命名空间改写(msprobe.*→hcuprobe.*)与 HCU 适配点

# 4. 更新本文「当前同步状态」的版本 / 对齐 commit,bump pyproject 的 +hcu.X.Y 后缀
```
