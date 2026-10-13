# compare 盲区原理(为什么必须用 verify_consistency.py、不能用自带 compare)

> 本文是 SKILL.md "二、hcu_probe 工具" B/C 小节的深挖层 —— 讲清 `scripts/verify_consistency.py` **为什么那样写**。
> 接入(dump + 跑脚本)不读本文也能完成;要**理解 / 改 / 排查** compare 时读。
> 本文为 skill 自包含的 compare 原理副本(原始底稿为内部记录,对外仓可不带)。

---

## 一句话

hcu_probe 自带 compare 在 verl(HF 训练 ↔ das vllm 推理)场景**不能开箱用**:第一次拿同样 dump 跑会满屏假 diff、误判"严重不一致"。根因是 das vllm 为吞吐做的优化改变了 dump 语义。`verify_consistency.py` 是绕过盲区、只筛严格可比算子的专用脚本。

---

## 三个盲区(跑自带 compare 会踩)

### 盲区 1 — vllm fused residual(最坑,das 特有)

vllm 的 `Qwen2DecoderLayer` 为吞吐返回 `(normed_hidden, residual)` **两个** output,HF(训练侧)只返回**一个**完整 hidden。自带 compare 的 `consistent_match` 把：

- `HF.output[0]`(完整 hidden,Max≈**1688**) 错配到
- `vllm.output[0]`(normed,Max≈**2**)

→ 制造 **1686 的假 diff**。**整层那些骇人的大 diff 全是工具不认 fused residual tuple 的伪问题,不是真发散。**

> ⚠️ das vllm 特有，华为 vllm-ascend 的 `DecoderLayer` 不返回 tuple、无此盲区。**vanilla compare 因此不能直接用**（逼我们自写 `verify_consistency.py`）；但官方 `pt_compare(consistent_check=True, backend="fsdp")` 在**叶子算子粒度**配对、把整层作排除项，反而绕过本盲区 —— 过滤 das 交错模块后能用（见 SKILL §D）。

### 盲区 2 — vllm batch prefill(粒度不一致)

vllm 把多 prompt 的 prefill 拼一批算，训练侧 `micro_batch=1` 是单 prompt → 两侧粒度不同、多数对不可比。

- **华为官方解法**：dispatch_log 选 `phase=prefill` 且 `requests` 数=1 的（只比单请求 prefill）。
- **我方额外努力**：`max_num_seqs=1` 重跑，让 80 个 request 全单独 prefill、全可比（超出华为方案）。

### 盲区 3 — dual_monotonic_sort 报错

自带 compare 的 `dual_monotonic_sort`（内置 `FSDP_MODULE_MAP` 只 5 条映射）报错，有**两个成因**：① 盲区 1/2 的连带症状（vanilla 无 consistent_check 时）；② **das vllm 0.18.1 版本特有的独立成因** —— 它多 dump 72 个训练侧无对应的交错模块（`Triton.kernel_unified_attention`×24 + `reshape_and_cache`×24 + `rotary_emb`×24），夹在推理序列中间让两列位置无法同时单调递增 → **即便开 consistent_check 也 raise**。修法 = 复核前过滤这 72 个 bench-only 模块（`scripts/run07_filtered.py`），再喂官方 pt_compare（见 SKILL §D）；某些 das vllm 版本模块序干净则无此问题。

---

## 解法:只比 normed 算子(layernorm output)

`verify_consistency.py` **不比全量**（虽然 dump 是全量 L0），只挑两侧严格可比的：`embed` 的 output + 每层 `input_layernorm` / `post_attention_layernorm` output 的 **Norm 相对误差**。

| 算子 | 两侧对应 | 能比吗 |
|---|---|---|
| **layernorm output**(normed） | 单 output、语义一致、不受 fused residual tuple 影响 | ✅ **比它** |
| 整层 DecoderLayer output | vllm `(normed, residual)` tuple vs HF 单 output | ❌ 排除（假 diff 几千倍） |
| 残差流 | `HF.output[0] ↔ vllm.output[1]` 理论成立但不普遍（实测 layer0 整层 900%+） | ❌ 不单独比 |

**够可靠**：embed + 每层 2 个 layernorm 覆盖网络每层的归一化节点；每层归一化输出 <1% 一致 = 各层前向数值一致。且 layernorm 的输入就是残差流，输出一致间接保证残差流也一致，不必单独比。

---

## 数据流层面的根:为什么 norm 可比、整层 output 不可比

一个 token 的 hidden 进到第 N 层 DecoderLayer：

- **HF(非 fused,训练侧)—— 残差加法在层内完成**：`input_layernorm → attn → +残差 → post_layernorm → mlp → +残差` → 输出**单个完整 hidden**（含所有残差，Max≈1688）。
- **vllm(fused residual,推理侧)—— 残差加法融合进 norm（`fused_add_rms_norm`）、残差流单独传、最后一次加法延迟到下一层**：`fused_norm(h,r): r += h; h_norm = rmsnorm(r) → attn → fused_norm → mlp` → 输出 **(mlp 输出 m, 累积残差 r) 两个**（完整 hidden = m + r，加法到下一层才做）。

| | HF(非 fused） | vllm(fused） | 对 compare 的影响 |
|---|---|---|---|
| 残差加法 | 层内完成 | 融合进 norm + 延迟到下层 | — |
| 整层 output | 1 个 = 完整 hidden | 2 个 =(mlp 输出, 残差)，完整 = 两者和 | ❌ 不能 `out[0]↔out[0]` 比（1688 vs 2） |
| **layernorm output** | `rmsnorm(完整残差)` | fused norm 内部也算 `rmsnorm(完整残差)` | ✅ 两侧一致，挑它比 |

> **锚定:norm 是「检查点」、output 是「交接点」**。
> - 检查点（norm）：vllm 在 fused_norm 里先 `r=r+h` 凑齐完整 hidden 再归一化 → 和 HF 同一个数 → ✅ 可比。
> - 交接点（output）：vllm 把完整 hidden 拆成 `(m,r)` 两半传给下层 → 形式不同 → ❌ 不可比。
> - fused residual 的「拆」只发生在交接处、不在检查点 → 专挑 norm 比就绕开了。

---

## 为什么不直接改 hcu_probe core 去认 fused residual

compare 逻辑（`consistent_match` / verl `mapping.py`）全在 `src/hcu_probe/core/compare/` —— 与华为 msprobe 上游**字字一致**的区。改它技术可行，但**不该直接改 core**：

1. 破坏 core 字字一致、未来跟上游 merge 冲突；
2. fused residual 是**社区 vllm 通病、非 HCU 特有**，改在 HCU core 不合理。

**正道**：给华为 msprobe **上游提 PR**（上游修、hcu_probe 跟着有）；HCU 这边短期把 `verify_consistency.py` 当专用脚本（归 `integrations/verl` 或 examples 仓），不污染 core。

---

## 延伸:compare 只 L0 可行(L1/tensor 受限)

**采集（dump）≠ 比对（compare）**。6 模式都能采集 dump，但训推 compare 只 L0 可行：

- **L1（API 级)训推算子对不齐(机制层面,未在 verl-0.7 实测)**：训练 = HF Qwen2 + FSDP（`Distributed.all_gather` + HF 算子分解），推理 = das vllm（`Torch.addmm` + 融合 kernel），异构框架 API 序列根本不重叠 —— RL 训推（训练框架 ≠ 推理框架）的本质限制。故 L1 训推 compare 预期不可行(属机制推断,未实跑坐实)。→ L1 适合**单侧** API 级精度剖析(需要时自行实跑验证)。
- **tensor（逐元素)推理侧爆 I/O**：推理侧逐元素 dump 每个 prefill 写几百 `.pt`，拖垮 rollout（实测推理钩子在时 10min+ 不出 step）。→ tensor 适合**单侧**逐元素留档。
- **L0（module 级 normed 算子)是训推 compare 最优粒度**：两侧都有 layernorm、语义对齐、不爆盘（实测 80/80）。
