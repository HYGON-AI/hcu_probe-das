# verl FSDP 后端训推一致 dump 训练侧 NaN 根因定位与修复(HCU / verl 0.8.0)

> **一句话结论**：HCU 上 verl FSDP 后端用默认 `flash_attention_2`，会命中 das flash-attn 包里**非 varlen** 的 `flash_attn_func` —— 该 kernel 在 hybrid(训推共卡）运行时态下产 NaN，不仅训练侧 dump 几乎全 NaN，**连普通训练的 grad_norm 都每步 NaN（训练根本不学）**。把 actor 的 `attn_implementation` 改成 **sdpa**（或 eager）即同时修好训练与 dump：训练 grad_norm 恢复有限、reward 正常上升；训推逐层一致性在排除融合/整层算子后，可比叶子算子 **97/98 < 1%**。机制已用 tripwire 钉死：是**运行时态问题，不是 kernel 数学 bug**（把出 NaN 那次的确切入参拿出来 standalone 重放是干净的）。

---

## 1. 验证场景与定界

- **目标**：用 hcu_probe 验证 verl FSDP 后端的**训练前向**与**推理前向**（vllm）在 prompt prefill 上逐层数值一致。
- **配方**：单节点 8 卡，Qwen2.5-0.5B-Instruct + gsm8k，verl 0.8.0 FSDP 后端（`--config-name=ppo_trainer`），Hybrid AgentLoop（训推共卡）。
- **比对定界**：训练侧 dump（actor 前向，钩子在 `FSDPEngine.forward_backward_batch` 循环体,开 `PROMPTS_ONLY` 只算 prompt）vs 推理侧 dump（vllm `execute_model` prefill），按 request_id 配对、逐 module 比 `Max/Min/Mean/Norm`。

> ⚠️ **两个正交的事别混**：①「训推一致性」= 两个引擎逐层 prefill 数值是否一致（dump+compare 验）；②「训练健康」= 训练是否收敛、loss/grad 是否正常（普通训练验）。dump 必开 `PROMPTS_ONLY`（把 response 裁成 0），训练在 dump run 里**本就是退化的**，所以一致性通过 ≠ 训练正常 —— 这次的核心教训就来自这里。

## 2. 现象

训练侧 dump（`task=statistics, level=L0`）：**317 个 module 里 311 个含 NaN**，第一个 NaN 精确在：

```
layers.0.self_attn.o_proj.Linear.forward.0  的 input_args[0]  (= attention 核输出)
```

而它之前的算子**全部有限**：

| module | 输出 Max/Min | 状态 |
|---|---|---|
| embed_tokens | 0.071 / -0.088 | ✓ |
| layer0.input_layernorm | 2.27 / -2.91 | ✓ |
| layer0.q_proj | **50 / -80.5** | ✓（Qwen2.5 第0层固有大激活） |
| layer0.k_proj | **121.5 / -130** | ✓ |
| layer0.v_proj | 0.30 / -0.20 | ✓ |
| **layer0.o_proj 的 input（attention 输出）** | **NaN** | ✗ ← 第一个 NaN |

**8 个 rank、所有 step 100% 确定性同一点；全程零 Inf**（不是渐进溢出，是 attention 核内部凭空产生）。

## 3. 排查思路与单变量对照（决定性）

> 教训前置：早期用 ~11 个 standalone repro「全部复现干净」就过早断言「不是 kernel」——**这是错的**：那些 repro 没碰到真实 run 的 flash 路径，也测错了函数（详见 §5）。真正定位靠的是**真实 run 内的单变量对照**。

在真实 run 里只改一个变量，看 NaN 是否翻转：

| attn_implementation | trainer.balance_batch | layer0 attention dump |
|---|---|---|
| flash_attention_2 | True（原始） | **NaN（311/317）** |
| flash_attention_2 | **False（官方文档要求）** | **NaN（311/317），完全一样** |
| **eager** | True | **干净（0）** |

**唯一翻转 NaN 的变量是 `attn_implementation`**。这同时证伪了几个旧假设：与 `use_remove_padding` / `PROMPTS_ONLY` / `seed_all(mode=True)` / `balance_batch` **都无关**。

## 4. ★ tripwire 机制钉死（kernel 数学 bug vs 运行时态）

为彻底分清「是 kernel 对这组输入的数学 bug，还是运行时态问题」，用 `sitecustomize.py`（放进 `PYTHONPATH` 头部，所有 ray worker 进程自动加载）包住三层函数，首次「输出 NaN」时把**该次调用的全部入参**存盘：

- `transformers.modeling_flash_attention_utils._flash_attention_forward`（verl `_ulysses_flash_attention_forward` 实际调的入口）
- `flash_attn.flash_attn_func`（非 varlen）
- `flash_attn.flash_attn_varlen_func`（varlen）

**真实 run 抓现场结果**：

| 包住的函数 | 是否产 NaN |
|---|---|
| `_flash_attention_forward`（入口） | **NaN**（全 8 rank） |
| **`flash_attn_func`（非 varlen）** | **NaN** ← 真凶 |
| `flash_attn_varlen_func`（varlen） | **不 NaN（没被触发）** |

→ 真实 run 对「rmpad 后的单序列」走的是 **`flash_attn_func`（非 varlen）**，不是我一直怀疑的 varlen。

**把 `flash_attn_func` 出 NaN 那次的确切入参（q Max=49.5 / k Max=121.5 / v Max=0.30，`in_finite=True`，`autocast=True`）拿出来 standalone 重放**：

| 重放条件 | 结果 |
|---|---|
| no autocast | **干净**（Max=0.3047，无 NaN） |
| autocast(bf16)（还原真实上下文） | **干净** |
| sdpa 参照 | **干净** |

**结论**：既不是 kernel 对该输入的**数学 bug**（确切输入重放干净），也不是 **autocast**。是 das `flash_attn_func` 在 **live 多进程 hybrid 运行时态**（并发 vllm 共卡 + `HSA_NO_SCRATCH_RECLAIM` 下 scratch/显存态/竞争）被污染产 NaN —— 同一 kernel、同一确切输入，**孤立跑干净、真实运行时出 NaN**，指向 scratch/显存处理的脆弱，而非数值溢出。

## 5. 修法：attn_implementation → sdpa（或 eager）

`transformers` 的 `attn_implementation` 三档：

| 档 | 实现 | 在 HCU 上 |
|---|---|---|
| `flash_attention_2` | das `flash_attn` 包（`flash_attn_func` 非 varlen） | **NaN（坏）** |
| **`sdpa`** | PyTorch 原生融合 `scaled_dot_product_attention` | **干净，且更贴 vllm** ← 推荐 |
| `eager` | 纯手写 matmul+softmax | 干净，但慢、数值离 vllm 略远 |

**纯配置改动**（verl `workers/config/model.py:184` 把 `override_config.attn_implementation` 透传给 `AutoConfig.from_pretrained`，无需改码）：

```bash
+actor_rollout_ref.model.override_config.attn_implementation=sdpa
```

eager 和 sdpa 都验证可消 NaN；**sdpa 更优**（见 §6 逐层一致性对比 + 更快）。

## 6. 逐层一致性 compare 结果

修好 NaN 后 compare 终于能跑（之前全 NaN 不可比）。**按 megatron 口径排除结构性不可比项**后（见 §8B），可比叶子算子：

| | eager | **sdpa** |
|---|---|---|
| 可比叶子算子数 | 98 | 98 |
| **< 1%** | 96/98 | **97/98** |
| < 0.5% | 85 | 92 |
| < 0.1% | 24 | **43** |
| 中位误差 | 0.199% | **0.121%** |
| o_proj 中位 / max | 0.34% / 2.67% | **0.14% / 1.78%** |
| embed 两侧 l2norm | 完全相同（误差 0） | 完全相同（误差 0） |

- **sdpa 全面更贴 vllm**（尾巴收紧、<0.1% 翻倍）→ sdpa 是比 eager 更优的修法。
- 仅剩最深 layer22 o_proj ~1.78%（训练 sdpa attn vs 推理 vllm attn 本质是**不同 kernel**，固有小尾巴，消不到 0）。
- 方法论上已与 megatron 后端的 99.4% < 1% 同档。

## 7. 训练健康对照（关键，且两后端都曾漏验）

跑**不带 hcu_probe（`DUMP_ON` 关）、不带 `PROMPTS_ONLY` 的普通训练**，读 tensorboard：

| 指标（前若干步） | 默认 flash | **eager / sdpa** | megatron（TE） |
|---|---|---|---|
| **actor/grad_norm** | **每步 NaN** | 0.004 ~ 1.5 有限 | 0.0017 ~ 0.0026 有限 |
| pg_loss / kl_loss | 间歇/大多 NaN | 有限稳定 | 有限稳定 |
| **critic/rewards/mean** | 卡 0（不学） | **0.0125 → 0.15（在学）** | 0（response_length=64 太短，非健康问题） |

→ **flash 的 NaN 不是 dump 路径特有，它让真实 FSDP 训练每步梯度 NaN、reward 卡死、根本不学**（grad 被 clip/skip 所以 run 不崩但不学）。改 sdpa/eager 后训练健康、reward 正常上升。

> ⚠️ verl 的训练 metric **只落 tensorboard，console 不打** → 用 `EventAccumulator` 读 `actor/grad_norm` 等，别只看 "Training Progress" 进度条（进度条照走 ≠ 在学）。

## 8. ★ hcu_probe 工具用法（dump + compare）

### A. hcu_probe dump 工具

- **引擎**：`from hcu_probe.pytorch import PrecisionDebugger`；`debugger.start(model=self.module)` / `debugger.stop()` / `debugger.step()` 围在前向前后。
- **钩子点（FSDP 后端）**：`FSDPEngine.forward_backward_batch` 里有显式 `for micro_batch in micro_batches` 循环，在循环体内包 debugger（与官方文档 `verl_async_consistency_preprocess_dump.md` 的 FSDP 节一致）。推理侧复用容器内 vllm `gpu_model_runner` patch。
- **config.json**（训练侧示例，`task=statistics` 免爆盘 / `level=L0` module 级）：

```json
{"task":"statistics","dump_path":".../dump_actor","rank":[],"step":[],
 "level":"L0","async_dump":false,
 "statistics":{"scope":[],"list":[],"tensor_list":[],"data_mode":["all"],"summary_mode":"statistics"}}
```

- **产物**：`dump_actor/stepN/rankM/dump.json`（每次 forward 逐 module 统计）+ `{pid}/update_actor_log.jsonl`（训练侧 request_id）；`dump_generate/{pid}/stepN/rankM/dump.json` + `dispatch_log.jsonl`（推理侧）。靠 **request_id 同源配对**。
- **环境前提**：`DUMP_ON=1 PROMPTS_ONLY=1 DUMP_PHASE=log_prob`；推理侧 `enforce_eager=True`（否则 cudagraph 绕过钩子 dump 空）；Megatron-LM 必须在 `PYTHONPATH`（verl_adaptor 硬 import megatron，FSDP 后端也要）。

### B. hcu_probe compare 工具 —— 能不能用自带 + 怎么用

- **能用自带 `pt_compare`，但要补两层**：
  1. **shape 盲区（修法B）**：FSDP 训练 dump 是 `[1,S,H]`（HF eager unsqueeze），vllm 推理是 `[S,H]`（packed）。hcu_probe 的 squeeze 容差 `VerlShapeErrChecker` **只挂 `task=tensor`**，`statistics` 模式恒用严格 `ShapeErrChecker` → 全 shape error。**修法**：改 hcu_probe `core/compare/indicator_analysis/{algorithm,calculator}.py`，让 statistics + consistent_check 也用 `VerlShapeErrChecker`（去 `RequiresGradErrChecker`），纯 .py 覆盖容器 dist-packages 即可，不重打 whl。
  2. **报告过滤（排除结构性不可比项）**：`pt_compare(..., backend="fsdp")` 自动用 `FSDP_MODULE_MAP`。但「真算子」过滤**必须再排除**：① 整层 wrapper（`Qwen2DecoderLayer/MLP/Attention` 的 output = 残差和，两侧结构不同，relative err 失真）② `IdentityOp` 融合占位 ③ **`act_fn` 融合算子** —— HF `act_fn=silu(gate)` 单独，vllm `act_fn=SiluAndMul=silu(gate)·up`，**算的是不同张量**（误差 80-141%），真正等价的比较点是 `down_proj`（两侧都吃 `silu(gate)·up`，< 1%）。
- **用法**：`python3 run_fsdp_compare.py [run目录]`（容器内跑，dump.json 是容器 root 0750）。脚本按 request_id 配对、挑 token 完全匹配样本、`pt_compare` npu=训练/bench=推理、再按上述排除项报真算子 NormRelativeErr 分布。
- **坑**：hcu_probe 对 Triton 融合算子（如 flash-attn 的 `cross_entropy_fwd_kernel`）会把**未初始化的 scratch 输出缓冲**也统计进 Max/Min → 产**假 NaN**。排查时这类 in-place 写出的 kernel **只看 output[0]（真 loss）**，别被 scratch buffer 误导。

## 9. 为什么 megatron 后端不 NaN（对比）

给 megatron 普通训练**也挂同一 tripwire**，跑几步：**零 `[TRIPWIRE] FIRST CALL`**（flash_attn 包任何函数都没被调），日志 `transformer_engine/pytorch` × 8。

→ **megatron 用 TransformerEngine(TE) 的融合 attention，根本不碰 das `flash_attn` 包**（mcore `attention_backend:'flash'` 路由到 TE 的 flash-style 融合，和 `flash_attn` 包是两套独立代码）。所以差异是 **flash_attn 包（FSDP）vs TE（megatron）两套库** —— megatron 从根上没碰那个坏 kernel，故一直没事；其训练侧 dump 583/583 全干净、普通训练 grad_norm 有限健康。

## 10. 复盘与教训（方法论）

1. **框架适配原则**：接任何框架做 dump 前，**第一步先验「不加 hcu_probe 的 baseline 普通训练健康」**（grad_norm/loss/reward 有限且趋势合理），再做 dump/compare。否则在坏训练（如 flash grad NaN）上验一致性毫无意义，还会把真问题（框架/kernel 坏）误当 dump 工具/一致性问题去查。
2. **一致性 ≠ 训练健康**：dump/compare 必开 PROMPTS_ONLY（训练退化），只验逐层 prefill 一致，**不验训练收敛**。两者要独立验，顺序是 baseline 训练健康 → dump 一致性。
3. **standalone repro 不能给 kernel 脱罪**：要保真到真实 run 的确切路径/确切函数/运行时态，靠**真实 run 内的单变量对照 + tripwire 抓现场**才靠谱。
4. **compare 要排结构性不可比项**：整层残差 / IdentityOp / 融合算子（act_fn）必须排除，否则一致率被污染、误判。

## 11. 可复现脚本

核心脚本随本 skill 分发(`scripts/` 下:run_fsdp_compare / run_fsdp_normal_eager / run_fsdp_clean / fsdp_nan_class / dispatch_logger / patch_gpu_dump / batch_compare)。下表是**完整调查脚本集**(含 repro/tripwire 等仅原始定位 NaN 用的,不随 skill 分发):

| 脚本 | 作用 |
|---|---|
| `analyze_nan.py` | 逐 module 扫 dump.json 定位第一个 NaN（排除 attention_mask 等加性 mask 误报） |
| `tripwire_sitecustomize.py` + `run_fsdp_tripwire.sh` + `replay_ffwd.py` | tripwire 抓现场入参 + 重放判 kernel-vs-运行时态 |
| `run_fsdp_{clean,sdpa}.sh` | eager / sdpa dump run（compare-ready） |
| `run_fsdp_normal{,_eager}.sh` | 普通训练（验训练健康，flash vs eager/sdpa） |
| `run_fsdp_compare.py` | FSDP compare（pt_compare backend=fsdp + 排除整层/act_fn） |

集群产物(作者环境记录,仅供参考,路径/容器名按你的环境替换):flash `verl_run_fsdp_v08`、eager `..._clean`、sdpa `..._sdpa`、普通训练 `..._normal{,_eager}`；容器 `<你的容器>`。
