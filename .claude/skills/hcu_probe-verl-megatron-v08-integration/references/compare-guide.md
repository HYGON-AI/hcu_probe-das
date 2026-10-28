# compare 操作指南:hcu_probe 官方 pt_compare(verl 0.8.0 Megatron 后端)

> 本文是 SKILL.md §二B/§三 的详细展开。verl 0.8.0 起 hcu_probe **自带** verl 训推一致 compare,**不再用自研脚本**(FSDP skill 当时用 `verify_consistency.py` 是因为那时 hcu_probe 还没集成)。

## 一、官方 pt_compare 用法

```python
from hcuprobe.pytorch.compare.pt_compare import pt_compare
pt_compare(
    input_param={"npu_path": <训练 dump.json>, "bench_path": <推理 dump.json>},
    output_path=<输出目录>,
    consistent_check=True,
    backend="megatron",
)
```
也可走 CLI:`hcuprobe compare -tp <训练> -gp <推理> --consistent_check --backend megatron -o <out>`(`-tp`=npu=训练,`-gp`=bench=推理)。命令名是 **`hcuprobe`**(我们 whl 注册的 console_script,见 `pyproject.toml [project.scripts]`,入口 `hcuprobe.cli:main`),**不是 `msprobe`**——容器装的是 hcu_probe fork,无 msprobe 命令。

### 关键点
- **npu=训练 megatron / bench=推理 vllm**,方向**不能反**。证据:`hcu_probe/core/compare/acc_compare.py:786-787`——`process_module_mapping(npu_df, engine='train', ...)` / `(bench_df, engine='infer', ...)`(:784-785 同处 `get_op_module_and_class` 也成对 train/infer)。反了 module 全 match 不上(报 shape/grad inconsistent)。
- **`MEGATRON_MODULE_MAP` 自动逐层映射**(`hcu_probe/core/compare/verl/mapping.py`),无需手写 cell_mapping:
  ```
  word_embeddings→embed_tokens   self_attention→self_attn   linear_qkv→qkv_proj
  q_layernorm→q_norm   k_layernorm→k_norm   core_attention→attn   linear_proj→o_proj
  router→gate   linear_fc1→gate_up_proj   linear_fc2→down_proj
  ```
- **`need_cat` + `BACKEND_CAT_RULES`** 自动按 seq 维 **concat** 拼接 SP 切片(不是平方和聚合——SP 是把不同 token 分到不同 rank,平方和数学上错)。
- **容器要 `pip install openpyxl`**(否则 compare 算完在存 xlsx 时崩 `No module named 'openpyxl'`)。
- 输出 xlsx 列:`NPU Name / Bench Name / NPU l2norm / Bench l2norm / NormRelativeErr / MaxRelativeErr / MeanRelativeErr / Result`。
- 现成封装 `scripts/run_megatron_compare.py`(单样本)、`scripts/batch_compare.py`(多样本统计通过率)。

## 二、compare 正确配置(让两侧 dump 对齐)

这是本任务最大教训——**反复在错误字段上试错**,经逐一核读 verl 源码才定位(下面每条都带 文件:行 证据,可自行复核)。

### 2.1 `num_requests=1`(训练单 prompt micro_batch)
```
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.use_dynamic_bsz=False
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False   # 须与 actor 一致,否则 engine_workers.py:557 assert 崩
```
- **真因**:`compute_log_prob`(dump 发生处,forward_only)走 `infer_batch`(`engine_workers.py:644`),读的是 `infer_micro_batch_size_per_gpu`(init 时赋值自 `rollout.log_prob_micro_batch_size_per_gpu`,`:564-566`),**不是** `actor.ppo_micro_batch_size_per_gpu`(那个只管 update_actor)。
- 且 `use_dynamic_bsz=True` 时,`prepare_micro_batches`(`engine/utils.py:74`)走 token 打包分支,把多 prompt 塞进一个 micro_batch(达 `max_token_len≈16384`),**完全忽略 micro_batch_size**。故必须 `use_dynamic_bsz=False`。
- 单批前提:`data.train_batch_size / ppo_mini_batch_size = 1`(mini_batch_num=1)。

### 2.2 关 sequence parallel(否则 output 沿 seq 切 1/TP)
```
++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=False
# (2026-10 修正:只改 actor;ref 是 oc.select 插值继承 actor,单改 ref 会丢 attention_backend → NVTE_FUSED_ATTN 断言)
```
- **真因**:`actor.megatron.sequence_parallel=False` **不生效**——默认 `vanilla_mbridge=True` 路径(`transformer_impl.py:192-199`)只读 `override_transformer_config`,`engine_config.sequence_parallel` 是死字段;SP 由 mbridge 按 TP>1 自动开。只有 `override_transformer_config.sequence_parallel` 经 `set_extra_args` 真正写进 TransformerConfig(两条路通吃)。
- 不关时 SP 沿 seq 切:`pipeline_parallel.py:34-44` 形状 `[total_token//TP, 1, h]`,如 `[43,1,4096]`=344/8。关后变 `[344,1,4096]`,配合 `use_remove_padding=True`(THD packed)与 vllm `[total_token, h]` 同构。

### 2.3 其余
```
actor_rollout_ref.model.use_remove_padding=True   # THD packed,去 pad,与推理 prefill shape 对齐
trainer.balance_batch=False                        # 不跨 rank 重排,保训推每卡 request 逐一配对
PROMPTS_ONLY=1                                      # 训练裁 response 只算 prompt prefill
actor_rollout_ref.rollout.enforce_eager=True       # 否则 vllm compile/cudagraph 绕过推理 hook,dump 为空
```

## 三、样本配对

1. **挑 prefill 单 request 样本**:推理 `dispatch_log.jsonl` 选 `phase=prefill` 且 `requests` 数=1 的行,拿 `request_id`。
2. **关联训练**:训练 `update_actor_log.jsonl` 搜同 `request_id`(strip das 加的 `-<8hex>` 后缀),拿 `step`/`rank`,定位 `dump_actor/step{S}/rank{R}/dump.json`。
3. **⚠️ 必须挑 token 数完全匹配的配对**:推理部分 prompt 被 chunk(dump 粒度),`dump_generate/{pid}/step{k}` 可能取到不完整 chunk(如完整 72 tok 取成 4 tok)。`run_megatron_compare.py` / `verify_id_match.py` 已含「train/gen embed input token 数相等」过滤——**不过滤会拿错配对、误差全是假象**(实测过滤前误差 1~130%,过滤后真算子 99.4% <1%)。

## 四、诚实定界(哪些不参与比对)

- **整层 `TransformerLayer` output**:层封装含 SP/residual,非 normed 算子(误差 7000%+,与 FSDP 不比 `DecoderLayer` 同理)。
- **`IdentityOp`**:megatron 把 layernorm 融进 `TELayerNormColumnParallelLinear`,层内 `input_layernorm`/`pre_mlp_layernorm` 是恒等占位,无真 norm 输出,不对应 vllm 的真 `RMSNorm`。
- **`final_layernorm`**:官方 `MEGATRON_MODULE_MAP` 未覆盖(没 `final_layernorm→model.norm`),官方 compare 不比;但可手动验(probe 两侧 Norm `1320==1320` 一致)。
- **真正参与比对的**:embed + `linear_qkv`/`linear_proj`/`linear_fc1`/`linear_fc2`/`core_attention` 等两侧都存在的真算子。结论:1083 个真算子 99.4% 误差<1%、中位 0%。

## 五、FSDP 后端也可用官方 pt_compare(回应"FSDP 是否该用 pt_compare")

hcu_probe 同样支持 `--backend fsdp`:
- `mapping.py` 有 `FSDP_MODULE_MAP`(`q_proj/k_proj/v_proj→qkv_proj`,`gate_proj/up_proj→gate_up_proj`)。
- `acc_compare.py:611` `if engine == 'infer' and backend == Const.FSDP`——**FSDP 的映射应用在 infer(推理)侧**,与 megatron 在 train 侧相反,故 FSDP 用时 npu/bench 方向要另核对。
- **建议**:FSDP 后端此前用自研 `verify_consistency.py`(已验证 80/80 训推一致,可靠),回头用官方 `pt_compare --backend fsdp` 复核会更权威、省自研维护。
- ⚠️ **本任务未实测 FSDP 走官方 pt_compare**(只在 megatron 后端实测通过),上面 FSDP 部分仅作方向参考、未坐实,作后续验证项,勿当已验证结论照搬。
