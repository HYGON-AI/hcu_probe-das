---
name: hcu_probe-verl-megatron-v08-integration
version: 2.0.0
description: "在 verl-das / hcu_verl(verl 0.8.0,Megatron 训练后端,推理 das vLLM 0.18.1 + vllm-hcu)接入 hcuprobe:训推一致性 dump(训练侧 MegatronEngineWithLMHead.forward_step 钩子 + 推理侧 vllm_hcu 注入)、逐层比对(hcuprobe.integrations.verl.consistent_compare,backend=megatron)、训练三通道 monitor。关键约束:rollout TP 必须等于 actor TP、关 SP 用 ++ 且只改 actor、use_fused_kernels=False。当用户要给 verl 0.8.0 Megatron 后端接入 hcuprobe、做训练 vs 推理前向逐层比对或训练监控时使用。面向训练已跑通的人;环境见 docs_hcu/zh/quick_start/verl_quick_start.md。区别于 hcu_probe-verl-fsdp-v08-integration(FSDP 后端)与独立 Megatron-LM-das 训练(hcu_probe-megatron-integration)。"
---

# verl 0.8.0 Megatron 后端接入 hcuprobe

训练侧(Megatron)与推理侧(vllm-hcu)各采集 prompt prefill 的 L0 统计量,按 `request_id` 配对,用 `MEGATRON_MODULE_MAP`(`word_embeddings→embed_tokens`、`linear_qkv→qkv_proj`、`linear_fc1→gate_up_proj`、`linear_fc2→down_proj` 等)逐层比对。完整流程见 **`docs_hcu/zh/quick_start/verl_quick_start.md`** 第 2.6 节。

## 验证状态(2026-10,BW 8 卡)

| 项 | 环境 / 结果 |
|---|---|
| 镜像 / 代码 | 同 FSDP:光源 `verl-das-...-20260617-2235` + `HYGON-AI/verl-das@ca8f078`(verl v0.8.0) |
| 模型 | Qwen3-8B,官方 `run_qwen3_8b_megatron_vllm.sh`(actor TP4 / PP2),数据只用 gsm8k |
| 训推一致 | **7 样本 × 233 可比算子 = 1631 个,100% <1%,中位 0%,最大 0.78%**;embed 两侧 l2norm 完全相同 |
| rollout TP≠actor TP 时 | 只有 61% <1%(列并行算子切分不同,不是数值问题) |
| monitor | ✅ Qwen3-8B 3 步:8 rank × 3 通道 = 72 个 CSV(逐个权重粒度),训练正常 |

## 文件

| 文件 | 作用 | 打在哪 |
|---|---|---|
| `patches/01-verl-megatron-dump.patch` | 训练钩子 + `PROMPTS_ONLY` + `request_id`(只接 Megatron 时打这个) | `verl-das/third_party/verl` |
| `patches/01b-verl-megatron-engine-only.patch` | 只含 `megatron/transformer_impl.py` 钩子;已打 FSDP skill 的 01 时打这个 | `verl-das/third_party/verl` |
| FSDP skill `patches/05-verl-engine-monitor.patch` | monitor 钩子(`BaseEngine.train_batch`,两后端共用) | `verl-das/third_party/verl` |
| `patches/02-hcu_verl-constants.patch` | `DUMP_*` / `MONITOR_CONFIG` 透传 ray worker | `verl-das` 根目录 |
| FSDP skill `scripts/patch_hcu_dump.py` + `dispatch_logger_hcu.py` | 推理侧 vllm-hcu 注入(两后端共用) | 容器内 |
| `scripts/run_megatron_compare.py` / `run_megatron_compare_multi.py` / `batch_compare.py` / `verify_id_match.py` | 配对 + `consistent_compare(backend="megatron")` + 统计 | 容器内 |
| `config/config_actor_v08.json` / `config_generate_v08.json` | L0 statistics 模板 | 改 `dump_path` |
| `scripts/patch_gpu_dump.py`、`patches/04a/04b`、`scripts/run_megatron_bw8.sh.example` | 仅旧 das1.7 / vLLM 0.11 镜像使用 | — |

## 接入

```bash
SK=<hcu_probe-das>/.claude/skills; M=$SK/hcu_probe-verl-megatron-v08-integration; F=$SK/hcu_probe-verl-fsdp-v08-integration
cd <verl-das>/third_party/verl && git apply $M/patches/01-verl-megatron-dump.patch $F/patches/05-verl-engine-monitor.patch
cd <verl-das>                  && git apply $M/patches/02-hcu_verl-constants.patch
VH=$(python3 -c "import vllm_hcu,os;print(os.path.dirname(vllm_hcu.__file__))")
cp $F/scripts/dispatch_logger_hcu.py $VH/v1/dispatch_logger.py && python3 $F/scripts/patch_hcu_dump.py
```

训练钩子在 **`MegatronEngineWithLMHead.forward_step` 的 `forward_fn` 前后**(Megatron 由 PP 调度、无显式 micro batch 循环);只覆盖 LMHead 的非 fused 路径,所以必须 `use_fused_kernels=False`。

## 训推一致 dump run

```bash
export DUMP_ON=1 PROMPTS_ONLY=1 DUMP_PHASE=log_prob TORCHDYNAMO_DISABLE=1
export DUMP_ACTOR_CONFIG=<RUN>/cfg_actor.json DUMP_CONFIG_PATH=<RUN>/cfg_generate.json
# 在官方 run_qwen3_8b_megatron_vllm.sh 的 main_ppo 命令末尾追加:
  data.train_files=<data>/gsm8k/train.parquet data.val_files=<data>/gsm8k/test.parquet data.max_prompt_length=512 \
  data.train_batch_size=16 actor_rollout_ref.actor.ppo_mini_batch_size=16 trainer.total_training_steps=2 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.actor.megatron.param_offload=True actor_rollout_ref.actor.megatron.optimizer_offload=True \
  actor_rollout_ref.actor.megatron.grad_offload=True actor_rollout_ref.ref.megatron.param_offload=True \
  actor_rollout_ref.model.use_fused_kernels=False actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=False \
  trainer.balance_batch=False
```

| 参数 | 原因 |
|---|---|
| `rollout.tensor_model_parallel_size` = actor TP | 列并行输出(`linear_qkv`/`linear_fc1`/`q_layernorm`/`k_layernorm`/`core_attention`)两侧切分一致才能逐层比 |
| `++actor...sequence_parallel=False` | 关 SP,否则训练侧输出沿 seq 被切成 1/TP。键已存在须用 `++`;**不要单独覆盖 `ref.megatron.override_transformer_config.*`**(它通过 `oc.select` 继承 actor,单改会替换整个节点、丢 `attention_backend` → `NVTE_FUSED_ATTN` 断言失败)。只改 `actor.megatron.sequence_parallel` 不生效(mbridge 路径只读 override_transformer_config) |
| `max_num_seqs=1` + 关 chunked prefill | 推理侧每个 prefill step 只含 1 个完整 prompt |
| `log_prob_micro_batch_size_per_gpu=1` + 关 dynamic bsz | compute_log_prob 走 `infer_batch`,读的是 `log_prob_*`(不是 `ppo_micro_batch_size`) |
| offload + `gpu_memory_utilization=0.2` | 共享节点上为 vLLM 留显存;仅用于短的 dump run |

## 比对

```bash
export PYTHONPATH=<verl-das>:<verl-das>/third_party/verl:<verl-das>/third_party/Megatron-LM
python3 $M/scripts/run_megatron_compare.py <RUN>          # 单样本
python3 $M/scripts/run_megatron_compare_multi.py <RUN>    # 多样本汇总
```

`consistent_compare(train_dump, infer_dump, out, backend="megatron")`:训练侧为 npu、推理侧为 bench(反了会全部匹配不上)。脚本只挑训练/推理 embed 输入 token 数完全一致的样本(推理侧部分 prompt 会被分块)。可比集排除整层 `TransformerLayer`(含残差)和 `IdentityOp`(layernorm 融进 `TELayerNormColumnParallelLinear` 后的占位);`final_layernorm` 不在官方映射表中。

## monitor

正常训练(不设 `DUMP_ON` / `PROMPTS_ONLY`),`export MONITOR_CONFIG=<RUN>/cfg_monitor.json`(配置须 `"patch_optimizer_step": false`)。钩子与 FSDP 共用 `05-verl-engine-monitor.patch`;产物 `<output_dir>/rank_<r>/{weight_grad,optimizer,param}_step<N>-<N>.csv`。

## 坑速查

| 现象 | 处理 |
|---|---|
| 比对只有约 60% <1%,误差集中在 qkv / fc1 / q_norm / k_norm / core_attention | rollout TP 与 actor TP 不一致 |
| `Could not append to config ... sequence_parallel` | 用 `++` |
| `NVTE_FUSED_ATTN set to 1, but expected 0 for attention backend type flash` | 去掉对 `ref.megatron.override_transformer_config.*` 的覆盖 |
| 训练侧 dump 为空 | `use_fused_kernels=False`;`DUMP_PHASE=log_prob` |
| vLLM 初始化 Free memory 不足 / 权重同步 OOM | 降 `gpu_memory_utilization`、开 actor/ref offload;检查他人占卡 |

## 参考

- `references/compare-guide.md`:pt_compare 配置与字段说明
- `references/env-and-pitfalls.md`:旧镜像环境与锚点定位
