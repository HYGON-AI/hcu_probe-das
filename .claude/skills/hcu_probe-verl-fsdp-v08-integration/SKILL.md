---
name: hcu_probe-verl-fsdp-v08-integration
version: 2.0.0
description: "在 verl-das / hcu_verl(verl 0.8.0,FSDP 训练后端,推理 das vLLM 0.18.1 + vllm-hcu)接入 hcuprobe:训推一致性 dump(训练侧 FSDPEngine 钩子 + 推理侧 vllm_hcu hcu_model_runner 注入)、离线逐层比对(hcuprobe.integrations.verl.consistent_compare)、以及训练三通道 monitor(BaseEngine.train_batch 钩子)。当用户要给 verl 0.8.0 FSDP 后端接入 hcuprobe、做训练 vs 推理前向数值比对、或监控训练梯度/优化器/参数时使用。面向训练已跑通的人;环境搭建见 docs_hcu/zh/quick_start/verl_quick_start.md。区别于 hcu_probe-verl-megatron-v08-integration(Megatron 后端)与 hcu_probe-verl-fsdp-v07-integration(verl 0.7.1)。"
---

# verl 0.8.0 FSDP 后端接入 hcuprobe

**训推一致性**:RL 训练中同一模型既做推理(rollout)又做训练(actor)。本 skill 在训练侧(FSDP)与推理侧(vllm-hcu)各采集 prompt prefill 的 L0 统计量,按 vLLM `request_id` 配对后逐层比对,定界精度差在哪一侧、哪一层。另提供训练三通道 monitor。

完整可复制流程(镜像、代码 commit、模型数据、命令、预期输出):**`docs_hcu/zh/quick_start/verl_quick_start.md`**。

## 验证状态(2026-10,BW 8 卡)

| 项 | 环境 / 结果 |
|---|---|
| 镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/custom:verl-das-ubuntu22.04-dtk26.04-py3.10-20260617-2235`(torch 2.10、vllm 0.18.1 + vllm-hcu、TE 2.10) |
| 代码 | GitHub `HYGON-AI/verl-das` commit `ca8f078`(子模块 verl = v0.8.0);verl-das main 已升 0.9,本补丁打不上 |
| 基线 | Qwen2.5-0.5B-Instruct 官方脚本(默认 attention)6 步 grad_norm 有限 |
| 训推一致 | 98 个可比叶子算子 **97/98 <1%,中位 0.19%**;唯一 >1% 为 `layers.22.o_proj` 2.79%(attention kernel 差异) |
| monitor | ✅ 4 步:8 rank × 3 通道 = 96 个 CSV(FSDP 单元 `_flat_param` 粒度),训练正常 |
| 未验证 | 多机;verl 0.9;L1 / tensor 级训推比对(训练与推理算子集异构,不推荐) |

## 文件

| 文件 | 作用 | 打在哪 |
|---|---|---|
| `patches/01-verl-fsdp-dump.patch` | `FSDPEngine.forward_backward_batch` 训练侧 dump、`ray_trainer.py` 的 `PROMPTS_ONLY`、`llm_server.py` 注入 `request_id` | `verl-das/third_party/verl` |
| `patches/05-verl-engine-monitor.patch` | `BaseEngine.train_batch` monitor 钩子(FSDP / Megatron 共用) | `verl-das/third_party/verl` |
| `patches/02-hcu_verl-constants.patch` | 把 `DUMP_*`、`MONITOR_CONFIG`、`TORCHDYNAMO_DISABLE` 透传给 ray worker | `verl-das` 根目录 |
| `scripts/patch_hcu_dump.py` + `dispatch_logger_hcu.py` | 推理侧注入 `vllm_hcu/v1/hcu_model_runner.py`(锚点校验、幂等;不符报 `ANCHOR_NOT_FOUND` 不改) | 容器内 site-packages |
| `scripts/run_fsdp_compare.py` / `batch_compare.py` | 配对 + `consistent_compare(backend="fsdp")` + 可比集统计 | 容器内运行 |
| `scripts/fsdp_nan_class.py` | 区分 output[0] 真 NaN 与良性 CE 缓冲 NaN | 容器内运行 |
| `config/config_actor_v08.json` / `config_generate_v08.json` | L0 statistics 配置模板 | 改 `dump_path` 后使用 |
| `scripts/patch_gpu_dump.py`、`patches/04a/04b`、`scripts/run_fsdp_*.sh` | 仅旧 das1.7 / vLLM 0.11 / torch 2.5 镜像 | 见 `references/legacy-record.md` |

## 接入

```bash
SK=<hcu_probe-das>/.claude/skills/hcu_probe-verl-fsdp-v08-integration
cd <verl-das>/third_party/verl && git apply $SK/patches/01-verl-fsdp-dump.patch $SK/patches/05-verl-engine-monitor.patch
cd <verl-das>                  && git apply $SK/patches/02-hcu_verl-constants.patch
VH=$(python3 -c "import vllm_hcu,os;print(os.path.dirname(vllm_hcu.__file__))")
cp $SK/scripts/dispatch_logger_hcu.py $VH/v1/dispatch_logger.py && python3 $SK/scripts/patch_hcu_dump.py   # 预期 PATCHED_HCU_DUMP
```

同时接 Megatron 后端时,再打 `hcu_probe-verl-megatron-v08-integration/patches/01b-verl-megatron-engine-only.patch`(两个后端的 01 公共段相同,只能打一份)。

| # | 落点 | 门控 |
|---|---|---|
| 训练 dump | `FSDPEngine.forward_backward_batch` 的 micro_batch 循环体:懒建 `PrecisionDebugger`,每个 micro batch `start/stop/step`,写 `update_actor_log.jsonl` | `DUMP_ON=1`,`DUMP_PHASE=log_prob` |
| 只算 prefill | `RayPPOTrainer` 中裁掉 response | `PROMPTS_ONLY=1` |
| 配对 | `LLMServerClient` 把 vLLM `request_id` 写入 `extra_fields` | 始终(无副作用) |
| 推理 dump | `hcu_model_runner.execute_model` 仅在 prefill step `start/stop/step`,dump 目录按 pid 隔离,写 `dispatch_log.jsonl` | `DUMP_CONFIG_PATH` |
| monitor | `BaseEngine.train_batch`:首个训练 step 前 `on_setup_done(model, optimizer)`,`optimizer_step` 后 `on_step_end`,进程退出 `on_train_end` | `MONITOR_CONFIG` |

## 训推一致 dump run

```bash
export DUMP_ON=1 PROMPTS_ONLY=1 DUMP_PHASE=log_prob TORCHDYNAMO_DISABLE=1
export DUMP_ACTOR_CONFIG=<RUN>/cfg_actor.json DUMP_CONFIG_PATH=<RUN>/cfg_generate.json   # dump_path 分别为 <RUN>/dump_actor、<RUN>/dump_generate
# 在官方 run_qwen2_5_0.5b_fsdp_vllm.sh 的 main_ppo 命令末尾追加:
  data.train_batch_size=16 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_dynamic_bsz=False \
  trainer.balance_batch=False actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 trainer.total_training_steps=2
```

- micro batch=1 + 关 dynamic bsz + 关 balance_batch:训练侧每个 micro batch 只含 1 个 prompt,可与推理侧一对一配对。
- `enforce_eager=True`:否则 vLLM CUDA Graph 绕过推理侧 hook,dump 为空。
- `gpu_memory_utilization=0.3`:共享节点上官方 0.5 可能在 vLLM `wake_up` 时 OOM。
- `PROMPTS_ONLY=1` 会让该 run 的训练退化(grad_norm=0),属预期;训练健康看普通基线。

## 比对

```bash
export PYTHONPATH=<verl-das>:<verl-das>/third_party/verl:<verl-das>/third_party/Megatron-LM
python3 $SK/scripts/fsdp_nan_class.py <RUN>     # 预期 output[0] 真 NaN = 0
python3 $SK/scripts/run_fsdp_compare.py <RUN>
```

```python
from hcuprobe.integrations.verl import consistent_compare
consistent_compare(train_dump_json, infer_dump_json, out_dir, backend="fsdp")
```

`consistent_compare` 是 hcuprobe 自维护封装(不改与上游同步的 `core/`):调用上游 `pt_compare(consistent_check=True)`,并 ① statistics 模式使用忽略 batch=1 维的 `VerlShapeErrChecker`(训练 `[1,S,H]` vs 推理 `[S,H]`);② 比对前剔除 `Triton.*` kernel 条目(vllm-hcu 的 Triton kernel 会与训练侧误配,报 "Conflicting comparison data")。结果为 `<RUN>/compare_out/compare_result_*.csv`(列名 `NPU Name`=训练侧,`Bench Name`=推理侧)。

**可比集口径**(脚本已内置):只统计非 wrapper 的叶子算子 output;排除整层 `Qwen2DecoderLayer` / `Qwen2MLP` / `Qwen2Attention`(含残差)、`IdentityOp`、`act_fn`(HF `silu(gate)` vs vLLM `SiluAndMul`)。HF 的 `q/k/v_proj`、`gate/up_proj` 分开,vLLM 融合,不可比。剩余 `o_proj` / `down_proj` / `input_layernorm` / `post_attention_layernorm` / `model.norm` / `embed` 共 98 个(Qwen2.5-0.5B)。

## monitor

```bash
export MONITOR_CONFIG=<RUN>/cfg_monitor.json   # 模板 hcu_probe-megatron-integration/config/monitor_v2_config_full_probe.json,改 output_dir
# 正常训练(不设 DUMP_ON / PROMPTS_ONLY)
```

产物 `<output_dir>/rank_<r>/{weight_grad,optimizer,param}_step<N>-<N>.csv`,列 `vpp_stage,step,module_name,scope,micro_step,min,max,mean,norm,nans`。FSDP1 下 `module_name` 为每个 FSDP 单元的 `_flat_param`(按层聚合,非逐个权重)。配置必须 `"patch_optimizer_step": false`,不开 `module` 通道。判断是否生效以 CSV 为准(ray worker 日志常不回传 driver)。

## 坑速查

| 现象 | 处理 |
|---|---|
| 推理侧 `ANCHOR_NOT_FOUND` | 用 `patch_hcu_dump.py`(vllm-hcu);`patch_gpu_dump.py` 只适用旧 vLLM 0.11 镜像 |
| 推理 dump 为空 | `actor_rollout_ref.rollout.enforce_eager=True` |
| vLLM `wake_up` OOM / Free memory 不足 | 降 `rollout.gpu_memory_utilization`;`hy-smi --showpids` 查他人占卡 |
| 比对几乎全 shape error 或 "Conflicting comparison data" | 直接调了上游 `pt_compare`,改用 `consistent_compare` |
| `grep NaN` 全部文件命中 | 良性 `Triton.cross_entropy_fwd_kernel` 缓冲,用 `fsdp_nan_class.py` 看 output[0] |
| monitor 无 CSV | 确认打了 05、`MONITOR_CONFIG` 经 02 透传;`output_dir` 父目录可写 |
| 旧 das1.7 镜像 grad_norm 每步 NaN | 该镜像需 `+actor_rollout_ref.model.override_config.attn_implementation=eager/sdpa`(新镜像不需要),见 `references/flash-nan-and-sdpa.md` |

## 参考

- `references/legacy-record.md`:重构前正文(das1.7 / vLLM 0.11 时代的完整记录与排查过程)
- `references/flash-nan-and-sdpa.md`:旧镜像 flash attention NaN 定位
- `references/hcu_probe-fsdp-statistics-fix.md`:statistics 模式 shape 容差问题原理
