# hcuprobe verl 训推一致性快速入门

<br>

## 1. 概述

强化学习（RL）训练中，同一个模型既做推理（rollout 生成）又做训练（actor 前向/反向）。**训推一致性**要回答的是：同一个 prompt，训练侧前向与推理侧前向的数值是否在浮点误差内一致。不一致会表现为 `ppo_kl` 偏大、reward 不涨、训练发散等难以定位的问题。

本文以 **verl-das（verl 0.8.0）+ das vLLM 0.18.1（vllm-hcu）** 在 HCU 上做 GRPO 训练为例，演示用 hcuprobe 分别在训练侧与推理侧采集 prompt prefill 阶段的逐层统计量，按 `request_id` 配对后用官方 `pt_compare` 做逐层比对的完整流程，覆盖 **FSDP** 与 **Megatron** 两种训练后端。

**体验地图**

| 步骤 | 环节 | 核心工具 | 说明 |
| :---: | :--- | :--- | :--- |
| 1 | 环境准备 | verl-das 官方镜像 + hcuprobe whl | 拉镜像、起容器、拉代码、装 whl |
| 2 | 模型与数据 | modelscope / verl 数据预处理 | Qwen2.5-0.5B-Instruct（FSDP）、Qwen3-8B（Megatron）、gsm8k |
| 3 | 基线训练（可选） | verl-das 官方脚本 | 先确认不带 hcuprobe 的训练能跑通 |
| 4 | 接入 hcuprobe | 训练侧 git patch + 推理侧注入脚本 | 由环境变量门控，不开 dump 时行为不变 |
| 5 | 训推 dump | PrecisionDebugger（L0 statistics） | 训练侧 `dump_actor/`、推理侧 `dump_generate/` |
| 6 | 训推比对 | `hcuprobe.integrations.verl.consistent_compare`（封装上游 `pt_compare`） | FSDP：`backend="fsdp"`；Megatron：`backend="megatron"` |
| 7 | 训练监控（可选） | `MONITOR_CONFIG`（TrainerMonitorV2） | 权重梯度 / 优化器状态 / 参数三通道 CSV |

> 👉 本教程的全部命令、镜像与代码版本均在海光 BW 8 卡节点上实测通过（2026-10）。若只关心某一个后端，完成第 2.1~2.4 节后可直接跳到 2.5（FSDP）或 2.6（Megatron）。

### 1.1 版本清单（复现必须一致）

| 组件 | 版本 / 地址 |
| --- | --- |
| 容器镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/custom:verl-das-ubuntu22.04-dtk26.04-py3.10-20260617-2235`（digest `sha256:07a5e0375010…`） |
| 镜像内栈 | torch `2.10.0+das.opt1.dtk2604`、vllm `0.18.1+das`、vllm-hcu `0.18.1+das`（`g1dcfc6`）、TransformerEngine `2.10.0+das`、flash-attn `2.8.3+das`、ray `2.56.1`、mbridge `0.15.1` |
| verl-das | <https://github.com/HYGON-AI/verl-das> commit `ca8f078`（子模块 verl = **v0.8.0** `7aed6b2`，Megatron-LM `266f1c9`） |
| hcuprobe | <https://github.com/HYGON-AI/hcu_probe-das> 分支 `26.2.0-dev`（发行名 `hcu-probe`，import 名 `hcuprobe`，命令行 `hcuprobe`） |
| 接入补丁 | hcu_probe-das 仓 `.claude/skills/hcu_probe-verl-fsdp-v08-integration/`、`.claude/skills/hcu_probe-verl-megatron-v08-integration/` |

> [!NOTE]说明
>
> verl-das 主干已升级到 verl 0.9。本教程的训练侧补丁锚定 **verl 0.8.0（verl-das `ca8f078`）**，请务必 checkout 该 commit；在 verl 0.9 上 `git apply` 会失败。

## 2. 操作步骤

### 2.1 环境准备（必做）

#### 2.1.1 前置条件

| 项目 | 要求 | 验证方法 |
| --- | --- | --- |
| 硬件 | 1 台 8 卡 HCU（BW 系列，单卡 64 GB），驱动正常 | `hy-smi` 能列出 8 张卡 |
| 容器 | 已安装 Docker，可访问 `harbor.sourcefind.cn:5443` | `docker ps` 无报错 |
| 网络 | 容器内可访问 GitHub、PyPI 镜像、ModelScope | 无外网时参考第 3 章 |
| 磁盘 | 工作目录可用空间 ≥ 60 GB（模型约 17 GB + dump 数百 MB） | `df -h` |

#### 2.1.2 宿主机：准备工作目录与代码

```bash
export WORK=/path/to/hcuprobe_verify          # 修改为你的工作目录(需能挂载进容器)
mkdir -p $WORK/{src,models,data,runs,scripts}
cd $WORK/src
git clone -b 26.2.0-dev https://github.com/HYGON-AI/hcu_probe-das.git
git clone https://github.com/HYGON-AI/verl-das.git
cd verl-das && git checkout ca8f078 && git submodule update --init third_party/verl third_party/Megatron-LM
git -C third_party/verl describe --tags     # 预期输出 v0.8.0
```

#### 2.1.3 宿主机：拉取镜像并启动容器

```bash
export IMG=harbor.sourcefind.cn:5443/hcu/admin/base/custom:verl-das-ubuntu22.04-dtk26.04-py3.10-20260617-2235
docker pull $IMG
docker run -itd --name hcuprobe_verl --shm-size=64G \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  --ipc=host --network=host --group-add video --privileged \
  -v /opt/hyhal:/opt/hyhal:ro -v $WORK:/workspace -w /workspace \
  $IMG /bin/bash
docker exec -it hcuprobe_verl bash
```

以下命令如无特别说明，均在**容器内**执行。

#### 2.1.4 容器内：安装依赖与 hcuprobe

```bash
cd /workspace
git config --global --add safe.directory '*'
# 1) verl-das 官方依赖。镜像已带 mbridge 0.15.1,去掉 requirements 里浮动的 mbridge@main 以保证可复现
grep -v "mbridge.git" src/verl-das/requirements.txt > /tmp/req.txt
pip install -r /tmp/req.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 源码构建 hcuprobe whl(注意 whl 文件名是 hcu_probe-*.whl)
cd /workspace/src/hcu_probe-das
pip wheel . --no-deps -w dist/
pip install dist/hcu_probe-*.whl --no-deps
# hcuprobe 运行依赖:用 constraints 钉住镜像自带的 torch/vllm/numpy,避免被 pyproject 中 numpy<2.0 降级
pip freeze | grep -iE "^(numpy|torch|vllm|vllm-hcu|transformers|flash.attn|transformer.engine|ray)==" > /tmp/constraints.txt
pip install -c /tmp/constraints.txt einops pandas pyyaml tqdm openpyxl matplotlib tensorboard skl2onnx \
  packaging pytz psutil h5py modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 2.1.5 容器内：检查环境

```bash
python3 -c "import torch, vllm, vllm_hcu, hcuprobe; from hcuprobe.pytorch import PrecisionDebugger; \
from hcuprobe.integrations.verl import consistent_compare; \
assert torch.cuda.device_count()==8; print('torch', torch.__version__, 'vllm', vllm.__version__)" \
&& hcuprobe --help >/dev/null && echo -e "\e[32m[PASS] HCU / vllm-hcu / hcuprobe 环境检查通过\e[0m"
```

若 `consistent_compare` 导入失败，说明 hcu_probe-das 版本过旧（缺少 `hcuprobe.integrations.verl`），请更新后重新构建 whl。

### 2.2 准备模型与数据

```bash
# 模型
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir /workspace/models/Qwen2.5-0.5B-Instruct   # FSDP 用
modelscope download --model Qwen/Qwen3-8B             --local_dir /workspace/models/Qwen3-8B              # Megatron 用

# gsm8k:目录需保持 HuggingFace openai/gsm8k 的原始结构(README.md + main/*.parquet)
#   /workspace/data/gsm8k/README.md
#   /workspace/data/gsm8k/main/{train,test}-00000-of-00001.parquet
cd /workspace/src/verl-das/third_party/verl
PYTHONPATH=$PWD python3 examples/data_preprocess/gsm8k.py \
  --local_dataset_path /workspace/data/gsm8k --local_save_dir /workspace/data/gsm8k
ls /workspace/data/gsm8k/{train,test}.parquet
```

> [!NOTE]说明
>
> - `README.md` 不能缺：它声明了 `main` 配置，缺失时 `load_dataset` 报 `BuilderConfig 'main' not found`。
> - 必须设置 `PYTHONPATH`，否则预处理脚本报 `No module named 'verl'`。

### 2.3 基线训练（可选，推荐）

先用 verl-das 官方脚本确认训练能跑通。官方 `examples/grpo_trainer/run.sh` 面向多机（mpirun + sshd），单机时只需做它的等价步骤：

```bash
cd /workspace/src/verl-das/examples/grpo_trainer
export VERL_PATH=/workspace/src/verl-das
export PYTHONPATH=$VERL_PATH:$VERL_PATH/third_party/verl:$VERL_PATH/third_party/Megatron-LM:$PYTHONPATH
export VLLM_CUDART_SO_PATH=/opt/dtk/hip/lib/libgalaxyhip.so PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error
cp $VERL_PATH/hcu_verl/patch_init.py $VERL_PATH/third_party/verl/verl/__init__.py   # run.sh 中的 hcu verl patch
echo "localhost slots=8" > /workspace/runs/hostfile
```

为快速验证，在官方脚本的 main_ppo 命令末尾追加小规模参数（hydra 后者覆盖前者），以 FSDP 为例：

```bash
cp run_qwen2_5_0.5b_fsdp_vllm.sh /workspace/runs/run_fsdp.sh
export SMALL="data.train_batch_size=16 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 data.max_response_length=256 \
  trainer.balance_batch=False trainer.total_training_steps=6 trainer.test_freq=-1 trainer.save_freq=-1"
sed -i 's#^\(\s*\)\${TRAINER_CONFIG\[@\]} \\$#\1${TRAINER_CONFIG[@]} ${SMALL} ${EXTRA} \\#' /workspace/runs/run_fsdp.sh
bash /workspace/runs/run_fsdp.sh --data_path=/workspace/data --host_file=/workspace/runs/hostfile \
  --hf_model_path=/workspace/models --mcore_model_path=/workspace/models --save_ckpt_path=/workspace/runs/fsdp_base --profiling=
```

**预期结果**：6 个 step 全部完成，日志中 `actor/grad_norm` 为有限值（实测 0.0002~0.58），无 NaN。Megatron 后端同理，复制 `run_qwen3_8b_megatron_vllm.sh`，并额外追加 `data.train_files=/workspace/data/gsm8k/train.parquet data.val_files=/workspace/data/gsm8k/test.parquet data.max_prompt_length=512`（官方脚本还会读取 math 数据集）。

### 2.4 接入 hcuprobe（训练侧 + 推理侧）

接入共 4 处改动，全部由环境变量门控：不设置 `DUMP_ON` / `DUMP_CONFIG_PATH` 时与原始行为一致。

| # | 落点 | 作用 |
| --- | --- | --- |
| ① | `verl/workers/engine/fsdp/transformer_impl.py`（`FSDPEngine.forward_backward_batch`）<br>`verl/workers/engine/megatron/transformer_impl.py`（`MegatronEngineWithLMHead.forward_step`） | 训练侧前向前后 `start/stop/step`，并记录 `request_id` |
| ② | `verl/trainer/ppo/ray_trainer.py` | `PROMPTS_ONLY=1` 时裁掉 response，训练前向只算 prompt prefill（对齐推理 prefill） |
| ③ | `verl/workers/rollout/llm_server.py` | 把 vLLM `request_id` 注入 `extra_fields`，贯穿到训练侧 micro batch，用于配对 |
| ④ | `hcu_verl/trainer/constants_ppo.py` | 把 `DUMP_*` 环境变量透传给 ray worker |
| 推理 | `vllm_hcu/v1/hcu_model_runner.py` | 推理侧仅在 prefill step 做 dump，并写 `dispatch_log.jsonl` |

```bash
SK=/workspace/src/hcu_probe-das/.claude/skills
V=/workspace/src/verl-das
# 训练侧(01 含 ②③ 与 FSDP 的 ①;01b 为 Megatron 的 ①,只用 Megatron 时也需先打 01)
cd $V/third_party/verl
git apply $SK/hcu_probe-verl-fsdp-v08-integration/patches/01-verl-fsdp-dump.patch
git apply $SK/hcu_probe-verl-megatron-v08-integration/patches/01b-verl-megatron-engine-only.patch
git apply $SK/hcu_probe-verl-fsdp-v08-integration/patches/05-verl-engine-monitor.patch          # 训练监控(2.7 节),不用可不打
cd $V && git apply $SK/hcu_probe-verl-fsdp-v08-integration/patches/02-hcu_verl-constants.patch
# 推理侧:vllm-hcu 0.18.1 的 hcu_model_runner 注入(带锚点校验与幂等标记)
VH=$(python3 -c "import vllm_hcu,os;print(os.path.dirname(vllm_hcu.__file__))")
cp $SK/hcu_probe-verl-fsdp-v08-integration/scripts/dispatch_logger_hcu.py $VH/v1/dispatch_logger.py
python3 $SK/hcu_probe-verl-fsdp-v08-integration/scripts/patch_hcu_dump.py     # 预期输出 PATCHED_HCU_DUMP
```

> [!NOTE]说明
>
> - 训练侧 `.patch` 锚定 verl 0.8.0 源码，与镜像无关；推理侧注入脚本锚定镜像中的 vllm-hcu 版本，锚点不匹配时报 `ANCHOR_NOT_FOUND` 并退出，不会误改。
> - skill 中的 `patch_gpu_dump.py` 面向旧的 das vLLM 0.11（无 vllm-hcu 插件）镜像，在本镜像上会报 `ANCHOR_NOT_FOUND: __init__`，请使用 `patch_hcu_dump.py`。
> - `04a/04b` 补丁仅用于 torch 2.5.1 的旧镜像，本镜像（torch 2.10）无需打。

### 2.5 FSDP 后端：采集与比对

#### 2.5.1 准备采集配置

```bash
export RUN=/workspace/runs/fsdp_dump && mkdir -p $RUN
for side in actor generate; do cat > $RUN/cfg_$side.json << EOF
{"task": "statistics", "dump_path": "$RUN/dump_$side", "rank": [], "step": [], "level": "L0",
 "async_dump": false, "statistics": {"scope": [], "list": [], "data_mode": ["all"], "summary_mode": "statistics"}}
EOF
done
```

`task=statistics` 只保存统计量（max/min/mean/l2norm），`level=L0` 采集 Module 级前向输入输出。

#### 2.5.2 启动采集

在 2.3 节环境变量基础上增加 dump 开关，并追加单 prompt 对齐参数：

```bash
export DUMP_ON=1 PROMPTS_ONLY=1 DUMP_PHASE=log_prob TORCHDYNAMO_DISABLE=1
export DUMP_ACTOR_CONFIG=$RUN/cfg_actor.json DUMP_CONFIG_PATH=$RUN/cfg_generate.json
export EXTRA="trainer.total_training_steps=2 actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.rollout.gpu_memory_utilization=0.3"
cd /workspace/src/verl-das/examples/grpo_trainer
bash /workspace/runs/run_fsdp.sh --data_path=/workspace/data --host_file=/workspace/runs/hostfile \
  --hf_model_path=/workspace/models --mcore_model_path=/workspace/models --save_ckpt_path=$RUN --profiling=
```

| 参数 | 作用 |
| --- | --- |
| `PROMPTS_ONLY=1` + `DUMP_PHASE=log_prob` | 只 dump `compute_log_prob` 的 prompt 前向，与推理 prefill 对齐 |
| `*log_prob_micro_batch_size_per_gpu=1` + `use_dynamic_bsz=False` + `balance_batch=False` | 训练侧每个 micro batch 只含 1 个 prompt，便于按 `request_id` 一对一配对 |
| `enforce_eager=True` | 关闭 vLLM CUDA Graph，否则推理侧 hook 被绕过、dump 为空 |
| `gpu_memory_utilization=0.3` | 共享节点上为 vLLM 唤醒留出余量（官方 0.5 在他人占卡时会 OOM） |

**预期结果**：日志出现 `[hcu_probe-infer] debugger init pid=...`（每个 vLLM worker 一条），运行结束后：

```text
$RUN/dump_actor/stepN/rankM/dump.json  +  $RUN/dump_actor/<pid>/update_actor_log.jsonl     # 训练侧
$RUN/dump_generate/<pid>/stepN/rankM/dump.json  +  .../dispatch_log.jsonl                 # 推理侧
```

`PROMPTS_ONLY=1` 会让训练退化（`grad_norm=0`），这是预期行为：该 run 只用于一致性比对，训练健康请看 2.3 节基线。

#### 2.5.3 检查 NaN 并比对

```bash
SKS=/workspace/src/hcu_probe-das/.claude/skills/hcu_probe-verl-fsdp-v08-integration/scripts
V=/workspace/src/verl-das
python3 $SKS/fsdp_nan_class.py $RUN
PYTHONPATH=$V:$V/third_party/verl:$V/third_party/Megatron-LM python3 $SKS/run_fsdp_compare.py $RUN
```

`run_fsdp_compare.py` 按 `request_id` 配对两侧 dump，挑选 embed 输入 token 数完全一致的样本，调用：

```python
from hcuprobe.integrations.verl import consistent_compare
consistent_compare(训练侧dump.json, 推理侧dump.json, out_dir, backend="fsdp")
```

`consistent_compare` 是 hcuprobe 在 `integrations/` 下自维护的封装，内部调用上游原样的 `pt_compare(consistent_check=True)`，并补两处 HCU 场景必需的处理（不修改与上游同步的 `core/`）：

- statistics 模式使用可忽略 batch=1 维的 shape 检查（训练侧 `[1,S,H]` 对推理侧 `[S,H]`）；
- 比对前剔除 `Triton.*` kernel 级条目（vllm-hcu 推理侧的 Triton kernel 会与训练侧 Triton 条目误配）。

**预期结果（实测）**：

```text
NaN 检查: output[0] 含 NaN(真信号): 0;仅 Triton.cross_entropy_fwd_kernel 的未初始化缓冲含 NaN(良性)
总配对: 24  token数完全匹配: 13
可比叶子算子 98 个:<1%: 97/98,中位 0.19%
  o_proj 24 / down_proj 24 / input_layernorm 24 / post_attention_layernorm 24 / model.norm 1 / embed 1
  唯一 >1%:layers.22.self_attn.o_proj 2.79%(训练与推理的 attention kernel 不同)
```

结果文件为 `$RUN/compare_out/compare_result_<时间戳>.csv`。

> [!NOTE]说明：可比集为什么只有 98 个
>
> - HF 训练侧 `q/k/v_proj`、`gate/up_proj` 是分开的，vLLM 是融合的 `qkv_proj` / `gate_up_proj`，形状和语义不同，不进入可比集。
> - `act_fn`：HF 是 `silu(gate)`，vLLM 是 `SiluAndMul = silu(gate)·up`，不是同一个张量。
> - 整层 `Qwen2DecoderLayer` / `Qwen2MLP` / `Qwen2Attention` 的输出含残差，两侧捕获点不同，不比较。

### 2.6 Megatron 后端：采集与比对

#### 2.6.1 启动采集

```bash
export RUN=/workspace/runs/megatron_dump && mkdir -p $RUN    # 同 2.5.1 生成 cfg_actor.json / cfg_generate.json
cp /workspace/src/verl-das/examples/grpo_trainer/run_qwen3_8b_megatron_vllm.sh /workspace/runs/run_megatron.sh
sed -i 's#^\(\s*\)\${TRAINER_CONFIG\[@\]} \\$#\1${TRAINER_CONFIG[@]} ${SMALL} ${EXTRA} \\#' /workspace/runs/run_megatron.sh
export DUMP_ON=1 PROMPTS_ONLY=1 DUMP_PHASE=log_prob TORCHDYNAMO_DISABLE=1
export DUMP_ACTOR_CONFIG=$RUN/cfg_actor.json DUMP_CONFIG_PATH=$RUN/cfg_generate.json
export EXTRA="trainer.total_training_steps=2 \
  data.train_files=/workspace/data/gsm8k/train.parquet data.val_files=/workspace/data/gsm8k/test.parquet data.max_prompt_length=512 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=False"
cd /workspace/src/verl-das/examples/grpo_trainer
bash /workspace/runs/run_megatron.sh --data_path=/workspace/data --host_file=/workspace/runs/hostfile \
  --hf_model_path=/workspace/models --mcore_model_path=/workspace/models --save_ckpt_path=$RUN --profiling=
```

| 参数 | 作用 |
| --- | --- |
| `rollout.tensor_model_parallel_size=4` | **推理 TP 必须等于训练 TP**（官方脚本 actor TP=4、rollout TP=2）。不对齐时列并行算子（`linear_qkv`/`linear_fc1`/`q_norm`/`k_norm`/`core_attention`）两侧切分不同，无法逐层比对 |
| `use_fused_kernels=False` | 否则前向走融合分支，绕过训练侧钩子 |
| `++…override_transformer_config.sequence_parallel=False` | 关闭 SP，否则训练侧输出沿序列维被切成 1/TP。用 `++`（该键已存在）；**只改 actor**，ref 通过插值自动继承 |
| `max_num_seqs=1` + `enable_chunked_prefill=False` | 推理侧每个 prefill step 只有 1 个完整 prompt |

#### 2.6.2 比对

```bash
SKM=/workspace/src/hcu_probe-das/.claude/skills/hcu_probe-verl-megatron-v08-integration/scripts
V=/workspace/src/verl-das
export PYTHONPATH=$V:$V/third_party/verl:$V/third_party/Megatron-LM
python3 $SKM/run_megatron_compare.py $RUN          # 单样本:consistent_compare(..., backend="megatron")
python3 $SKM/run_megatron_compare_multi.py $RUN    # 多样本汇总
```

**预期结果（实测，Qwen3-8B，actor TP4/PP2 + rollout TP4）**：

```text
总配对: 32  token数完全匹配: 7
sample 0 token=68 可比算子=233 <1%=233/233 中位=0% max=0.6993%
...
sample 6 token=88 可比算子=233 <1%=233/233 中位=0% max=0.7692%
总体: 所有样本可比算子合计 1631  其中 <1% 合计 1631
embed(word_embeddings ↔ embed_tokens)两侧 l2norm 完全相同
```

各类算子（`linear_qkv`、`q_layernorm`、`k_layernorm`、`core_attention`、`linear_proj`、`linear_fc1`、`linear_fc2`、`mlp`）误差均 <1%，多数完全相同。`MEGATRON_MODULE_MAP` 自动完成 megatron ↔ vLLM 命名映射（`linear_qkv→qkv_proj`、`linear_fc1→gate_up_proj` 等）。整层 `TransformerLayer` 的输出含残差，不参与统计；`IdentityOp`（layernorm 融合进 `TELayerNormColumnParallelLinear` 后的占位）也排除在外。

> [!NOTE]说明
>
> 若推理 TP 与训练 TP 不一致（官方默认 rollout TP=2），实测只有约 61% 的算子 <1%：误差集中在 `linear_qkv`、`linear_fc1`、`q/k_layernorm`、`core_attention` 这类列并行输出上，原因是两侧张量切分不同，并非数值问题。

### 2.7 训练三通道监控（Monitor）

Monitor 在每个训练 step 的 `optimizer.step()` 之后记录权重梯度、优化器状态（Adam m/v）、参数的 min/max/mean/norm/nans，用于发现 NaN、梯度爆炸、参数异常。钩子在 `BaseEngine.train_batch`（`05-verl-engine-monitor.patch`），FSDP 与 Megatron 后端共用，由 `MONITOR_CONFIG` 环境变量开启（`02` 补丁负责透传给 ray worker）。

```bash
mkdir -p /workspace/runs/fsdp_monitor
python3 -c "
import json
c = json.load(open('/workspace/src/hcu_probe-das/.claude/skills/hcu_probe-megatron-integration/config/monitor_v2_config_full_probe.json'))
c['output_dir'] = '/workspace/runs/fsdp_monitor/monitor'
json.dump(c, open('/workspace/runs/fsdp_monitor/cfg_monitor.json', 'w'), indent=2)"
unset DUMP_ON PROMPTS_ONLY DUMP_ACTOR_CONFIG DUMP_CONFIG_PATH      # Monitor 用正常训练,不裁 response
export MONITOR_CONFIG=/workspace/runs/fsdp_monitor/cfg_monitor.json
export EXTRA="trainer.total_training_steps=4 actor_rollout_ref.rollout.gpu_memory_utilization=0.3"
bash /workspace/runs/run_fsdp.sh --data_path=/workspace/data --host_file=/workspace/runs/hostfile \
  --hf_model_path=/workspace/models --mcore_model_path=/workspace/models --save_ckpt_path=/workspace/runs/fsdp_monitor --profiling=
```

配置要点：`"patch_optimizer_step": false`（由钩子每步手动驱动）、只开 `weight_grad` / `optimizer` / `param` 三个通道。Megatron 后端用 `run_megatron.sh`，同样只需设置 `MONITOR_CONFIG`。

**预期结果（实测）**：

- 日志出现 `[hcu_probe.monitor] TrainerMonitorV2 initialized and started`（在 `WorkerDict` 进程中）。
- `monitor/rank_<0..7>/{weight_grad,optimizer,param}_step<N>-<N>.csv`：FSDP 4 步共 96 个文件，Megatron（Qwen3-8B）3 步共 72 个文件；训练 grad_norm 正常。

```text
# FSDP:按 FSDP 单元聚合(_flat_param,每层一行)
vpp_stage,step,module_name,scope,micro_step,min,max,mean,norm,nans
-1,3,_fsdp_wrapped_module.model.layers.0._fsdp_wrapped_module._flat_param,reduced,1,-0.0014,0.0014,4.83e-09,0.0248,0.0
# Megatron:逐个权重
-1,2,module.module.output_layer.weight,unreduced,1,-0.0017,0.0019,1.25e-11,0.0130,0.0
```

> [!NOTE]说明
>
> - FSDP1 把每层参数展平为一个 `_flat_param`，所以 FSDP 下的 Monitor 粒度是"每层"，不是"每个权重"；Megatron 下是逐个权重。
> - ray worker 的日志不一定回传到 driver，判断是否生效以 CSV 产物为准。
> - GRPO 中若一个 batch 内所有回答得分相同，优势为 0，该步梯度全 0（`norm` 为 0）属正常。

## 3. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| vLLM `wake_up` 报 `CUDA Error: out of memory` / `Free memory on device ... less than desired GPU memory utilization` | 共享节点上其他任务占用显存 | 降低 `actor_rollout_ref.rollout.gpu_memory_utilization`（FSDP 0.3、Megatron 0.2 实测可用），`hy-smi --showpids` 查看占卡进程 |
| 比对报 `Conflicting comparison data: the two columns cannot be monotonically increasing` | 直接调用了上游 `pt_compare`，推理侧 vllm-hcu 的 Triton kernel 条目与训练侧 Triton 条目误配 | 改用 `hcuprobe.integrations.verl.consistent_compare`（skill 比对脚本已使用） |
| FSDP 比对几乎全部 shape error | 直接调用了上游 `pt_compare`：statistics 模式的 shape 检查不忽略 batch=1 维（训练 `[1,S,H]` vs 推理 `[S,H]`） | 改用 `consistent_compare` |
| Megatron 比对 <1% 比例只有约 60% | 推理 TP 与训练 TP 不一致 | 设置 `actor_rollout_ref.rollout.tensor_model_parallel_size` 等于 actor TP |
| `Could not append to config. An item is already at ...sequence_parallel` | 该键在 verl 0.8.0 配置中已存在 | 用 `++` 前缀 |
| `AssertionError: NVTE_FUSED_ATTN set to 1, but expected 0 for attention backend type flash` | 单独覆盖了 `ref.megatron.override_transformer_config.*`，破坏了对 actor 配置的插值 | 只覆盖 actor 的配置 |
| 推理侧 dump 为空 | vLLM CUDA Graph 绕过 hook | `actor_rollout_ref.rollout.enforce_eager=True` |
| Monitor 没有 CSV | 未打 `05-verl-engine-monitor.patch`，或 `MONITOR_CONFIG` 未经 `02` 透传 | 按 2.4 / 2.7 检查；以 `monitor/rank_*/` 下的 CSV 为准 |
| `pip install dist/hcuprobe-*.whl` 找不到文件 | whl 文件名为 `hcu_probe-*.whl`（发行名 `hcu-probe`） | 用 `dist/hcu_probe-*.whl` |
| 比对脚本找不到 `*.xlsx` | 新版 hcuprobe 默认输出 CSV，列名为 `NPU Name` | 使用更新后的 skill 脚本（同时兼容 csv/xlsx） |
