# verl 0.7.1 Megatron 后端:环境指针 + 坑 + 锚点定位

本 skill 假设 Megatron 训练已在 HCU 跑通(模型/数据/容器/多卡就绪),只讲 dump/compare 接入。环境搭建给指针。

## 环境栈(首验配方)

- **基座镜像**:`harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.10`
  —— 自洽 das 栈:torch 2.7.1 + TE 2.10 + apex + triton(带 `enable_mmacfuse`)+ das vllm 0.11 + mcore 0.15。
  **别用 FSDP 的 torch2.10 镜像**(das megatron 栈只配 torch2.7.1)。
- **committed 环境镜像**(推荐复现):搭好环境的容器 commit 成镜像(示例:`verl071_megatron_env:v1`),含全部 pip 栈。
- **增量依赖**:`pip install codetiming hydra-core omegaconf pylatexenc wandb latex2sympy2_extended math_verify tensorboard pyvers orjson cloudpickle mbridge pycountry`;`pip install --no-deps tensordict liger-kernel`;`pip install --no-deps <hcu_probe whl>`。
- **★ transformers 降 4.57.1 + huggingface-hub 降 0.35.3**(镜像自带 transformers 5.3.0 会让 mcore 转换器崩在 `Qwen2Config.rope_theta`):`pip install --no-deps transformers==4.57.1 huggingface-hub==0.35.3`。das vllm 0.11 仍 import 正常。
- **mcore 权重转换**:`bash examples/scripts/converter_hf_to_mcore.sh --hf_model_path=<HF> --mcore_model_path=<out>`(单卡秒级)。
- **PYTHONPATH**:含顶层仓根 + verl + Megatron-LM(verl `__init__` 有 `from hcu_verl import verl_adaptor`)。
- **compare 阶段**:容器内 `pip install openpyxl`。

## 关键坑

1. **NEW engine compute_log_prob 崩(本 skill 存在的根本原因)**:`use_legacy_worker_impl=disable` 走 NEW engine,其 compute_log_prob 前向在 `word_embeddings` 的 `F.embedding(weight)` 报 `tensor ... data is not allocated yet`(疑 tied-embedding + dist_checkpointing 权重未物化)。**零 dump 插桩也崩** → 与 hcu_probe 无关,是 verl 0.7.1 NEW engine bug。**修法 = 不加该 flag,走 legacy `MegatronPPOActor`**。
2. **stale .pyc + NFS 时钟偏差**:改了 .py 但 Python 用旧 .pyc → 改 megatron/verl 源码后必 `docker exec <容器> find <路径> -name '*.pyc' -delete`(.pyc 是容器 root 建的,宿主普通用户删不掉,报 Permission denied)。
3. **dump 目录 0750 root**:宿主(普通用户)`find`/`cat` 看不到内层 → 验证 dump 必 `docker exec`(root)。
4. **request_id 丢在 micro_batch split**:legacy `forward_backward_batch` 切 micro_batch 只保 `.batch`(TensorDict),non_tensor 的 request_id 丢 → 在 `compute_log_prob` 层捕获(见 SKILL 一节)。
5. **PROMPTS_ONLY 副作用**:dump run 裁 response → 采完 step0 后训练退化、Training Progress 停 0%。正常,别误判成"挂了/推理 dump 拖挂 rollout"。训练健康在 baseline 普通 run 单独验。
6. **metric 主要落 tensorboard**:console 不一定打印 metric dict,用 `read_tb.sh`(EventAccumulator)读 grad_norm/reward/ppo_kl。baseline 脚本 `save_freq=-1` 无 ckpt、不 resume。
7. **改源码后清 pyc / pkill**:pkill 用 `pkill -f mai[n]_ppo` 的 `[x]` trick 防匹配到自身命令行而自杀;改 megatron/verl 源码后必删对应 `.pyc`(容器 root)。

## 推理侧锚点定位(patch_gpu_dump.py 报 ANCHOR_NOT_FOUND 时)

`patch_gpu_dump.py` 靠**代码字符串锚点**注入 `gpu_model_runner.py`,不符即报错退出(不静默打错)。das vllm 版本漂移时锚点变,重定位:

- **__init__ 锚点** = `GPUModelRunner.__init__` 里 `self.sampled_token_ids_pinned_cpu = torch.empty(...)` 那块。在容器内 `grep -n "sampled_token_ids_pinned_cpu" <gpu_model_runner.py>` 找,把 `patch_gpu_dump.py` 的 `INIT_ANCHOR` 换成新版本对应代码块。
- **forward 锚点** = `execute_model` 里 `model_output = self.model(input_ids=..., positions=..., intermediate_tensors=..., inputs_embeds=..., **model_kwargs)` 那块。`grep -n "model_output = self.model" <文件>` 定位,更新 `FWD_ANCHOR`。
- 校验:重跑 `python3 patch_gpu_dump.py` 打印 `PATCHED_GPU_DUMP` 即成功;`ALREADY` 表示已打(幂等 marker `_hcu_debugger`)。

## 训练侧钩子点漂移(git apply 失败时)

`megatron_actor_dump.patch` 换 verl 版本会 "patch does not apply"。按 SKILL"一"节的方法/结构描述在新版本 `megatron_actor.py` 手动改,或改完 `git diff verl/workers/actor/megatron_actor.py` 重新生成 patch。落点(方法名,跨版本稳定,只行号漂):`MegatronPPOActor.__init__` / 新增 `_ensure_debugger` / 新增 `_log_update_actor_step` / `forward_backward_batch`(设 `_should_dump`)/ 嵌套 `forward_step` 非融合分支(包 debugger)/ `compute_log_prob` 开头(捕获 request_id)。