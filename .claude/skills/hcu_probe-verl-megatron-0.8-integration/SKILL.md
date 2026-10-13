---
name: hcu_probe-verl-megatron-0.8-integration
version: 1.0.0
description: "在 hcu_verl(verl 0.8.0 官方 patch 版,Megatron 训练后端 + das vllm 0.11 推理)接入 hcu_probe 做训推一致性 dump + 离线 compare:训练侧 4 处最小侵入 patch(钩子下沉 MegatronEngineWithLMHead.forward_step)、推理侧 patch 上游 vllm gpu_model_runner、用 hcu_probe 官方 pt_compare(consistent_check + backend=megatron + MEGATRON_MODULE_MAP)逐层比对。当用户要给 verl 0.8.0 Megatron 后端接入 hcu_probe、定位 megatron 训练/推理前向钩子、做训练 vs 推理前向数值比对时使用。**面向训练已跑通的人**(环境/模型/数据/容器就绪),只讲 dump/compare 接入,不含环境搭建。⚠️ 区别于 hcu_probe-verl-fsdp-0.7-integration(FSDP 后端,推理 vllm_hcu,自研 compare),也区别于独立 hcu_megatron 框架。"
---

# 在 verl 0.8.0 Megatron 后端接入 hcu_probe 做训推一致性 dump + compare

## 定位

RL 训练里同一个模型既做推理(rollout 生成)又做训练(actor 更新)。**训推一致性**验证的是:同一个 prompt,训练侧前向 ≈ 推理侧前向(浮点误差内)。本 skill 教:在训练侧(Megatron)和推理侧(das vllm)各挂 hcu_probe 钩子 dump 出 **prompt prefill** 的算子统计量,按 `request_id` 把同一 prompt 的两侧 dump 配对,用 hcu_probe **官方 `pt_compare`** 逐层比相对误差,定界精度差在哪侧哪层。

> **首验配方**(真算子 99.4% 误差<1%、中位 0%):Qwen3-8B + gsm8k + Hybrid AgentLoop(共卡)+ `task=statistics level=L0` + 只比 prefill。逐层 compare 多样本脚本见 `scripts/run_megatron_compare_multi.py`,坑与实测状态见 `references/`;原始底稿为内部记录,对外仓可不带。

> **⚠️ 已知问题(2026-06-20 重验,务必知道)—— 一致性可靠,但 run 跑不久 + 验的范围有限**：
> 1. **逐层一致性可靠**：全新独立 run 复现，推理 dump 0 NaN，严格口径(排除整层/IdentityOp)397 可比算子、**2 个独立样本都 99.2% <1%、中位 0%(多数 bit-identical)**。`scripts/run_megatron_compare_multi.py` 做多样本。**但 token 完全匹配的干净可比样本只有 ~2 个**(多数 prompt 的 train/infer prefill token 数不完全对齐)。
> 2. **推理(vllm)侧 run 跑不稳 —— 两种失败,`gpu_mem` 0.18/0.3 都跑不完整(2026-06-20 复测确认)**:观测到两种失败 —— **崩**(`ActorUnavailableError: Socket closed` → exit **137**/OOM,reverify `gpu_mem=0.3` 实测)与**挂死**(进程不退、GPU HCU% 空转 0~5%、日志 30+min 不更新,dumplimit `gpu_mem=0.18` 实测挂 34min),都发生在 rollout/推理-dump 阶段;提高显存只改变失败形态、**不解决**。**dump/compare 只用崩/挂之前的早期 step 产物(实测有效、0 NaN)**。**★ 根因已深挖(2026-06-20):无 dump 的普通 run 同样 `step4` 崩(同 `Socket closed`+exit137+`cpu_memory_used` 333GB)→ dump 洗清(只是死前最后一条日志,非因果);真因(失败模式一)= 全 offload(`param/optimizer/grad_offload=True`)把 8B 优化器态压进 CPU → 单进程 333GB,节点 751GB + ~40 容器共享 → OS OOM-kill;GPU 仅用 17.5/64GB,offload 是有害多余开销。**修法已实测(2026-06-21)= 两层**:关 `optimizer_offload`/`grad_offload` 确实堵住 OOM-崩137(GPU→36GB 常驻无 OOM、步速 100s→32s/it、step1-4 干净),**但 run 仍在 step5 挂死** —— py-spy 实证是**独立的失败模式二:vllm rollout 集合通信挂死**(driver 卡 `ray_trainer.py:643 _validate`、vllm `EngineCore` 卡 `shm_broadcast.acquire_read` 等 8 个 TP worker 响应、GPU 空转),与 offload/host-RAM/hcu_probe 全无关,是 **vllm-on-HCU rollout 稳定性问题**。**已实测确认触发点 = in-loop validation 的 rollout**(2026-06-21 `verl_run_v08_noval`=关 offload + `trainer.test_freq=-1` + 无 dump → **稳跑 step24+、GPU 在算、无崩无挂**;对比 validation 开着死在 step5)→ 挂死是 validation rollout 特有,普通训练 rollout 不挂。非 hcu_probe / 非数值问题**。⇒ 本 skill **只需跑出前几步 dump 即可做一致性比对**,别期望长跑。**长训配方(实测可用)= 关 `optimizer_offload`/`grad_offload`(解模式①OOM)+ `trainer.test_freq=-1` 关 in-loop validation(避模式②挂死)**;要恢复 in-loop 验证须另查 vllm 验证 rollout 为何卡 `shm_broadcast.acquire_read`。
> 3. **训练健康 ≠ 一致性**：dump run 开 PROMPTS_ONLY → 训练退化(`Response mask all False`、无有效 loss）；要验训练健康得另跑不带 dump/PROMPTS_ONLY 的普通 run（已验 megatron 普通训练 grad_norm 有限健康）。这两件事正交,别拿一致性通过冒充训练健康。

> **适用范围 + 版本辨析(照搬别的文档必错)**
> - **后端**:verl 0.8.0 的 **Megatron 训练后端**(= 官方 verl 0.8.0 + NVIDIA Megatron-LM `core_v0.13.1` + 海光 das adaptor;adaptor 只做算子/通信适配,不碰训练逻辑)。
> - **推理栈**:das vllm **0.11.0**(das1.7 内联进上游 vllm,**无独立 `vllm_hcu` 包**)。
> - 训练前向走 0.8.0 的统一 `Engine` 抽象,**没有 FSDP 那样显式的 micro_batch 循环**,PP 调度接管,钩子下沉到 `MegatronEngineWithLMHead.forward_step`——**不要套 FSDP 文档的 `dp_actor.compute_log_prob` 钩子,也不要套 v0.6.x SPMD**。
> - 推理钩子挂上游 `vllm/v1/worker/gpu_model_runner.py`,**不是** FSDP 时代的 `vllm_hcu/v1/hcu_model_runner.py`(那套锚点在 0.8.0 全漂)。
> - **rollout 模式**:本 skill = **Hybrid AgentLoop**(verl v0.7 起的默认异步模式,训推共卡 `hybrid_engine=True`,入口 `main_ppo`,`PROMPTS_ONLY` 裁剪在 `ray_trainer.py`)。**Fully Async**(`FullyAsyncRollouter`+`FullyAsyncTrainer`,训推独立卡池)**未在 HCU 验证**——推理 dump / request_id 注入 / compare 全部可复用,但需:① 入口换 `python3 -m verl.experimental.fully_async_policy.fully_async_main`;② `PROMPTS_ONLY` 裁剪从 `ray_trainer.py` 挪到 `experimental/fully_async_policy/fully_async_trainer.py:FullyAsyncTrainer._fit_generate()`;③ `hybrid_engine=False` + 关 bypass(`algorithm.rollout_correction=null`)+ `TORCHDYNAMO_DISABLE=1`。详见华为上游 `docs/zh/dump/verl_async_consistency_preprocess_dump.md` 的 Fully Async 节。

## 📦 配套文件 + 怎么接(本 skill 自包含,光读 SKILL.md 接不了)

| 目录/文件 | 内容 | 怎么用 |
|---|---|---|
| `patches/` | 训练侧 diff(`01`=①②③ verl submodule / `02`=④ hcu_verl / `04a`+`04b`=das torch2.5.1 兼容坑修) | 各仓 `git apply`(注意 cd 层级,见步骤 1) |
| `scripts/patch_gpu_dump.py` + `dispatch_logger.py` | 推理侧注入脚本(带锚点校验+幂等) + 调度日志模块 | `docker cp` 进容器后跑(步骤 2) |
| `scripts/{run_megatron_compare,batch_compare,verify_id_match}.py` | compare(官方 pt_compare 封装)+ 配对/批量统计 | 容器内 `python3 <脚本> <RUN目录>`(已支持 argv 传 RUN) |
| `scripts/run_megatron_bw8.sh.example` | **launch 模板**(清残留 / ray hycu patch / das env / 三仓 PYTHONPATH / DUMP_* 透传 / wrap run_qwen3) | `cp` 成 `.sh`,改头部路径 + 追加 §四 compare override |
| `config/{config_actor_v08,config_generate_v08}.json` | 训练/推理侧 dump config(L0 statistics) | 放 RUN 目录,改 `dump_path`,export 给 `DUMP_ACTOR_CONFIG`/`DUMP_CONFIG_PATH` |
| `references/` | `compare-guide.md`(pt_compare 详解)/`env-and-pitfalls.md`(环境+坑+锚点定位) | 接入/排错查 |

**接入步骤**:
> 前置约定:`<容器>`/`<RUN>`/镜像 tag 都是**示例,按你的环境替换**。三仓物理布局:**顶层仓根**(含 `hcu_verl/`、`verl/`、`Megatron-LM/` 三子目录)是 02/04b 的 apply 目录;`verl/` submodule 是 01/04a 的 apply 目录(别 cd 进 `hcu_verl/` 包目录本身,否则路径变 `hcu_verl/hcu_verl/...` apply 失败)。commit 是 fork 前上游基线。

```bash
# 1. 训练侧 git apply(注意 cd 层级)
cd <顶层仓根>/verl  && git apply <skill>/patches/01-verl-megatron-dump.patch  <skill>/patches/04a-das-compat-verl.patch
cd <顶层仓根>       && git apply <skill>/patches/02-hcu_verl-constants.patch  <skill>/patches/04b-das-compat-hcu_verl.patch
# 2. 推理侧:脚本先 docker cp 进容器再跑(docker exec 在容器内,看不到宿主 scripts/)
docker cp <skill>/scripts/dispatch_logger.py <容器>:/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/
docker cp <skill>/scripts/patch_gpu_dump.py  <容器>:/tmp/
docker exec <容器> python3 /tmp/patch_gpu_dump.py     # 注入 gpu_model_runner.py;锚点不符报错不改(漂移定位见 references)
# 3. launch:cp 模板,改头部 VERL/OUT/数据/模型路径 + 追加 §四 compare override
cp <skill>/scripts/run_megatron_bw8.sh.example  run_megatron_bw8.sh
# 4. config 放 <RUN> 下,dump_path 设 <RUN>/dump_{actor,generate};**必须 export 绝对路径(patch 里默认值仅占位、文件不存在)**:
export DUMP_ACTOR_CONFIG=<RUN>/config_actor_v08.json  DUMP_CONFIG_PATH=<RUN>/config_generate_v08.json
# 5. 按"四、验证"跑;compare 脚本传 <RUN>:python3 run_megatron_compare.py <RUN>
```

> **⚠️ patch 是什么 + 换版本/换镜像还有效吗(换环境必读)**
> - **`.patch`(`01/02/04a/04b`)= 标准 git 统一差异**(unified diff:记录对源码的增删行 + 上下文),`git apply` 打。`01`=verl 训练钩子(ray_trainer PROMPTS_ONLY + transformer_impl `MegatronEngineWithLMHead.forward_step` 钩子 + llm_server request_id),`02`=hcu_verl env 透传,`04a/04b`=das torch2.5 兼容坑修。
> - **训练侧 .patch 打的是 verl/hcu_verl 源码(你自己 git clone 的,不在 docker 镜像里)→ 与 docker 镜像无关,但与 verl/hcu_verl 版本强相关**:针对 **verl 0.8.0(hcu_verl 头 commit `91455ac`)**;**换 verl 版本 → patch 带的 `index` 哈希/`@@` 行号/上下文对不上,`git apply` 报 "patch does not apply"** → 按 §一表里钩子落点(`MegatronEngineWithLMHead.forward_step`,落点稳定、只行号漂)手动改,或在新版本源码上重新 `git diff` 生成。
> - **推理侧 `patch_gpu_dump.py` 是锚点注入脚本(非 .patch),打的是镜像里 das vllm 的 `gpu_model_runner.py`(dist-packages)→ 这部分才是镜像/das vllm 版本相关**:换 das vllm 版本,锚点字符串可能漂移 → 脚本**先校验锚点、不符即报错退出(不静默打错)**,需按新版本重定位锚点(定位见 `references/env-and-pitfalls.md` 锚点定位节)。
> - **一句话**:这套补丁锚定我们 pinned 的版本(**verl 0.8.0 + 该 das vllm 0.11 镜像**),**不是任意版本/镜像通用**。客户换 verl 版本 → 重生成训练侧 patch(§一落点是稳定指引);换 das 镜像 → 推理锚点重定位(脚本会报错提示)。

> ⚠️ 推理侧用注入脚本 `patch_gpu_dump.py` 而非静态 .patch:幂等(marker `_hcu_debugger`)、锚点核对失败即报错不改、可传 argv 指向源码副本。**vllm/das 版本漂移时锚点会变,定位方法见 `references/env-and-pitfalls.md` 锚点定位节**。compare 脚本 `run_megatron_compare.py` / `verify_*` 是调官方 `pt_compare` + 配对逻辑,别让 AI 重写。

## ⚙️ 前置:HCU 环境(已跑通就跳过)

> 本 skill 假设 Megatron 训练已在 HCU 跑通(模型/数据/容器/多卡就绪)。环境搭建不在范围,只给指针:0.8.0 特有的 Megatron-LM 切 `core_v0.13.1`、HF→mcore 转换(converter 多进程 save 死锁→`cp patch_init`)、das 不支持 sleep mode、共享节点 `gpu_memory_utilization=0.18` 等,全部见 `references/env-and-pitfalls.md`。增量依赖见同文档(注意 `tensordict` 不要 `--no-deps`,compare 阶段要 `pip install openpyxl`)。

## 一、dump 接入(训练 4 处 + 推理 1 处,真实钩子点)

华为 async 原文为 vllm-ascend 写。翻到 das vllm + Megatron 后端共 **5 处改动**(训练 4 + 推理 1),跨 3 个 git 仓,都靠 env 开关门控(不开 dump = 原行为,最小侵入)。

| # | 落点(文件 → 位置) | 作用 |
|---|---|---|
| ① | `verl/workers/engine/megatron/transformer_impl.py`(`__init__`:100 / `forward_backward_batch`:662 / `_ensure_debugger`:770 / `forward_step` forward_fn 前后 :1016) | **核心**:Megatron engine 钩子(懒建 debugger + 前向 start/stop/step + 写 request_id 日志) |
| ② | `verl/trainer/ppo/ray_trainer.py`(`RayPPOTrainer` :1534) | `PROMPTS_ONLY` 裁掉 response,训练前向只算 prompt prefill(对齐推理 prefill) |
| ③ | `verl/workers/rollout/llm_server.py`(`LLMServerClient` :209-220,注入赋值在 :220) | 注入 vllm `request_id` 到 `extra_fields`,贯穿至训练侧 micro_batch |
| ④ | `hcu_verl/trainer/constants_ppo.py`(`PPO_RAY_RUNTIME_ENV` :73) | 透传 `DUMP_ON/DUMP_PHASE/DUMP_ACTOR_CONFIG/DUMP_CONFIG_PATH` env 给 ray worker(shell export 到不了 worker) |
| 推理 | `vllm/v1/worker/gpu_model_runner.py`(`GPUModelRunner.__init__` 末尾 + `execute_model` 的 `model_output=self.model(...)` 前后) | 推理侧 dump,由 `scripts/patch_gpu_dump.py` 注入 |

**钩子点纠正(照搬华为/FSDP 必错)**:
- 训练钩子在 **`MegatronEngineWithLMHead.forward_step` 的 `forward_fn` 前后**(`transformer_impl.py:1016`),**不是** FSDP 的 `dp_actor`,也不是 vllm-ascend 的 `model_runner`。megatron 走 PP 调度无显式 micro_batch 循环,故下沉到 forward_step。
- **`use_fused_kernels=False` 必须**:开 fused 时前向走融合分支绕过 `forward_fn`,钩子不命中。run 脚本加 `actor_rollout_ref.model.use_fused_kernels=False`。
- 只挂 LMHead **非 fused** 路径;`:953` fused_forward_fn 与 ValueHead 路径未挂。
- import 是 **`from hcu_probe.pytorch import PrecisionDebugger, seed_all`**,不是 `msprobe`(容器装的是 hcu_probe fork,无 msprobe 命名空间)。

详细 diff 见 `patches/`,钩子三段式(占位 / 门控 / forward 前后包)代码见 `references/env-and-pitfalls.md`。

## 二、★ hcu_probe 工具(本 skill 重点)

> 接框架精力多耗在 patch,hcu_probe 工具本身怎么用最易被忽略。本节讲透 dump + compare。

### A. dump 工具:PrecisionDebugger
```python
from hcu_probe.pytorch import PrecisionDebugger, seed_all   # 注意:hcu_probe.pytorch,非 msprobe
seed_all(mode=True)                                         # 训练侧 mode=True;推理侧 mode=False(见坑)
dbg = PrecisionDebugger(config_path="config_actor.json")
dbg.start(model=model); <前向>; dbg.stop(); dbg.step()      # 围在 forward 前后
```
config 关键字段(`config/*.json`):`task=statistics`(只统计量免爆盘)、`level=L0`(module 级)、`dump_path`、`rank=[] step=[]`(全采)。产物(目录名 **`stepN`/`rankM`,无下划线**,如 step0/rank0):
- **训练侧**(`dump_path`=`dump_actor`):dump.json 在 `dump_actor/stepN/rankM/dump.json`(**无 pid 层**) + 配对日志 `dump_actor/<pid>/update_actor_log.jsonl`。
- **推理侧**(`patch_gpu_dump.py` 把 `config.dump_path` 改写带 pid):dump.json 在 `dump_generate/<pid>/stepN/rankM/dump.json` + 配对日志 `dump_generate/<pid>/dispatch_log.jsonl`。

两侧配对日志各记 request_id(脚本 `run_megatron_compare.py` 按此 glob 配对,见 :15/:20/:26)。

### B. compare 工具:hcu_probe 官方 pt_compare(本 skill 与 FSDP skill 的最大不同)
**FSDP skill 当时用自研 `verify_consistency.py`(那时 hcu_probe 没集成 verl compare)。verl 0.8.0 起 hcu_probe 自带 verl 训推一致 compare,直接用官方的**:

```python
from hcu_probe.pytorch.compare.pt_compare import pt_compare
pt_compare(
    {"npu_path": 训练dump.json, "bench_path": 推理dump.json},   # npu=训练 / bench=推理,反了 module 全 match 不上
    output_path, consistent_check=True, backend="megatron",
)
```
- 内置 `MEGATRON_MODULE_MAP`(`word_embeddings→embed_tokens` / `linear_qkv→qkv_proj` / `q_layernorm→q_norm` / `linear_fc1→gate_up_proj` / `linear_fc2→down_proj` 等 10 条)自动逐层映射 megatron↔vllm 命名,**无需手写 mapping**。
- `need_cat`+`BACKEND_CAT_RULES` 自动按 seq 维 **concat** 拼接 SP 切片(**不是**平方和聚合)。
- 也支持 `--backend fsdp`(有 `FSDP_MODULE_MAP`)——**FSDP 后端现在也该改用官方 pt_compare**,只是 FSDP 映射在 infer 侧、megatron 在 train 侧,npu/bench 方向要另核对。
- 现成封装:`scripts/run_megatron_compare.py`(单样本)、`scripts/batch_compare.py`(多样本统计)。

### C. compare 只在 L0 可行
L0(module 级,layernorm 等 normed 算子两侧都有、对得齐、不爆盘)是训推 compare 最优粒度。L1(API 级)训练 Megatron-LM vs 推理 das vllm 算子集异构、交集小;tensor(逐元素)推理爆 I/O。L1/tensor 适合单侧精度剖析,不做训推 compare。

## 三、★ compare 正确配置(易错,本次最大教训)

让训练侧 dump 与推理侧**单 prompt prefill** 对齐,需要以下 verl 0.8.0 **实测正确字段**(此前在错误字段上反复试错,经源码定位才纠正):

| 目的 | 正确配置 | ❌ 易错点 |
|---|---|---|
| 训练单 prompt micro_batch(`num_requests=1`) | `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1` + `actor.use_dynamic_bsz=False` + `rollout.log_prob_use_dynamic_bsz=False` | **不是** `ppo_micro_batch_size`(那个只管 update_actor;compute_log_prob 走 `infer_batch` 读 `log_prob_*`)。`use_dynamic_bsz=True` 时按 token 打包会忽略 micro_batch |
| 关 sequence parallel(否则 output 沿 seq 切 1/TP,如 `[43,1,4096]`=344/8) | `+actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=False`(ref 同理) | **不是** `actor.megatron.sequence_parallel`(`vanilla_mbridge=True` 路径不读它,SP 由 mbridge 按 TP>1 自动开) |
| shape 对齐 + 逐卡配对 | `model.use_remove_padding=True` + `trainer.balance_batch=False` + `PROMPTS_ONLY=1` + `rollout.enforce_eager=True` | enforce_eager 必须(否则 vllm compile/cudagraph 绕过推理 hook,dump 为空) |

关 SP 后训练 final_norm 从 `[43,1,4096]` 变 `[72,1,4096]`(完整),与推理 `[72,4096]` 同构。**配对须挑 train/gen embed input token 数完全匹配的样本**——推理部分 prompt 被 chunk,`dump_generate/{pid}/step{k}` 可能取到不完整 chunk(`verify_id_match.py` / `run_megatron_compare.py` 已含该过滤)。详见 `references/compare-guide.md`。

## 四、验证(两阶段:先冒烟验钩子,再产可比对 dump)

### 阶段 A:验钩子链通(`PROMPTS_ONLY=0`,不破坏训练,首次接入务必先跑)
```bash
# run 脚本加 use_fused_kernels=False + enforce_eager=True;launch 设 env(经 ④ 透传 worker):
export DUMP_ON=1 PROMPTS_ONLY=0 DUMP_PHASE=log_prob \
       DUMP_ACTOR_CONFIG=<RUN>/config_actor_v08.json  DUMP_CONFIG_PATH=<RUN>/config_generate_v08.json
bash run_megatron_bw8.sh
# 跑一步后 docker exec(root)确认两侧钩子都通:
#   dump_actor 下有 dump.json + update_actor_log.jsonl(request_ids 非空) + dump_generate 下有 dump.json + dispatch_log.jsonl
```
> `DUMP_PHASE=log_prob` 固定用它:只 dump `compute_log_prob`(forward_only=True)的 prompt 前向,对齐推理 prefill。`all`/`update_actor` 仅用于单侧训练梯度剖析,不用于训推一致。

### 阶段 B:产可比对 dump + compare(`PROMPTS_ONLY=1` + 单 prompt)
launch 调 `run_qwen3_8b_megatron.sh` 的命令行追加这套 override(**让 dump 单 prompt、可比;缺任一条 compare 都对不齐**):
```bash
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=False \
  +actor_rollout_ref.ref.megatron.override_transformer_config.sequence_parallel=False \
  trainer.balance_batch=False
# 另设 n_resp_per_prompt=1 + train_batch_size=ppo_mini_batch_size(单 prompt batch,rollout 快);export PROMPTS_ONLY=1(其余 DUMP_* 同阶段 A)
bash run_megatron_bw8.sh
# compare:脚本先 docker cp 进容器,容器内跑(传 <RUN>=产物目录,== config dump_path 父目录):
docker exec <容器> pip install -q openpyxl                          # 否则存 xlsx 崩
docker cp <skill>/scripts/run_megatron_compare.py <容器>:/tmp/ && docker cp <skill>/scripts/batch_compare.py <容器>:/tmp/
docker exec <容器> python3 /tmp/run_megatron_compare.py <RUN>       # 单样本,xlsx + 逐层误差
docker exec <容器> python3 /tmp/batch_compare.py <RUN>              # 多样本统计通过率
```
> dump.json 容器 root 建、0750,宿主 `find` 看不到 → compare 必须**容器内**跑(`docker exec`,root)。`<RUN>` 填容器内可见的产物目录(共享存储挂载点)。

**已验证结论**:Qwen3-8B 3 个不同长度 prompt(72/80/112 tok),真算子(embed + qkv/proj/mlp/attention)1083 个 **99.4% 误差<1%、中位 0%、embed 两侧 Norm 完全相同 = 训推一致**。整层 `TransformerLayer` output(层封装含 SP/residual,非 normed)与 `IdentityOp`(megatron 把 layernorm 融进 `TELayerNormColumnParallelLinear`)不参与比对;`final_layernorm` 官方映射表未覆盖,但手动验 Norm `1320==1320` 一致。

> **★ 诚实定界(2026-06-20 直接读官方 compare xlsx 复核,务必照实说,别夸成"逐 tensor 全一致")**:训推一致 = **可比叶子算子 + 激活统计量**一致,**不是全部 tensor 一致**。官方 `pt_compare` 的 4867 行实测:**3927 pass**(253 权重同源必同 + 3673 激活统计量一致 + 1)、**939 "error" 里 903 是 `requires_grad` 元数据差**(训练 True/推理 False,benign)、1 是 int dtype(值相同)、**35 是整层 `TransformerLayer`/`IdentityOp`**;**带 RelativeErr 数值的可比算子(~466)误差全 ≈0、0 个真叶子算子发散(2 独立样本 s0/s1 完全一致)**。被排除的 35 个整层输出**量级显著不同**(训练侧残差流 massive-activation max~2.2e4 vs 推理侧 ~5.7,shape `[S,1,H]` vs `[S,H]`)—— 这是方法学排除(不同模块边界 + LLaMA/Qwen 残差 massive-activation 现象),不是 bug,但**不能据此宣称逐 tensor 全一致**。

## 五、坑速查

全部坑(10+ das/版本环境坑 + 4 推理坑 + compare 配置坑)逐条「现象→根因→修复」见 `references/env-and-pitfalls.md`。最高频:
- **推理 dump 为空(modules=0)**:vllm compile+cudagraph 绕过 module hook → `enforce_eager=True`。
- **推理 `seed_all(mode=True)` 崩**(`cumsum` 无确定性实现)→ 推理侧 `seed_all(mode=False)`。
- **`self.model` 是 `CUDAGraphWrapper`** → 循环解包 `.runnable`(`patch_gpu_dump.py` 已处理)。
- **compare module 全 match 不上** → npu/bench 反了(npu=训练 / bench=推理)。
- **无 `vllm_hcu`** → patch 上游 `gpu_model_runner.py`。

## 范例 + 延伸
- 深入(本 skill 自带):`references/{compare-guide.md, env-and-pitfalls.md}`。
- 相关 skill:`hcu_probe-verl-fsdp-0.8-integration`(verl 0.8.0 **FSDP** 后端,das vllm 0.11 + 官方 pt_compare)、`hcu_probe-verl-fsdp-0.7-integration`(verl 0.7.1 FSDP,vllm-hcu + 自研 compare)。
