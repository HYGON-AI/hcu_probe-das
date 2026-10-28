---
name: hcu_probe-verl-megatron-v07-integration
version: 1.0.0
description: "在 hcu_verl(verl 0.7.1 官方 patch 版,Megatron 训练后端 + das vllm 0.11 推理)接入 hcu_probe 做训推一致性 dump + 离线 compare。训练侧钩子挂 LEGACY 路径的 MegatronPPOActor(megatron_actor.py),推理侧 patch 上游 vllm gpu_model_runner,用官方 pt_compare(consistent_check + backend=megatron)逐层比对。当用户要给 verl 0.7.1 Megatron 后端接入 hcu_probe、定位 megatron 训练/推理前向钩子、做训练 vs 推理前向数值比对时使用。面向训练已跑通的人(环境/模型/数据/容器就绪),只讲 dump/compare 接入,不含环境搭建。⚠️ 区别于 hcu_probe-verl-megatron-v08-integration(0.8.0 钩 NEW engine 的 transformer_impl.py),也区别于 hcu_probe-verl-fsdp-v07-integration(FSDP 后端)。"
---

> ⚠️ **复验状态(2026-10)**:本 skill 尚未在 hcu_probe-das 26.2.0(同步上游 msprobe 26.2.0)上重新实跑,内容为旧版 dcu_probe 时期的结论;import 已统一为 `hcuprobe.*`。已在新版本复验的是 `hcu_probe-verl-fsdp-v08-integration`、`hcu_probe-verl-megatron-v08-integration`、`hcu_probe-megatron-integration`,见 `docs_hcu/zh/quick_start/`。

# 在 verl 0.7.1 Megatron 后端接入 hcu_probe 做训推一致性 dump + compare

## 定位

RL 训练里同一个模型既做推理(rollout 生成)又做训练(actor 更新)。**训推一致性**验证的是:同一个 prompt,训练侧前向 ≈ 推理侧前向(浮点误差内)。本 skill 教:在训练侧(Megatron)和推理侧(das vllm)各挂 hcu_probe 钩子,dump 出 **prompt prefill** 的算子统计量(L0 模块级),按 `request_id` 把同一 prompt 的两侧 dump 配对,用官方 `pt_compare` 逐层比相对误差。

> **首验配方**(Qwen2.5-0.5B + GSM8K,单节点 8 卡 HCU,`task=statistics level=L0`,只比 prefill):真算子 output 中位误差 0%、叶子算子(layernorm/linear/attention/act_fn/embed)全 <1%、embed 与 MLP down_proj bit-identical(RelErr=0);>1% 的全是 `TransformerLayer` 整层聚合输出(massive-activation 量级差,方法学排除)。与 0.8.0 megatron 结论吻合。

## ⚠️ 版本辨析 + 关键陷阱(照搬 0.8.0 / FSDP 文档必错)

- **后端**:verl **0.7.1**(hcu_verl,官方 patch 版)+ Megatron-LM(hcu_verl 自带)+ mcore 0.15 + 海光 das adaptor。
- **推理栈**:das vllm **0.11.0**(das1.7 内联进上游 vllm,**无独立 `vllm_hcu` 包**;推理 patch 打上游 `vllm/v1/worker/gpu_model_runner.py`)。
- **★★ 最大的坑 —— 训练侧钩子挂 LEGACY 路径,不要走 NEW engine。**
  verl 0.7.1 有两条 actor 执行路径:LEGACY(`megatron_workers.py` → `MegatronPPOActor`,即 `verl/workers/actor/megatron_actor.py`)和 NEW engine(`verl/workers/engine/megatron/transformer_impl.py` 的 `MegatronEngine`)。上游 async 文档 + 0.8.0 skill 的钩子点都在 **NEW engine**,靠 `trainer.use_legacy_worker_impl=disable` 强制走它。**但 0.7.1 的 NEW engine 是不成熟路径,它的 `compute_log_prob` 有权重未物化 bug** —— 前向一进第一个 leaf `word_embeddings` 就 `F.embedding(weight)` 报 `RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet`(EXIT=1),dump 全空。**这个崩与 hcu_probe 无关**(零 dump 插桩也崩,隔离实验坐实)。
  → **本 skill 的钩子挂在 proven-healthy 的 LEGACY `MegatronPPOActor`,launch 时不加 `use_legacy_worker_impl=disable`(默认 auto 即 legacy)。** 这也是与 0.8.0 skill 的核心区别:0.8.0 的 NEW engine 没有这个 bug、钩 `MegatronEngineWithLMHead.forward_step`;0.7.1 必须钩 legacy `MegatronPPOActor.forward_step`。
- **别套 FSDP 的钩子**:FSDP 挂 `dp_actor.compute_log_prob` 里包 `_forward_micro_batch`;megatron legacy 挂 `MegatronPPOActor` 的嵌套 `forward_step`,机制不同。

## 📦 配套文件(本 skill 自包含,光读 SKILL.md 接不了)

| 目录/文件 | 内容 | 怎么用 |
|---|---|---|
| `patches/megatron_actor_dump.patch` | 训练侧钩子 diff(`MegatronPPOActor`,含 ①dump 钩子 + ③request_id) | `cd <verl submodule 根> && git apply <skill>/patches/megatron_actor_dump.patch` |
| `patches/patch_gpu_dump.py` + `dispatch_logger.py` | 推理侧注入脚本(锚点校验 + 幂等 marker `_hcu_debugger`)+ 调度日志模块 | `docker cp` 进容器后跑(见接入步骤) |
| `scripts/run_megatron_compare.py` | compare(官方 `pt_compare` 封装 + request_id 配对 + 挑 token 匹配样本) | 容器内 `python3 run_megatron_compare.py <RUN目录>` |
| `scripts/check_dump.sh` | 验 dump 模块数 / NaN / request_id 交集 | 容器内跑 |
| `scripts/read_tb.sh` | 读 tensorboard 看 baseline 健康(grad_norm/reward/ppo_kl) | 容器内跑,传 tensorboard 目录 |
| `scripts/run_baseline.sh.example` | baseline 普通训练模板(验健康) | `cp` 成 `.sh`,改头部路径 |
| `scripts/run_dump.sh.example` | dump 采集模板(两侧同 run,含 DUMP_* env) | `cp` 成 `.sh`,改头部路径 + config 指向 |
| `config/config_actor.json` + `config_generate.json` | hcu_probe dump 配置(**L0 + statistics**) | 改 `dump_path`,export 给 `DUMP_ACTOR_CONFIG`/`DUMP_CONFIG_PATH` |
| `references/env-and-pitfalls.md` | 环境指针 + 坑 + 锚点定位 | 接入/排错查 |

**接入步骤**(`<...>` 都是示例,按你的环境替换):
```bash
# 1. 训练侧:在 verl submodule 根打 patch(钩子挂 MegatronPPOActor)
cd <hcu_verl 顶层仓>/verl && git apply <skill>/patches/megatron_actor_dump.patch
# 确认 hcu_verl/trainer/constants_ppo.py 已透传 DUMP_ON/DUMP_ACTOR_CONFIG/DUMP_CONFIG_PATH 给 ray worker(没有则加)
# 2. 推理侧:脚本 docker cp 进容器再跑(docker exec 在容器内,看不到宿主 scripts/)
docker cp <skill>/patches/dispatch_logger.py <容器>:/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/
docker cp <skill>/patches/patch_gpu_dump.py  <容器>:/tmp/
docker exec <容器> python3 /tmp/patch_gpu_dump.py   # 注入 gpu_model_runner.py;锚点不符报错不改
# 3. config 放共享盘,dump_path 设 <RUN>/dump_{actor,generate};run 脚本 export 绝对路径
# 4. 按"验证"跑 baseline → dump → compare
```

> **⚠️ patch 换版本/换镜像还有效吗**
> - `megatron_actor_dump.patch` 打的是 **verl 0.7.1 的 `megatron_actor.py` 源码**(你 git clone 的,不在镜像里)。换 verl 版本 → `index` 哈希/`@@` 上下文对不上,`git apply` 会报 "patch does not apply" → 按下方"钩子点(按方法名)"在新版本源码上手动改或重新 `git diff`。
> - `patch_gpu_dump.py` 打的是镜像里 das vllm 的 `gpu_model_runner.py`,锚点是**代码字符串**(非行号)。换 das vllm 版本锚点可能漂移 → 脚本**先校验锚点、不符即报错退出**,按 `references/env-and-pitfalls.md` 重定位。
> - **本 skill 刻意不写具体行号**(不同版本行号会漂),钩子点一律用方法名 + 结构位置描述。

## 前置:baseline 训练健康(铁律,先验健康再 dump)

接 dump 前先跑**普通训练**(无 dump、无 PROMPTS_ONLY,`scripts/run_baseline.sh.example`),确认 megatron 后端训练健康:`actor/grad_norm` 有限无 NaN、`critic/rewards/mean` 有信号且在涨、`response_length` 不截断、`actor/ppo_kl`≈0(logprob 级训推已基本一致)。**metric 主要落 tensorboard**(console 不一定打印),用 `scripts/read_tb.sh` 读。baseline 走默认 auto = LEGACY,**和 dump 采集是同一条路径**(健康验证的路径 == dump 采集的路径)。别拿 dump run 判训练健康——dump run 开 PROMPTS_ONLY 会让训练退化(见下)。

## 一、dump 接入(训练侧钩子在 MegatronPPOActor + 推理侧)

`megatron_actor_dump.patch` 对 `verl/workers/actor/megatron_actor.py` 的 `MegatronPPOActor` 做这几处(**按方法/结构描述,无行号**):

| 落点(方法/结构) | 作用 |
|---|---|
| `__init__` 末尾 | 加 dump 状态:`_debugger`/`_update_actor_logger_fp`/`_should_dump`,以及 ③ 的 `_pending_request_ids`/`_micro_dump_idx` |
| 新增 `_ensure_debugger()` 方法 | 懒初始化 `PrecisionDebugger`(读 `DUMP_ACTOR_CONFIG`);**用 `self.actor_optimizer is None` 跳过 ref**(只 actor dump) |
| 新增 `_log_update_actor_step()` 方法 | 按 `_micro_dump_idx` 从 `_pending_request_ids` 取本 micro 的 request_id,写 `update_actor_log.jsonl` |
| `forward_backward_batch` 里(拿到 `forward_backward_func` 之后) | 调 `_ensure_debugger()` + 按 `forward_only`→phase 设 `_should_dump`(`DUMP_PHASE=log_prob` 时只在 `forward_only=True` 的 compute_log_prob 前向 dump) |
| 嵌套 `forward_step` 的**非融合分支**(`get_mcore_forward_fn` 的 `forward_fn(...)` 调用前后) | `_should_dump` 时 `debugger.start(model=model)` → forward → `_log_update_actor_step()` + `stop()` + `step()` |
| `compute_log_prob` 开头 | ③:从 `data.non_tensor_batch.get("request_id")` 捕获本 rank 有序 request_id 存 `_pending_request_ids`,重置 `_micro_dump_idx=0` |

**为什么 ③ 要在 compute_log_prob 捕获**:legacy `forward_backward_batch` 切 micro_batch 时只 split `.batch`(TensorDict),会**丢掉 non_tensor 的 request_id**(不是 FSDP 那种 select 丢)。所以在 compute_log_prob 层(non_tensor 还在)捕获、按 micro 顺序索引还原;`micro_batch_size=1` 时逐样本对齐。上游注入(agent_loop 把 `vllm_request_id` 写进 `extra_fields`)+ `constants_ppo` 的 DUMP env 透传是前置,通常源码已带。

**推理侧**(`patch_gpu_dump.py` 注入 `gpu_model_runner.py`,与 0.8.0 相同):`GPUModelRunner.__init__` 末尾建 debugger + DispatchLogger(env `DUMP_CONFIG_PATH` 开关、按 pid 隔离 dump_path);`execute_model` 的 `self.model(...)` 前后,**只对 prefill**(`len(scheduled_new_reqs)>0`)做 start/stop/step。`self.model` 是 CUDAGraphWrapper,循环解包 `.runnable` 到真实 nn.Module 再传给 debugger;`seed_all(mode=False)`(mode=True 会让 vllm sampler 的 cumsum deterministic 崩)。

> **② 推理 dump 不挂 rollout(0.7.1 legacy 实测)**:legacy 路径下推理 dump 开着,step0 rollout 照常跑完、两侧 dump 都采到。dump run 采完 step0 后停在 Training Progress 0% 是 **PROMPTS_ONLY 裁 response** 导致(训练-only run 不开推理 dump 也一样停),与推理 dump 无关。**本 skill 只需前几步 dump 即可比对,别期望长跑。**

## 二、验证 dump(看实际内容,不凭产出存在判通过)

dump 目录是容器 root 建的 0750,**必须 `docker exec`(root)读**。跑 `scripts/check_dump.sh` 看:训练侧 dump.json ~391 模块、推理侧 ~290 模块、**NaN=0/Inf=0**、两侧 `request_id` 有交集(DP8 每 rank 1 样本 → 8/8)。dump.json 存在 ≠ 有效,要看模块数 + 统计值非 NaN。

## 三、compare 逐层一致性

```bash
docker exec <容器> bash -c 'pip install -q openpyxl; export PYTHONPATH=<顶层仓>:<顶层仓>/verl:<顶层仓>/Megatron-LM && python3 <RUN>/run_megatron_compare.py <RUN>'
```
`run_megatron_compare.py` 做的:推理 dispatch_log(phase=prefill、requests=1)→ request_id → dump.json;训练 update_actor_log(num_requests=1)→ request_id + step/rank → dump.json;**strip `-8hex` 后缀**配对;挑 token 数完全匹配的样本 → `pt_compare(consistent_check=True, backend="megatron")` → 读 xlsx。**采集级别两侧都是 L0 + statistics**,与上游 msprobe verl 一致性文档一致——训推一致性天然只能在 L0 做(L1 API 级两框架算子分解/融合不同,对不齐)。

**看结果**:真算子 output(排除 `IdentityOp`)NormRelativeErr 分布 —— 叶子算子(layernorm/linear/attention/act_fn/embed)应全 <1%、中位 0%;`TransformerLayer` 整层聚合输出 >1% 属正常(massive-activation + shape `[S,1,H]`vs`[S,H]`,量级差大,方法学排除,不比)。

## 已知坑

- **别加 `use_legacy_worker_impl=disable`**(NEW engine compute_log_prob 权重崩,见上)。
- 改 megatron 源码后 `.pyc` 是容器 root 建的,清理用 `docker exec <容器> find <路径> -name '*.pyc' -delete`(NFS 时钟偏差会让 Python 用旧 pyc)。
- dump 目录 0750 root,验证必 `docker exec`。
- compare 需容器内 `pip install openpyxl`,PYTHONPATH 含 verl + Megatron-LM。
- dump run 的 `PROMPTS_ONLY=1` 会让训练退化(采完 step0 停),正常;训练健康在 baseline 普通 run 单独验。