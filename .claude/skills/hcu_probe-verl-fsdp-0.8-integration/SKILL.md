---
name: hcu_probe-verl-fsdp-0.8-integration
version: 1.0.0
description: "在 hcu_verl(verl 0.8.0 官方 patch 版,FSDP 训练后端 + das vllm 0.11 推理)接入 hcu_probe 做训推一致性 dump + 离线 compare:训练侧钩子下沉 FSDPEngine.forward_backward_batch 循环体、推理侧 patch 上游 vllm gpu_model_runner、用官方 pt_compare(consistent_check + backend=fsdp + FSDP_MODULE_MAP)逐层比对。⚠️ FSDP 硬先决条件 = attn_implementation 必须 eager/sdpa(默认 flash_attention_2 训练每步 grad_norm NaN)。当用户要给 verl 0.8.0 FSDP 后端接入 hcu_probe、定位 FSDP 训练/推理前向钩子、做训练 vs 推理前向数值比对时使用。面向训练已跑通的人(环境/模型/数据/容器就绪),只讲 dump/compare 接入,不含环境搭建。⚠️ 区别于 hcu_probe-verl-megatron-0.8-integration(Megatron 后端,默认可用),也区别于 hcu_probe-verl-fsdp-0.7-integration(verl 0.7.1 FSDP,推理 vllm-hcu、自研 verify_consistency.py)。"
---

# 在 verl 0.8.0 FSDP 后端接入 hcu_probe 做训推一致性 dump + compare

> **前置假设(本 skill 范围外,不重复)**:① 容器/das vllm 0.11/**hcu_probe whl 已装**(`docker exec <容器> pip show hcu_probe verl` 验);② 模型 `Qwen2.5-0.5B-Instruct` + `gsm8k` 已就位(放 `<ROOT>/verl_assets/models` 与 `<ROOT>/data`);③ base 训练已能在 HCU 跑起来(环境搭建不在本 skill 范围,另按你的 HCU 环境流程)。**⚠️ 所有 `scripts/run_fsdp_*.sh` 头部的 `ROOT`/`VERL`/`OUT` 是示例(`/public4/share/tanbo_work` = 作者集群),用前必改成你的实际路径;`--data_path`/`--hf_model_path` 同理。**

## 定位

RL 训练里同一个模型既推理(rollout 生成)又训练(actor 更新)。**训推一致性**验证:同一 prompt,训练侧前向 ≈ 推理侧前向(浮点误差内)。本 skill 教:训练侧(FSDP)和推理侧(das vllm)各挂 hcu_probe 钩子 dump 出 **prompt prefill** 的算子统计量,按 `request_id` 配对,用官方 `pt_compare(backend=fsdp)` 逐层比相对误差。

> **首验配方**:Qwen2.5-0.5B-Instruct + gsm8k + `run_qwen2_5_0.5b.sh`(`--config-name=ppo_trainer` = FSDP 后端)+ Hybrid AgentLoop(共卡)+ `task=statistics level=L0` + 只比 prefill。

## ⚠️ 硬先决条件:attn_implementation 必须 eager / sdpa(本 skill 最关键的一条)

| | 默认 `flash_attention_2` | `eager` / `sdpa` |
|---|---|---|
| 训练 | **坏**:`grad_norm` 每步 NaN、pg_loss/kl 间歇 NaN、reward 卡 0(被 clip/skip 不学) | **健康且在学**:grad_norm 0.004~1.5 有限、**reward 0.0125→0.21(~15x,学会 gsm8k)** |
| dump | 311/317 module output NaN(attention 核产 NaN) | 干净(0 真 NaN) |

- **根因(tripwire 钉死)**:FSDP 走 transformers → das `flash_attn_func`(非 varlen),它在 live 多进程 hybrid 运行时(共卡 vllm + scratch 态)产 NaN —— **运行时状态污染,非 kernel 数学 bug**(确切入参 standalone 重放全干净)。eager/sdpa 不调这个 das kernel 故绕开。
- **对比 Megatron**:Megatron 用 TransformerEngine(TE)融合 attention、根本不碰 das flash_attn 包 → 默认就好。**这是 FSDP 比 Megatron 多的一条先决条件**。
- **修法**:run 脚本 actor 命令行加 `+actor_rollout_ref.model.override_config.attn_implementation=eager`(或 **sdpa,更贴 vllm、更快、o_proj 尾巴更小**)。
- **⚠️ 没满足这条,验什么都没意义**(在坏训练上验一致性是浪费、还会误判)。

## 适用范围 + 版本辨析(照搬别的文档/版本必错)

- **后端**:verl 0.8.0 的 **FSDP 训练后端**(`--config-name=ppo_trainer`;Megatron 后端是 `ppo_megatron_trainer`)。FSDP 训练侧 = **HF 风格模型**(`embed_tokens` / `q_proj`/`k_proj`/`v_proj`(分开)/ `gate_proj`/`up_proj`(分开)/ `o_proj` / `down_proj` / `RMSNorm`)。
- **推理栈**:das vllm **0.11.0**(das1.7 内联进上游 vllm,**无独立 vllm_hcu 包**)——与 Megatron 后端共用同一容器/推理 patch。
- **训练前向走 0.8.0 的 `FSDPEngine`**,钩子下沉到 **`FSDPEngine.forward_backward_batch` 的 micro_batch 循环体**(vs Megatron 的 `forward_step`,vs 0.7.1 FSDP 的 `dp_actor.compute_log_prob`)。
- **区别于 `hcu_probe-verl-fsdp-0.7-integration`(verl 0.7.1 FSDP)**:那套推理用独立 `vllm-hcu` 包、自研 `verify_consistency.py`(只比 normed 算子、逐 prompt 报 80/80);本套用 das vllm 0.11 内联 + 官方 `pt_compare`(逐算子报、比 normed + o_proj + down_proj)。

## 📦 配套文件 + 怎么接(本 skill 自包含)

| 目录/文件 | 内容 | 怎么用 |
|---|---|---|
| `patches/01-verl-fsdp-dump.patch` | 训练侧 dump 钩子(FSDPEngine.forward_backward_batch + PROMPTS_ONLY + request_id) | verl submodule `git apply` |
| `patches/02-hcu_verl-constants.patch` | hcu_verl `constants_ppo.py` 透传 `DUMP_*` env 给 ray worker | 顶层仓根 `git apply` |
| `patches/04a` + `04b` | das torch2.5.1 兼容坑修(同 Megatron) | 各仓 `git apply` |
| `scripts/patch_gpu_dump.py` + `dispatch_logger.py` | 推理侧注入(上游 vllm `gpu_model_runner.py`,与 Megatron 共用)+ 调度日志 | `docker cp` 进容器后跑 |
| `scripts/run_fsdp_compare.py` | **compare 主脚本**(官方 `pt_compare(backend=fsdp)` + 排整层/IdentityOp/act_fn + 逐类型统计) | 容器内 `python3 run_fsdp_compare.py <RUN>` |
| `scripts/batch_compare.py` | 多样本 compare 统计 | 容器内跑 |
| `scripts/fsdp_nan_class.py` | **dump NaN 分类器**(区分 output[0] 真 NaN vs 良性 CE 缓冲) | 容器内跑,review dump 必用(看实际产物再下结论) |
| `scripts/run_fsdp_normal_eager.sh` | **阶段0 健康验证**(eager 普通训练,无 dump,看 reward 上升) | 先跑这个确认先决条件 |
| `scripts/run_fsdp_clean.sh` | **dump run**(eager + PROMPTS_ONLY + 单 prompt,产 dump_actor/dump_generate) | 改头部路径后跑 |
| `config/config_actor_v08.json` `config_generate_v08.json` | 训练/推理 dump config(L0 statistics) | dump run 内已 inline 生成同款,可直接引用 |

**三仓物理布局**(决定 `git apply` 在哪层 cd):
```
<顶层仓根>/            # = hcu_verl_v0.8.0,02/04b apply 在这层(改 hcu_verl/...)
├── hcu_verl/         #   patch 层(constants_ppo.py 等)
├── verl/             # verl submodule,01/04a apply 在这层(cd <顶层仓根>/verl)
└── Megatron-LM/      # 必须在 PYTHONPATH(修法B 的 VerlShapeErrChecker 依赖)
```
**接入步骤**(`<容器>`/`<RUN>`/路径按你的环境替换):
```bash
# 1. 训练侧 git apply
cd <顶层仓根>/verl  && git apply <skill>/patches/01-verl-fsdp-dump.patch  <skill>/patches/04a-das-compat-verl.patch
cd <顶层仓根>       && git apply <skill>/patches/02-hcu_verl-constants.patch  <skill>/patches/04b-das-compat-hcu_verl.patch
# 2. 推理侧:脚本 docker cp 进容器再跑(与 Megatron 共用同一 gpu_model_runner patch)
docker cp <skill>/scripts/dispatch_logger.py <容器>:/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/
docker cp <skill>/scripts/patch_gpu_dump.py  <容器>:/tmp/ && docker exec <容器> python3 /tmp/patch_gpu_dump.py
# 3. ★ 修法B:hcu_probe 的 FSDP statistics+consistent_check shape squeeze 盲区(见「三」),覆盖容器 dist-packages
# 4. 健康验证 → dump → compare(见「四」)
```

> **⚠️ patch 是什么 + 换版本/换镜像还有效吗(换环境必读)**
> - **`.patch`(`01/02/04a/04b`)= 标准 git 统一差异**(unified diff:记录对源码文件的增删行 + 上下文),用 `git apply` 打。`01`=verl 训练钩子(ray_trainer PROMPTS_ONLY + transformer_impl FSDPEngine 钩子 + llm_server request_id),`02`=hcu_verl env 透传,`04a/04b`=das torch2.5 兼容坑修。
> - **训练侧 .patch 打的是 verl/hcu_verl 源码(你自己 git clone 的,不在 docker 镜像里)** → **与 docker 镜像无关,但与 verl/hcu_verl 版本强相关**:这套针对 **verl 0.8.0(hcu_verl 头 commit `91455ac`)**。同版本能 apply;**换 verl 版本 → patch 带的 `index` 哈希/`@@` 行号/上下文对不上,`git apply` 报 "patch does not apply"** → 这时按 §一表里的**钩子落点**(`FSDPEngine.forward_backward_batch` 等,落点稳定、只行号漂)手动改,或在新版本源码上重新 `git diff` 生成 patch。
> - **推理侧 `patch_gpu_dump.py` 不是 .patch,是锚点注入脚本** —— 打的是**镜像里 das vllm 的 `gpu_model_runner.py`**(dist-packages)。**这部分才是镜像/das vllm 版本相关**:换 das vllm 版本,锚点字符串(`INIT_ANCHOR`/`FWD_ANCHOR`)可能漂移 → 脚本**先校验锚点、不符就报错退出(`SystemExit(1)`,不会静默打错)**,需按新版本的 `gpu_model_runner.py` 重定位锚点(比 .patch 鲁棒,但仍要求锚点字符串存在)。
> - **一句话**:这套补丁锚定我们的 pinned 版本(**verl 0.8.0 + 该 das vllm 0.11 镜像**),**不是任意版本/镜像通用**。客户换 verl 版本 → 重生成训练侧 patch(§一落点是稳定指引);换 das 镜像 → 推理锚点重定位(脚本会报错提示,§五坑速查有锚点漂移定位)。

## 一、dump 接入(训练 1 处钩子 + 推理 1 处 + hcu_probe 修法B)

| # | 落点 | 作用 |
|---|---|---|
| ① | `verl/workers/engine/fsdp/transformer_impl.py`(`FSDPEngine.forward_backward_batch` micro_batch 循环体)| **核心**:FSDP 训练钩子(懒建 debugger + 每 micro_batch 前向 start/stop/step + 取 request_id) |
| ② | `verl/trainer/ppo/ray_trainer.py` | `PROMPTS_ONLY` 裁掉 response,训练前向只算 prompt prefill(对齐推理 prefill) |
| ③ | `verl/workers/rollout/llm_server.py`(或 agent_loop) | 注入 vllm `request_id` 贯穿训推(配对靠它) |
| ④ | `hcu_verl/trainer/constants_ppo.py` | 透传 `DUMP_ON/DUMP_PHASE/DUMP_ACTOR_CONFIG/DUMP_CONFIG_PATH` env 给 ray worker |
| 推理 | `vllm/v1/worker/gpu_model_runner.py` | 推理侧 dump(`patch_gpu_dump.py` 注入,门控 `if DUMP_CONFIG_PATH`,与 Megatron 共用)|

**钩子点纠正(照搬必错)**:
- FSDP 训练钩子在 **`FSDPEngine.forward_backward_batch` 的 micro_batch 循环体内**(0.8.0 Engine 抽象),**不是** Megatron 的 `forward_step`,也不是 0.7.1 FSDP 的 `dp_actor.compute_log_prob`。
- **import 是 `from hcu_probe.pytorch import PrecisionDebugger, seed_all`**(非 `msprobe` —— 接入先确认导的是 hcu_probe 这个 fork 包)。

## 二、★ hcu_probe 工具 + 官方 compare(backend=fsdp)

### A. dump:PrecisionDebugger(L0 statistics)
```python
from hcu_probe.pytorch import PrecisionDebugger, seed_all
dbg = PrecisionDebugger(config_path="config_actor.json")
dbg.start(model=model); <前向>; dbg.stop(); dbg.step()
```
config:`task=statistics`、`level=L0`、`dump_path`、`rank=[] step=[]`(全采)。产物 `dump_actor/stepN/rankM/dump.json` + `update_actor_log.jsonl`;推理 `dump_generate/<pid>/stepN/rankM/dump.json` + `dispatch_log.jsonl`。

### B. compare:官方 pt_compare(backend=fsdp)
```python
from hcu_probe.pytorch.compare.pt_compare import pt_compare
pt_compare({"npu_path": 训练dump.json, "bench_path": 推理dump.json}, out, consistent_check=True, backend="fsdp")
```
- 内置 `FSDP_MODULE_MAP`(`q/k/v_proj→qkv_proj`、`gate/up→gate_up_proj` 等)做训推命名映射。
- 封装好的 `scripts/run_fsdp_compare.py` 已含**正确排除口径**(见「三」)+ 逐类型统计,直接用。

## 三、★ compare 正确配置(FSDP 特有,易错)

1. **单 prompt 对齐**:`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1` + 关 dynamic_bsz(让 `num_requests=1`、训练 `[1,L]` 对齐推理 `[L]`)+ `trainer.balance_batch=False` + `PROMPTS_ONLY=1` + `enforce_eager=True`。
2. **★ 修法B(hcu_probe FSDP statistics 盲区,必须打)**:FSDP 训练 dump 是 `[1,S,H]`(HF eager `unsqueeze(0)`)vs 推理 vllm `[S,H]`(packed);hcu_probe 的 squeeze 容差 `VerlShapeErrChecker` **只挂 task=tensor**,statistics 模式恒用严格 `ShapeErrChecker` → 全 shape error、配对几乎为 0。**★ 修法B 已合入 hcu_probe 主仓**(`VerlShapeErrChecker` 类 + `VERL_STATISTICS_CHECKERS` 列表 + `calculator._add_algorithm` 的 `if self.consistent_check` 分支)——**用含此修正的 hcu_probe 即可,不必手改**(本次集群容器已含,compare 跑通 98 算子即证)。**验证**:`docker exec <容器> python3 -c "from hcu_probe.core.compare.indicator_analysis.algorithm import VERL_STATISTICS_CHECKERS, VerlShapeErrChecker; print('修法B OK')"`;**ImportError = hcu_probe 太旧 → 按 `references/hcu_probe-fsdp-statistics-fix.md` 手动加这 3 处(给了精确代码)**。**另:`Megatron-LM` 必须在 `PYTHONPATH`**(VerlShapeErrChecker 依赖,compare 脚本已加)。
3. **排除口径(run_fsdp_compare.py 已内置)**:可比叶子算子 = output、非 wrapper、有 NormRelativeErr;**排除** 整层 `Qwen2DecoderLayer`/`Qwen2MLP`/`Qwen2Attention`(残差和,非 normed)+ `IdentityOp` + **`act_fn`/`SiLU`(融合算子:HF `silu(gate)` vs vllm `SiluAndMul=silu(gate)·up`,不同张量)**。

> ⚠️ **FSDP 比 Megatron 少比 qkv / gate_up**:HF 训练侧 `q/k/v_proj`、`gate/up_proj` 是**分开**的,与推理 vllm 的**融合** `qkv_proj`/`gate_up_proj` 配不上(shape/语义不同)→ 不进可比集。可比集只剩 o_proj / down_proj / layernorm / embed / norm。要补需在 compare 侧先把训练分开的 q/k/v cat 成 qkv 再比(`FSDP_MODULE_MAP` 目前未做这层聚合,是可补盲区)。

## 四、验证(三步:健康 → dump → compare)

### 阶段 0:先验 base 训练健康(eager,无 dump)
```bash
bash scripts/run_fsdp_normal_eager.sh    # eager 普通训练,无 dump/PROMPTS_ONLY
# 看 tensorboard(EventAccumulator 读):reward 应从 ~0.0125 升到 ~0.2、grad_norm 有限无 NaN
```
不健康(默认 flash → grad_norm 每步 NaN)先满足先决条件(改 eager/sdpa),再往下。

### 阶段 1:产可比对 dump(eager + PROMPTS_ONLY + 单 prompt)
```bash
bash scripts/run_fsdp_clean.sh           # 内含 attn=eager + DUMP_ON=1 PROMPTS_ONLY=1 + 单 prompt override
# 产物:<RUN>/dump_actor(训练侧)+ <RUN>/dump_generate(推理侧)+ update_actor_log/dispatch_log
```

### 阶段 2:review dump + compare
```bash
# 0. 脚本先 docker cp 进容器(dump.json 是容器 root 0750,compare/NaN 检查必须容器内跑)
docker cp <skill>/scripts/fsdp_nan_class.py <skill>/scripts/run_fsdp_compare.py <skill>/scripts/batch_compare.py <容器>:/tmp/
docker exec <容器> pip install -q openpyxl                 # pandas 读 xlsx 后端,缺则 compare 静默崩
# 1. ★ review dump NaN:区分真 NaN vs 良性 CE 缓冲。传 <RUN>(= dump_path 父目录)
docker exec <容器> python3 /tmp/fsdp_nan_class.py <RUN>
#   预期:output[0] 含真 NaN=0;唯一 NaN 是 1 个 Triton.cross_entropy_fwd_kernel(良性未初始化 z_losses)。
#   ⚠️ 别被 grep -rl NaN 的"全部文件命中"吓到——那全是良性 CE NaN(每文件都有该模块);output[0]>0 才中止。
# 2. compare(官方 pt_compare backend=fsdp;PYTHONPATH 必须含 Megatron-LM)
docker exec -e PYTHONPATH=<VERL>:<VERL>/verl:<VERL>/Megatron-LM <容器> python3 /tmp/run_fsdp_compare.py <RUN>
```
> 若 compare 配对≈0 / 全 shape error → 修法B 没生效(`references/hcu_probe-fsdp-statistics-fix.md` 验证);若 import 报错 → Megatron-LM 不在 PYTHONPATH。

**已验证结论(2026-06-21,Qwen2.5-0.5B,排整层+IdentityOp+act_fn 后)**:
- **可比叶子算子 98 个**(o_proj 24 / down_proj 24 / input_layernorm 24 / post_attention 24 / model.norm 1 / embed 1);**eager 96/98=97.96% &lt;1%、中位 0.199%**;**sdpa 更优 97/98=98.98% &lt;1%、中位 0.121%**。
- embed 两侧 l2norm 完全相同(误差 0);**唯一 &gt;1% 是 o_proj**(attention 输出,eager 2.67% / sdpa 1.78% = train attn kernel vs vllm attn kernel 的代价,非 bug)。
- **诚实定界**:一致性 = 可比叶子算子一致,**非逐 tensor 全一致**;不比整层/IdentityOp/融合 act_fn;qkv/gate_up 因 HF 分开 vs vllm 融合不可比。
- **dump 非侵入**:eager 普通训练 + 训练侧 dump,reward 轨迹与无-dump run 重合、dump 0 真 NaN。

## 五、坑速查

- **训练 grad_norm 每步 NaN / dump 大面积 NaN** → 默认 flash_attention_2,改 `attn_implementation=eager`/`sdpa`(本 skill 头号坑)。
- **`grep -rl NaN` 全部文件命中** → 良性 CE 缓冲(`Triton.cross_entropy_fwd_kernel` 未初始化 z_losses),用 `fsdp_nan_class.py` 看 output[0] 才算真 NaN。
- **compare 配对几乎为 0 / 全 shape error** → 修法B 没打(FSDP statistics+consistent_check 不 squeeze);且 `Megatron-LM` 要在 PYTHONPATH。
- **compare 可比算子过少 / qkv 缺失** → 正常:HF 分开的 q/k/v、gate/up 配不上 vllm 融合,不进可比集(非 bug,是 HF 分开 vs vllm 融合的算子对不齐)。
- **推理 dump 为空** → vllm compile/cudagraph 绕过 hook,`enforce_eager=True`;`self.model` 是 CUDAGraphWrapper 要解包(`patch_gpu_dump.py` 已处理)。
- **compare 整层 diff 7000%+** → 不是发散:整层 output 含残差(massive activation),与推理捕获点/口径不同,方法学固定排除。

## 范例 + 延伸
- 深入机制(本 skill 自带):`references/flash-nan-and-sdpa.md`(flash NaN 的 tripwire 定位 + eager/sdpa 对比)、`references/hcu_probe-fsdp-statistics-fix.md`(修法B 精确代码 + 验证)。
- 相关 skill:`hcu_probe-verl-megatron-0.8-integration`(Megatron 后端,默认可用)、`hcu_probe-verl-fsdp-0.7-integration`(verl 0.7.1 FSDP,vllm-hcu + 自研 compare)。
