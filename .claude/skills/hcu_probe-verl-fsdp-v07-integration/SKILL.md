---
name: hcu_probe-verl-fsdp-v07-integration
version: 1.0.0
description: "在 hcu_verl(>0.7,verl 官方 patch 版,FSDP 后端)接入 hcu_probe 做训推一致性 dump + 离线 compare,以及 monitor 三通道训练监控,用 --hcu-probe 命令行 flag 控制采集模式(off/L0/L1/tensor/monitor/combined)。当用户要给 verl/hcu_verl >0.7 接入 hcu_probe、定位前向/训练钩子、做训练 vs 推理前向数值比对、或接 monitor 时使用。**面向训练已跑通的人**(环境/模型/数据/容器就绪),只讲 dump/compare 接入,不含环境搭建。"
---

# 在 hcu_verl(>0.7)上接入 hcu_probe 做训推一致性 dump + compare

## 定位

RL 训练里**同一个模型**既做**推理**(vllm rollout 生成 response)又做**训练**(FSDP actor 的 `compute_log_prob` / `update_actor` 前向)。两侧对**同一 prompt 的 prefill 前向**理论上应数值一致。本 skill 教你:用 hcu_probe 在两侧分别 dump prefill 前向的统计量,按 `request_id` 配对、比相对误差,**定界精度差到底在训练侧还是推理侧**。

首验配方:**Qwen2.5-0.5B-Instruct / GSM8K / Hybrid AgentLoop(colocated)**,`task=statistics`(只比统计量免爆盘)、`level=L0`、只 prefill。本 skill 自包含(SKILL.md + `scripts/` + `patches/` + `config/` + `references/`),只讲接入操作。

> 适用范围:hcu_verl **0.7.1**(对应底座 verl `bec9ef74`/v0.7.1)+ **FSDP** 后端 + das vllm(`vllm_hcu`)。verl 的 Megatron 后端是独立工作。**版本辨析(关键)**:verl v0.7 起移除 SPMD、转异步 AgentLoop 调度,采集方案随之变。**不要套 v0.6.x 的 fsdp/megatron 专门文档**(那是 SPMD 架构,钩子点不同)。

---

## 📦 配套文件 + 怎么接(本 skill 自包含,光读 SKILL.md 接不了)

本 skill 目录捆了三样**可跑的东西**,必须配合用:

| 目录 | 内容 | 怎么用 |
|---|---|---|
| `patches/` | 6 处代码改动的 git diff(3 个 `.patch`):dump ①-⑤ + monitor + HCU 适配 | `git apply`(见下) |
| `scripts/` | hcu_probe 工具 + 启动脚本(`manage.sh` 带 `--hcu-probe` flag、`verify_consistency`(自研 compare)/`run07_filtered.py`+`run_official_compare.sh`(官方 pt_compare 复核,§D)/`dispatch_logger`/`find_compare_pairs`/`read_tb`/`freeze_dump`/`patch_hcu_dump`/`run_*.sh`/`patch_rocm_visibility.py`) | scp 进容器,按"三、验证"节跑 |
| `config/` | dump + monitor 配置模板(`config_actor/generate.json` L0+statistics;`config_monitor.json` 三通道) | 改 `dump_path`/`output_dir` 后用 |
| `references/` | 深挖原理(`compare-blindspots.md`:compare 盲区 + 数据流) | 接通不必读;理解/改 compare 时读 |

**接入四步**:

> **前置约定**:路径占位符 `/data/hcu_probe_verl`(= 你的运行盘,记 `$RUN`)、容器名 `<container>` 都是**示例,全部替换成你自己的**(见步 3 一键替换)。**三仓根布局**(决定 git apply 的 cwd 与 `-p` 层级):`verl` 仓根直含 `experimental/ workers/ trainer/`;`hcu_verl` 仓根**同时含**嵌套 `hcu_verl/` 包 + 顶层 `examples/`(patch 02 两个前缀 `hcu_verl/`、`examples/` 都从这个根 `-p1` 命中);`vllm-hcu` 是容器内 `site-packages/vllm_hcu`。下列 commit 是 **fork 前的上游基线**。

```bash
# 1. 三仓 checkout 到上游基线 → 应用 patches/(cwd = 各仓根;git apply 默认 -p1;先 --check 自检再 apply)
cd <verl_submodule> && git checkout bec9ef74 && git apply --check <skill>/patches/01-verl-submodule.patch && git apply <skill>/patches/01-verl-submodule.patch  # verl v0.7.1 基线
cd <hcu_verl_main>  && git checkout ce853ab  && git apply --check <skill>/patches/02-hcu_verl-main.patch  && git apply <skill>/patches/02-hcu_verl-main.patch   # cwd = 含 hcu_verl/ 包 + examples/ 的仓根
cd <vllm-hcu>       && git checkout 4b4388   && git apply --check <skill>/patches/03-vllm-hcu.patch      && git apply <skill>/patches/03-vllm-hcu.patch       # OpenDAS 公开仓
# 2. 推理侧注入(①dispatch_logger + ②hcu_model_runner 钩子)——【二选一,别都做,两者内容逐字相同】:
#    A. 有 vllm-hcu 源码仓:patch 03 已改好,把改后 2 个 .py(hcu_model_runner + dispatch_logger)同步进容器 site-packages/vllm_hcu/v1/
#    B. 只有容器闭源 whl(无源码):把 scripts/{dispatch_logger.py,patch_hcu_dump.py} 放 $RUN,跑 manage.sh infer-deploy
#       → infer-deploy 自动:动态定位 vllm_hcu/v1(不写死 py 版本)+ 落 dispatch_logger.py + 字符串锚点注入 hcu_model_runner + py_compile 自检(推荐;复用官方 .so,不重打 wheel)
# 3. 一键把占位符换成你的运行盘(同一字符串,散落 config/scripts/patches 约 15 处):
#    RUN=<你的运行盘>; grep -rl /data/hcu_probe_verl <skill>/ | xargs sed -i "s#/data/hcu_probe_verl#$RUN#g"
#    (漏改会让 patch 里 config 默认路径与真跑路径对不上 → dump 落错盘)
# 4. 按"三、验证"节跑:start --hcu-probe L0(验训推一致)/ off(看收敛)/ monitor(三通道)
```

> ⚠️ **`scripts/verify_consistency.py` 是 compare 核心 —— AI 写不出 fused residual 绕盲区逻辑(取 `output[1]`、batch prefill 可比性检查、strip `-8hex` 配对),必须用这个,别让 AI 重写。** patches 里 ①`dispatch_logger.py`(新建)②`hcu_model_runner` 注入是推理侧;③④⑤ + HCU 绑卡 + prefix fix 在 verl/hcu_verl。

---

## ⚙️ 前置:HCU 环境(已跑通就跳过本节)

> 本 skill 假设 **verl 已在你的 HCU 环境跑通 base 训练**(模型 / 数据 / 容器 / 多卡都就绪)——**环境怎么搭不是本 skill 的范围**,本 skill 只讲 dump/compare 接入。HCU 适配的 6 类阻抗失配(ray hycu 补丁 / HIP-CUDA 可见性 / cudagraph OOM / HIP 碎片 / sleep mode / colocated 显存)按你的 HCU 环境流程处理(本 skill `scripts/patch_rocm_visibility.py` 提供 HIP-CUDA 可见性补丁示例,`patches/` 提供 ray / vllm 侧改动)。

---

## 一、dump 接入(5 处改动,真实钩子点)

华为 async 训推一致 dump 原文为 **vllm-ascend** 写,共 6 处;翻到 **das vllm `vllm_hcu` + hcu_verl 0.7.1** 是 **5 处**(跳过 fully-async 那处)。改动跨 **3 个 git 仓**(verl submodule / hcu_verl 主仓 / das vllm-hcu),都靠 **env 开关**门控——不开 dump 就是原行为(最小侵入)。

| # | 落点(文件 → 位置) | 作用 |
|---|---|---|
| ⑤ | `verl/trainer/ppo/ray_trainer.py` → `fit()`(reward 后,`compute_old_log_prob` 前) | `PROMPTS_ONLY` 裁掉 response、只留 prompt,对齐推理 prefill 粒度 |
| ③ | `verl/workers/actor/dp_actor.py` → `compute_log_prob` 包 `_forward_micro_batch` | 训练侧前向 dump |
| ④ | `verl/experimental/agent_loop/agent_loop.py` → `AsyncLLMServerManager.generate` | 注入训推配对用的 `request_id` |
| ④ | `dp_actor.py` → `data.select` 处 | 把 `request_id` 加进保留键(`non_tensor_batch_keys`),否则被滤掉 |
| ② | `vllm_hcu/v1/hcu_model_runner.py::GPUModelRunner`(`__init__` 建 debugger;`execute_model` 围 `_model_forward`) | 推理侧前向 dump,**只 prefill 门控**防爆盘 |
| ① | `vllm_hcu/v1/dispatch_logger.py`(新建) | 记推理侧 request_id(das 无现成 PrecisionDebugger,要连 debugger 一起加) |

> ⚠️ 表内**不标行号**——das vllm 0.18.1 / verl 版本间行号会漂,`patches/` 和 `patch_hcu_dump.py` 全靠**字符串锚点 + diff 上下文**定位、不靠行号;认落点看「文件 + 类/方法名」即可。

**钩子点纠正(华为假设 ≠ 实际落点,照搬必错)**:

- ③ 华为挂 `FSDPEngine.forward_backward_batch`(v0.6.x SPMD 路径);**hcu_verl 0.7.1 actor 实际走 `DataParallelPPOActor`,钩子必须挂 `dp_actor.compute_log_prob`**,别套 v0.6.x 文档移到 `FSDPEngine`。
- ④ 华为落 `verl/workers/rollout/llm_server.py`;**verl 0.7.1 已删 `llm_server.py`,rollout 重构成默认 async AgentLoop,落点是 `agent_loop.py`**(`④ 在 agent_loop 注入` 即证版本=async)。

env 透传补一处:`hcu_verl/trainer/constants_ppo.py → PPO_RAY_RUNTIME_ENV`,把 dump 开关(`DUMP_ON` 等)+ monitor 配置(`MONITOR_CONFIG`)透传给 ray worker(shell `export` 到不了 worker)。monitor 是第 6 处独立钩子(挂 `update_policy` 而非前向),详见下方参数化节。

> 部署方式:纯 Python 改动。训练侧(③④⑤)在 hcu_verl 源码,scp 进容器同路径即可;推理侧(①②)在容器闭源 `vllm_hcu` wheel,同步 `.py` 进容器 `site-packages/vllm_hcu/v1/`(复用官方编译的 `.so`,**不必重打 wheel**)。三仓均无 push 权限,发 PR 须各自 fork。

---

## 二、★ hcu_probe 工具(本 skill 重点)

> 接框架时精力多耗在打通 patch,**hcu_probe 这两个工具本身怎么用反而最容易被忽略**。务必讲透。

### A. dump 工具:`PrecisionDebugger`

forward 前后围一圈,三个 API 即可:

```python
from hcuprobe.pytorch import PrecisionDebugger
dbg = PrecisionDebugger(config_path)   # 读 config.json
dbg.start(model=model)                 # forward 之前
# output = model_forward(...)
dbg.stop(); dbg.step()                 # forward 之后 + 步进
```

config.json 关键字段(本实验取值,见本 skill `config/config_actor.json` / `config/config_generate.json`):

| 字段 | 取值 | 本实验 |
|---|---|---|
| `task` | `statistics`(只 Max/Min/Mean/Norm,免 tensor 爆盘)/ `tensor`(逐元素) | **statistics** |
| `level` | `L0`(module)/ `L1`(API)/ `L2`(kernel) | **L0** |
| `dump_path` | 落盘目录,训推两侧**必须不同** | dump_actor / dump_generate |
| `rank` / `step` | 过滤(空 `[]` = 全) | `[]` |

产物:`dump.json`(每次 forward 一份,记算子统计量)+ 配对日志(训练侧 `update_actor_log.jsonl` / 推理侧 `dispatch_log.jsonl` 各记 request_id)。配对脚本 `scripts/find_compare_pairs.py`:选推理侧 `phase=prefill` 且 `requests` 数=1 的拿 request_id → 训练侧搜同 id(strip das 的 `-<8hex>` 后缀)→ 读两边 dump.json 比。

### B. compare 工具:vanilla 自带 compare 不能开箱用(consistent_check 版见 §D)

hcu_probe **vanilla** compare(`hcuprobe compare` CLI 或裸 `pt_compare`,**不开 `consistent_check`**)本场景**不能开箱用**,会满屏假 diff、大概率误判"严重不一致":

| | vanilla compare(不开 consistent_check) | `verify_consistency.py`(外挂) |
|---|---|---|
| 入口 | 裸 `pt_compare` / `hcuprobe compare` CLI | `python3 verify_consistency.py [run_dir] [--threshold 0.01]` |
| 状态 | ❌ `hcuprobe compare` CLI 崩(import 已删模块);裸 `pt_compare` 不开 consistent_check → fused residual / batch prefill 盲区假 diff | ✅ 绕盲区可用 |
| 怎么比 | 通用 mapping,不认 das fused residual | request_id 配对 + **只比 normed 算子**(layernorm output)+ 阈值 |
| 结果 | 满屏假 diff | 1% 阈值 **80/80** 一致 |

> ✅ **但官方 `pt_compare` 开 `consistent_check=True, backend="fsdp"` 是另一条路、能用** —— consistent_check 按四元组语义坐标配对**所有粒度**算子,复核脚本 `run07_filtered.py` **在分析阶段**剔掉整层 `Qwen2DecoderLayer`/`act_fn`(融合语义不同)、只看叶子算子 output,正好避开下面 fused residual 那个「整层」坑;过滤 das 交错模块后实测**可比叶子 98/98 <1%**(比自研只比 norm 的 80/80 覆盖更广)。两者互补,详见 **§D**。

**核心盲区 — fused residual(das vllm 特有,华为 vllm-ascend 无)**:das 的 `Qwen2DecoderLayer` 为吞吐返回 `(normed_hidden, residual)` **两个** output,HF 只返回一个完整 hidden。自带 compare 把 `HF.output[0]`(Max≈1688)错配到 `vllm.output[0]`(normed,Max≈2)→ 制造几千倍假 diff。

解法:**只信 normed 算子(layernorm)比对**——fused residual 让整层 output 拆成两半(mlp 输出 + 残差),但 layernorm 内部两侧都算 `rmsnorm(完整残差)`、语义完全一致,**在 norm 这个"检查点"两流凑齐、可比;在整层 output 这个"交接点"才拆开、不可比**。所以专挑每层的 `input_layernorm` / `post_attention_layernorm` output 比就绕开了。**完整原理(三个盲区 + 数据流层面"检查点 vs 交接点"图解)见 [`references/compare-blindspots.md`](references/compare-blindspots.md)。**

> 别直接改 hcu_probe core 去认 fused residual——compare 逻辑在 `python/hcuprobe/core/compare/`,与华为 msprobe 上游字字一致,改了破坏一致性 + 未来 merge 冲突。正道是给华为 msprobe 上游提 PR,HCU 这边把 `verify_consistency.py` 当专用脚本(归 `integrations/verl` 或 examples 仓)。

### C. compare 只在 L0 可行(L1/tensor 实测受限)

**采集(dump)≠ 比对(compare)**:6 模式都能**采集** dump(L0/L1/tensor 粒度均集群实测落盘),但**训推 compare 只 L0 可行**:

- **L1(API 级)训推算子对不齐(机制层面,未在 verl-0.7 实测)**:训练侧 = HF Qwen2 + FSDP(`Distributed.all_gather` + HF 算子分解),推理侧 = das vllm(`Torch.addmm` + 融合 kernel),**异构框架 API 序列根本不重叠**——这是 RL 训推(训练框架 ≠ 推理框架)的本质限制。华为 msprobe L1 compare 依赖"同实现同算子名",verl 这种 HF↔vllm 异构对不上,故 L1 训推 compare 预期不可行(属机制推断,**未实跑坐实**)。→ L1 若做**单侧** API 级精度剖析可开,需要时自行实跑验证。
- **tensor(逐元素)推理侧爆 I/O**:推理侧逐元素 dump 每个 prefill 写几百 `.pt`,拖垮 rollout(实测推理钩子在时 10min+ 不出 step,关推理钩子后训练侧秒落)。→ tensor 适合**单侧**逐元素留档。
- **L0 是训推 compare 最优粒度**:module 级算子两侧都有、对得齐、不爆盘。`verify_consistency.py` 保守只挑 normed 算子(layernorm)比(80/80);官方 `pt_compare(consistent_check)` 能把可比范围扩到**全部叶子算子**(qkv/o/gate_up/down + norm,98/98,见 §D)——**同一份 L0 dump、两工具覆盖不同,是互补,不是「只能比 norm」**。

> 三个 compare 盲区(fused residual / batch prefill / dual_monotonic_sort)+ 数据流层面"为什么只比 normed"的完整原理 → [`references/compare-blindspots.md`](references/compare-blindspots.md)。

### D. 官方 `pt_compare --backend fsdp` 复核(过滤版)

**分工**:日常/开箱用轻量的 `verify_consistency.py`(§B,只比 norm、无重前置);要**权威交叉核对、或想覆盖到 o_proj/down_proj 等全部叶子**时再上官方 `pt_compare`(需过滤交错模块 + 修法B 容器 + Megatron-LM 在 PYTHONPATH,前置重)——两者互补不替代。

自研 `verify_consistency.py` 的 80/80 结论,可用**官方 `pt_compare(consistent_check=True, backend="fsdp")`** 交叉核对——但**不是零改动 drop-in**,要先过滤一批模块:

- **未过滤的官方 compare(即便已开 `consistent_check`)在 0.7 dump 上必挂 `dual_monotonic_sort: Conflicting comparison data`**:0.7 用 das vllm **0.18.1**,会多 dump 72 个训练侧(HF FSDP)无对应的交错模块——`Triton.kernel_unified_attention_2d`×24 + `Triton.reshape_and_cache_kernel_flash`×24 + `rotary_emb.HcuRotaryEmbedding`×24。这些交错项让官方 `consistent_match` 的两列位置无法同时单调递增 → raise。**这正是当年 0.7 弃用 vanilla 工具、改写自研脚本的真因**(某些 das vllm 版本模块序干净则官方能直接过)。
- **修法 = 复核前把这批 bench-only 交错模块从 gen dump.json 过滤掉**(整目录拷 gen rank 目录 → 改 dump.json `data`,剔 `Triton\.|rotary_emb|reshape_and_cache|unified_attention`)再喂**原版**官方 pt_compare(不带 monkey-patch)。脚本 = `scripts/run07_filtered.py`(过滤 + 跑 pt_compare + 拆解误差分布)+ `scripts/run_official_compare.sh`(容器内环境自检 + 调 run07_filtered)。
- **前提**:跑 compare 需"**修法B**"(statistics 模式用 `VerlShapeErrChecker`,FSDP 训练 `[1,S,H]` vs 推理 `[S,H]` 的 shape squeeze;hcu_probe-das 的 core/ 与上游一致不含它,请用 `hcuprobe.integrations.verl.consistent_compare` 代替直接调 `pt_compare`)+ **Megatron-LM 在 PYTHONPATH** + openpyxl + HCU 设备挂载(pt_compare import torch → dlopen flash_attn,缺设备 SIGABRT 134)。含"修法B"的 compare 容器就绪后离线复用即可,**无需重跑训练**(dump 冻结件在则直接比)。
- **实测结果**:可比叶子算子 **98 个全 <1%**(98/98,中位 0.153%、max 0.613% o_proj;<0.5% 97/98),embed 两侧 l2norm 完全相同 → **官方工具确认自研 80/80,且扩到 o_proj/down_proj 仍全 <1%**。
- **compare 侧配对/排除逻辑**:consistent_check 按 `[方向·input-output槽·层号·归一化模块名]` 四元组配对(非算子名);q/k/v 三个训练算子聚合成一行再和推理 `qkv_proj` 比一次(Max 取三者最大、Min 取三者最小无损重构,Mean/Norm 只拼串)。分析时排除整层 `Qwen2DecoderLayer`/`Qwen2MLP`/`act_fn`/`IdentityOp`(融合语义不同),只看叶子算子 output 的 `NormRelativeErr`。
- **教训**:官方 pt_compare 跨 das vllm 版本复核旧 dump,先查 bench 侧是否有该版本特有的交错模块撑爆 `dual_monotonic_sort`,过滤后即可,别以为工具坏了。

---

## 参数化:`--hcu-probe <mode>` 命令行 flag 控制采集模式

用**一个命令行 flag** 选采集模式,不硬编码(不传 = 原行为,最小侵入):

```bash
manage.sh start --hcu-probe <off|L0|L1|tensor|monitor|combined>
```

verl 是**手动钩子**(非 megatron 的 Feature 0 行代码机制),所以 flag 在**启动脚本层解析 → 设 env**,接入代码(`dp_actor.py` / `constants_ppo.py`)读 env。6 模式:

| mode | 行为 | 设的 env | 验什么 | 状态 |
|---|---|---|---|---|
| `off`(默认) | 纯训练 | — | — | — |
| `L0` | 训推一致 dump,module 级 statistics | `DUMP_ON=1 PROMPTS_ONLY=1` + actor/generate config | 训推一致(前向) | ✅ 已验证 |
| `L1` | 训推一致 dump,API 级 | + `TORCHDYNAMO_DISABLE=1`,派生 L1 config | 训推一致(更细) | ⚠️ verl 上**未实测**(派生 config 机制就绪;L1 训推 compare 本就不可行,见 §C) |
| `tensor` | 训推一致 dump,逐元素 | 派生 tensor config | 训推一致(精细) | ⚠️ verl 上**未系统实测**(机制就绪;推理逐元素极慢/易爆盘) |
| `monitor` | 训练三通道监控 | `MONITOR_CONFIG=config_monitor.json` | 训练健康(梯度/优化器/参数) | ✅ 已验证(8rank×3通道 csv) |
| `combined` | 正常训练 + 前向 dump + monitor | `DUMP_ON=1`(不裁)+ `MONITOR_CONFIG` | 前向 + 三通道 | ✅ 已验证(dump+monitor 叠加) |

L1/tensor 的 config 由 `manage.sh` 从 L0 base 自动派生(`gen_cfg` 改 `level`/`task` 字段),不必手建文件。

**⚠️ verl vs megatron 的 `combined` 不同**:megatron dump 在正常训练中抓前向,能和 monitor 直接叠加;**verl 的训推一致 dump(L0/L1)必须 `PROMPTS_ONLY` 裁 response 对齐推理 prefill → 破坏训练(梯度无意义)→ 与 monitor 看真实梯度互斥**(一致性 dump 与训练监控是两次配置相反、不能同跑的独立 run)。故 verl `combined` =「**正常训练(不裁)+ 前向统计 dump + monitor**」,此处 dump 是「前向统计」非「训推一致」;真训推一致 dump 仍走 `L0`/`L1` 独立 run。

### monitor 接入(本次新增钩子,第 6 处)

monitor 与 dump **钩子点不同**:dump 挂 `compute_log_prob`(前向,`torch.no_grad`),**monitor 挂 `dp_actor.update_policy`**——真正做 `loss.backward()` + `_optimizer_step()` 的地方,且能拿到 `self.actor_module`(model)+ `self.actor_optimizer`(optimizer),正是 `TrainerMonitorV2.start(model, optimizer)` 所需。

| 落点 | 作用 |
|---|---|
| `dp_actor.py::__init__` | `_monitor=None` 占位 |
| `dp_actor.py::_ensure_monitor`(懒初始化) | 读 `MONITOR_CONFIG` → **先 `os.makedirs(output_dir)`** → `TrainerMonitorV2(cfg, fr="pytorch").start(model=actor_module, optimizer=actor_optimizer)`;仅 actor(有 optimizer),ref 跳过 |
| `dp_actor.py::update_policy` 开头 | 调 `_ensure_monitor()` |
| `dp_actor.py::update_policy` 每次 `_optimizer_step()` 后 | `self._monitor.step()` 采三通道 |
| `constants_ppo.py::PPO_RAY_RUNTIME_ENV` | 透传 `MONITOR_CONFIG` 给 ray worker(shell export 到不了 worker) |

monitor config(`config/config_monitor.json`)三通道 + 约束(照搬 megatron):
- `patch_optimizer_step: false` —— 手动 `monitor.step()` 驱动(避免与自动 patch 重复计数)
- **不开 `module` 通道** —— FSDP 包了 model,module hook 易抓到 wrapper(同 megatron pipeline 冲突的保守做法)
- 三通道 = `weight_grad` / `optimizer`(`mv_distribution`)/ `param`(`param_distribution`),`ops=[min,max,mean,norm,nans]`

> ⚠️ **集群实测暴露的关键坑(已修)**:`TrainerMonitorV2` 构造时 `CSVWriterV2.make_dir` 调 msprobe 安全检查 `check_path_exists`,**要求 `output_dir` 父目录预先存在、不递归创建**;缺则抛 `FileCheckException`,被 `_ensure_monitor` 的 try/except **静默吞掉**(`_monitor=None`,训练照常但无 csv;且 ray worker 的 `logger.warning` marker 常传不回 driver stdout → 看似什么都没发生)。故 `_ensure_monitor` 构造前必须 `os.makedirs(output_dir, exist_ok=True)`。**monitor 验证以 `verl_run/monitor/rank*/` 的 csv 产物为准,别只看 log marker。**

> **设计原则**(所有框架通用):接入做成「参数可开关 + 采集模式可选」,不写死一种。verl 已集群实测 **L0 dump**(80/80 训推一致)+ **monitor**(单节点 8 卡 × 3 通道 csv 落盘);**L1 / tensor 机制就绪、verl 上未系统实测**(L1 训推 compare 本就不可行、tensor 推理侧易爆盘,见 §C);接的人按需开 + 自验。

---

## 三、验证(三类,各自独立 run)

> ⚠️ **下面「前向一致 / 端到端收敛」是两个独立验证目标,与 `--hcu-probe` 的 `L0/L1`(dump level)是两回事,别混。** 配置相反不能同次跑:dump 要裁 response 单请求对齐,收敛要正常吞吐。

| 验证 | mode | 关键配置 | 看什么 |
|---|---|---|---|
| **前向一致** | `--hcu-probe L0` | `PROMPTS_ONLY=1`(裁 response)+ `max_num_seqs=1` + `enable_prefix_caching=False` + `enforce_eager=True` | dump 落盘后跑 `verify_consistency.py` |
| **端到端收敛** | `--hcu-probe off`(或 `run_l4.sh`) | `max_num_seqs=256` + 正常 prefix caching / chunked | reward 曲线 + `ppo_kl`(≈0 即训推一致) |
| **训练三通道** | `--hcuprobe monitor` | `MONITOR_CONFIG` 设(正常训练不裁) | **`monitor/rank*/` csv**(8rank×3通道;marker 常不传回 stdout,见上,以 csv 为准)✅ |

```bash
RUN=/data/hcu_probe_verl
# 前向一致(L0 dump):部署推理钩子 → 清盘 → 后台启动 → 验证
docker exec <container> bash $RUN/manage.sh infer-deploy
docker exec <container> bash $RUN/manage.sh clean
docker exec -d <container> bash $RUN/manage.sh start --hcu-probe L0   # max_num_seqs=1 串行 ~7min
docker exec <container> python3 $RUN/verify_consistency.py

# 端到端收敛(关 dump):正常 GRPO，读 tensorboard（verl actor/* 走 stdout 被 buffer，须读 event）
docker exec -d <container> bash $RUN/run_l4.sh
docker exec <container> python3 $RUN/read_tb.py

# 训练三通道 monitor:后台启动 → 等到训练 step 1 → 看 csv 产物(marker 常不传回 stdout,以 csv 为准)
docker exec -d <container> bash $RUN/manage.sh start --hcuprobe monitor
docker exec <container> find $RUN/verl_run/monitor -name '*.csv' | wc -l   # 期望 8rank×3通道×step数(实测 step0 = 24)
docker exec <container> bash -c "chmod -R 755 $RUN/verl_run/monitor; ls $RUN/verl_run/monitor/rank_0/"
```

已验证结果(L0 dump + 收敛):前向 **1% 阈值 80/80**(最差 0.657%、中位 0.399%)、id 配对训推同源 **5120/5120**;端到端 **49 step,reward 0→0.3 单调爬升、`ppo_kl≈0`**(ratio≈1)。**monitor + combined**:单节点 8 卡实测通过(monitor 3 通道 csv 落盘;combined dump_actor + monitor csv 叠加同时产出)——集群暴露并修复 `output_dir` makedirs 坑。**L1**:verl 上**未实测**(`--hcu-probe L1` 派生 config 的机制就绪、但没在 0.7 实跑;且 L1 训推 compare 本就不可行,见 §C。理论上 verl 的 HF Qwen2MLP 直接 `act_fn(x)`、无 megatron mcore 的 `is F.silu` identity 检查,应无需 patch `support_wrap_ops.yaml`——但未实跑坐实)。**tensor**:verl 上**未系统实测**(逐元素落盘机制就绪;机制上推理侧逐元素 dump **极慢/易爆盘**,钩子在时 rollout 会被拖住,`infer-unpatch` 关推理钩子才快)。

---

## 四、坑速查

华为没明说、实跑才暴露的坑:

1. **request_id 落点不是 `llm_server`**(0.7.1 已删),是 `agent_loop.AsyncLLMServerManager.generate`——入参 `request_id` 只做 sticky-session 路由,:158 内部 `uuid4().hex` 另生成新 id 传 vllm。必须在**这层**注入 `vllm_request_id` 并写 `output.extra_fields["request_id"]`,否则两侧各 5120 个 id 但零重叠、配对为 0。
2. **`dp_actor.data.select` 会滤掉 request_id**——默认只保留少数键,必须把 `request_id` 加进 `non_tensor_batch_keys`,否则训练侧 `update_actor_log.jsonl` 恒空。
3. **das vllm 给 request_id 加 `-<8hex>` 后缀**——推理侧 dispatch_log 记 `<32hex>-<8hex>`,compare 配对前 strip 后缀才与训练侧 32hex 对齐。
4. **dump.json 易失**——训练 step 推进时 hcu_probe 会清/覆盖旧 step 的 dump.json。`verify` 必须在 dump 落盘后、step 推进前跑;要留档用 `scripts/freeze_dump.sh`(后台等 `update_actor_log.jsonl` 在共享盘**非空**再 pkill→cp,`pkill -9` 强杀会让 append fp 未 close、共享盘拿到 0 字节→配对 0)。
5. **advantage 空保护**——`PROMPTS_ONLY` 裁掉 response → response_mask 空 → advantage 空 → `metric_utils.py` 的 `torch.max(空)` 崩。已加最小保护(valid_adv `numel()==0` 时 `return {}`),不动裁 response 架构。
6. **`enable_prefix_caching=False` 默认传不进 vllm**(真因不是 das 强制)——verl `build_cli_args_from_config`(`vllm_rollout/utils.py:263`)对 bool False 直接跳过,`--no-enable-prefix-caching` 从未进 `vllm serve` CLI。修复:`vllm_async_server.py` 在 `build_cli_args_from_config` 后补 `--no-enable-prefix-caching`(**别顺手强关 `enable_chunked_prefill`**,会触发 vllm SchedulerConfig 校验失败 `max_num_batched_tokens<max_model_len`)。

---

## 范例 + 延伸

- 可跑脚本均在本 skill `scripts/`(dispatch_logger / verify_consistency / find_compare_pairs / freeze_dump / read_tb / manage / patch_hcu_dump / patch_rocm_visibility / run_0.5b / run_l4),配置在 `config/`,patch 在 `patches/`,compare 原理深挖在 `references/compare-blindspots.md`。
- 完整原始底稿(环境/逐条源码/诚实复盘)为内部记录,对外仓可不带。
