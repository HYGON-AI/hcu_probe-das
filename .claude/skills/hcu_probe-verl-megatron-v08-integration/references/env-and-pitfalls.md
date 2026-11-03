# 环境前置 + 训练侧钩子代码 + 全部坑(verl 0.8.0 Megatron 后端)

> 本文是 SKILL.md 的详细支撑。环境只给指针(训练已跑通则跳过),重点是钩子三段式代码 + 全部踩过的坑。

## 一、环境前置(已跑通就跳过)

### 1.1 节点 / 镜像 / 容器

| 项 | 值(示例,按实际替换) |
|---|---|
| 节点 | 海光 BW 系列 HCU,8 卡单节点(若共享节点,card0 可能被他人占,`gpu_memory_utilization` 要调小) |
| 镜像 | `harbor.sourcefind.cn:5443/.../vllm:0.11.0-...-dtk25.04.2-...-das1.7-py3.10`(自带 vllm0.11 / torch2.5.1 / ray2.48 / megatron-core0.13.2 / apex / TE) |
| 容器 | `--privileged` + `--device=/dev/kfd --device=/dev/mkfd --device=/dev/dri` + `-v /opt/hyhal:/opt/hyhal:ro`(HCU 标准) |
| hcu_verl | 内网 GitLab,**只有 main**(无 release 分支/tag);纯代码库无模型 |
| Megatron-LM submodule | **必须 `git checkout core_v0.13.1`**(默认指向 0.17,与 verl 0.8.0 不配套) |

### 1.2 增量依赖
```
tensordict==0.10  mbridge==0.15.1  hydra-core  omegaconf  codetiming
mathruler  pylatexenc  hcu_probe  tensorboard  py-spy  modelscope
# compare 阶段额外: openpyxl(pt_compare 存 xlsx)
```
- `tensordict` **不要 `--no-deps`**(会漏 python-version 依赖)。
- `tensorboard` 是 verl trainer logger 隐性依赖,缺了到 `fit()` 才崩。

### 1.3 数据 / 模型
- gsm8k:`python examples/data_preprocess/gsm8k.py`。
- Qwen3-8B HF → mcore:`bash examples/scripts/converter_hf_to_mcore.sh`(~16G dist 权重)。转换器多进程 save 死锁 → `cp patch_init` 走 verl_adaptor `preload_tensors` 同步 patch。

### 1.4 launch 脚本(仿 0.7.1 run_0.5b,不用 mpirun,main_ppo local ray)
清残留 + ray hycu patch(pyamdsmi sed) + `cp patch_init` + das env + `bash run_model`。
sed `run_qwen3`:`train_cp=1` / `enable_sleep=False` / `gpu_memory_utilization=0.18`;`HIP_VISIBLE_DEVICES=0-7` + `GPUS_PER_NODE=8`;TP8 / gen_tp8 / PP1。

## 二、训练侧钩子三段式代码(`transformer_impl.py`)

均以 `DUMP_ON` env 总开关、懒加载、仅 actor engine 生效(ref 的 `forward_only` 跳过)。

**a. `MegatronEngine.__init__`(:100)— 占位**
```python
self._debugger = None
self._update_actor_logger_fp = None
self._should_dump = False
self.rank = torch.distributed.get_rank()
```
**b. `forward_backward_batch`(:662)— 懒初始化 + 门控**
```python
self._ensure_debugger()
dump_phase = os.environ.get("DUMP_PHASE", "log_prob")   # all | log_prob | update_actor
phase = "log_prob" if forward_only else "update_actor"
self._should_dump = self._debugger is not None and (dump_phase == "all" or dump_phase == phase)
```
**c. `_ensure_debugger`(:770)— 建 hcu_probe debugger**
```python
if self._debugger is not None or self.engine_config.forward_only:  # ref engine 跳
    return
if not int(os.environ.get("DUMP_ON", 0)):
    return
from hcuprobe.pytorch import PrecisionDebugger, seed_all   # 不是 msprobe!
seed_all(mode=True)
self._debugger = PrecisionDebugger(config_path=os.environ.get("DUMP_ACTOR_CONFIG", "/home/config_actor.json"))
# 在 dump_path/<pid>/ 开 update_actor_log.jsonl
```
**d. `_log_update_actor_step`(:792)— 提 request_id 写一行**
```python
req = tu.get(micro_batch, key="request_id", default=None)   # 从 micro_batch 取
record = {"source":"update_actor","rank":self.rank,
          "step":self._debugger.service.current_iter,
          "request_ids":[...], "num_requests":len(...)}      # 写 jsonl
```
**e. `MegatronEngineWithLMHead.forward_step`(:1016)— forward_fn 前后包**
```python
if self._should_dump:
    self._debugger.start(model=model)
output = forward_fn(model, input_ids, ...)   # 非 fused 路径,use_fused_kernels=False
if self._should_dump:
    if not self._debugger.service.should_stop_service:
        self._log_update_actor_step(batch)
    self._debugger.stop()
    self._debugger.step()
```

## 三、全部坑速查

### 环境 / 版本坑(10 个,搭环境时)
| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | `tensordict` 装不全 | `--no-deps` 漏 python-version 依赖 | 补装依赖 |
| 2 | `DTensorSpec` NameError | torch 2.5 下 future annotations 未启用,verl 函数签名注解无条件引用(仅 torch>=2.6 import) | `verl/utils/fsdp_utils.py` 头加 `from __future__ import annotations`(见 `04a` patch) |
| 3 | `No module named megatron.training` | Megatron-LM submodule 未 clone | clone submodule |
| 4 | `ModelType.encoder_and_decoder` API 不匹配 | submodule 是 0.17,verl 0.8.0 要 0.13 | `git checkout core_v0.13.1` |
| 5 | converter 转换卡死 | HF→mcore 多进程 save 死锁 | `cp patch_init` 走 verl_adaptor `preload_tensors` |
| 6 | `'_CudaDeviceProperties' object has no attribute 'uuid'` | das torch2.5.1 device_properties 无 uuid | `get_device_uuid` fallback device_id(见 `04b` patch) |
| 7 | `Sleep mode is not supported` | das vllm 不支持 sleep mode | `enable_sleep=False` |
| 8 | `No module named 'tensorboard'` | trainer logger 隐性依赖 | `pip install tensorboard` |
| 9 | vllm worker OOM(`Free memory < desired`) | 共享节点 card0 被占 | `gpu_memory_utilization=0.18` 挤(非连续卡会 HIP invalid ordinal,用连续 8 卡) |
| 10 | `No module named 'msprobe'` | 容器装 hcu_probe fork,无 msprobe 命名空间 | import 改 `from hcuprobe.pytorch` |

### 推理侧坑(4 个,接 dump 时)
| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | `No module named vllm_hcu` | das1.7 内联进上游 vllm 0.11,无独立 vllm_hcu 包 | patch 上游 `gpu_model_runner.py`,FSDP 锚点全重定位 |
| 2 | `cumsum_cuda_kernel does not have a deterministic implementation` | `seed_all(mode=True)` 设 `use_deterministic_algorithms`,vllm sampler `torch.cumsum` 无确定性实现,profile_run 崩 | 推理侧 `seed_all(mode=False)`(仍固定种子,不开 deterministic) |
| 3 | `model 必须是 nn.Module,当前 CUDAGraphWrapper` | vllm 0.11 `self.model` 被 cuda graph 包一层 | 循环解包 `.runnable`(`patch_gpu_dump.py` 已处理) |
| 4 | dump.json data 空(modules=0) | vllm 默认 torch.compile+cudagraph 绕过 python module forward hook | rollout `enforce_eager=True` → 命中,434 modules |

### compare 配置坑(见 compare-guide.md 详解)
- `num_requests` 不为 1:改错字段(用了 `ppo_micro_batch_size`)→ 应 `rollout.log_prob_micro_batch_size_per_gpu=1` + `use_dynamic_bsz=False`。
- output 被 SP 切成 1/TP:`actor.megatron.sequence_parallel=False` 不生效 → 应 `override_transformer_config.sequence_parallel=False`。
- compare module 全 match 不上:npu/bench 反了 → npu=训练 / bench=推理。
- compare 误差大:配对挑到被 chunk 的不完整 prefill → 挑 token 数完全匹配的样本。

### 推理 patch 锚点漂移定位(`patch_gpu_dump.py` 报 `ANCHOR_NOT_FOUND` 时)
两个锚点是 vllm 0.11.0 / das1.7 实测的,换 vllm/das 版本可能漂移。容器内定位新锚点:
- **INIT**(在 `GPUModelRunner.__init__` 末尾):`grep -n 'sampled_token_ids_pinned_cpu' /usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_model_runner.py`
- **FWD**(在 `execute_model` 的 `set_forward_context` 块内):`grep -n 'model_output = self.model' <同文件>`

把 `patch_gpu_dump.py` 顶部的 `INIT_ANCHOR` / `FWD_ANCHOR` 三引号块改成新版本对应代码(注入逻辑不动),先 `python3 patch_gpu_dump.py <源码副本>` 在副本上验通过,再打容器。

### 操作提示(通用)
- dump.json 容器 root 建、0750:宿主 `find` 看不到,compare/查产物一律 `docker exec`(root)。
- 多跳 ssh 进集群时注意 shell 转义(某些字符如中文括号会触发 bash syntax error)。

### ⚠️ 推理(vllm)在 ~step4 不稳(2026-06-20 重验实测,已知未解)
- **现象**:Qwen3-8B Hybrid 共卡,vllm 引擎跑到约 step4 必出事,且都卡在推理侧 `msprobe: debugger.start` dump 那一刻:
  - `gpu_memory_utilization=0.18`(原 launch 模板值)→ `vllm.v1.engine.exceptions.EngineDeadError`(worker 进程死,EngineCore `collective_rpc → worker_response_mq.dequeue` 超时)。
  - 调到 `0.3` → 不报 EngineDeadError,但变成**挂死**(vllm worker 进程消失、main 死等死锁、GPU HCU% 空转、日志数分钟不更新)。
- **影响**:**前 ~3 步的 dump 是好的**(一致性 99.x% 就用这些算),所以**做一致性比对只需跑出前几步 dump 即可,别期望长跑**。但谁照本 skill 想长训(非 dump 用途),会在 ~step4 卡住。
- **根因未深挖(待查)**:是 hcu_probe 推理 dump 触发、vllm 多步后累积状态、还是 das vllm bug,没定位。临时缓解=`gpu_memory_utilization` 拉高一点能把"崩"变"挂"(更易 kill 清理),但没真修。
- 多样本 compare 用 `scripts/run_megatron_compare_multi.py`(对每个 token 完全匹配样本各跑一次,严格排除整层/IdentityOp,逐样本+汇总)。注意 token 完全匹配的干净样本通常只 ~2 个。
