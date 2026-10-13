# hcu_megatron 接入运行手册(坑 + 验证 + 实测)

> 本文是 SKILL.md 的深挖层 —— 接通(写 cfg + launcher 加 flag)不读也能跑;**排查报错 / 理解约束 / 确认验证状态**时读。
> 与 SKILL.md 正文同源,此处沉淀实跑暴露的坑和验证方法。

---

## 一、6 个核心坑(实跑暴露,新环境验证前必看)

### 坑 1 — tensorboard 隐性依赖(最迷惑)
hcu_probe 的 **pytorch 通用集成路径**(monitor v2 全部 + 通用 `hcu_probe.dump`)**import tensorboard**;缺了 → `No module named 'tensorboard'` → PrecisionDebugger / monitor init **静默失败**,只落 loss_tracking、没有 dump/csv(训练 loss 仍正常 → **假象**)。
- ⚠️ **megatron 集成路径(L0 dump)不依赖 tensorboard** → 会出现"L0 dump 成功但 monitor 失败"的迷惑现象。
- 容器无外网:从同 image 已装 tensorboard 的容器 `docker cp`(tensorboard + tensorboard_data_server + absl + markdown + werkzeug + dist-info,ABI 同 Python 3.10 兼容)。

### 坑 2 — L1 dump 的 `support_wrap_ops.yaml` patch(见 SKILL Step 4)
L1 wrap `F.gelu`/`F.silu` → Megatron MLP `activation_func is F.silu` identity 检查失败 → `ValueError: Only support fusion of gelu and swiglu`。必须注释 yaml 顶层 `- gelu`/`- silu`。**whl 不含此 patch**,每次重装 whl / 新容器都要手动 patch。**仅 L1 受影响**。

### 坑 3 — L1 的 `distributed:` 段(EP/MoE 才需)
完整 L1 patch 还有 `distributed:` 段(send/recv/all_reduce/all_gather/reduce_scatter/all_to_all/batch_isend_irecv 等 18 个通信算子)。
- **EP/MoE 模型(如 mixtral)必须连这段一起注释** —— EP 的 all_to_all 时序敏感,L1 wrap 会破坏。
- **llama2-7B(TP1/PP2 无 EP)实测**:注释 distributed 与否,L1 逐 iter 数值字字一致,PP 的 batch_isend_irecv 被 wrap 也不破坏。→ **llama 只 gelu/silu 够,MoE/EP 才加 distributed**。

### 坑 4 — `risk_level` 字段大小写 + 只 L1 生效
- `"risk_level"`(若用)必须**大写** `ALL` / `FOCUS` / `CORE`,小写 init failed。
- `risk_level` 只在 **L1** cfg 生效,L0 下被忽略。

### 坑 5 — monitor cfg 必设 `patch_optimizer_step: false`
Megatron nested optimizer(DistributedOptimizer → ChainedOptimizer)让 upstream 的自动 patch 扑空 → 必须 false,由 wrapper 在 `on_step_end` 手动驱动 `step()`。

### 坑 6 — monitor cfg 不开 `monitors.module` 通道
module 通道与 Megatron pipeline `deallocate_pipeline_outputs` 冲突。三通道只开 weight_grad / optimizer / param。

> 另:L1 cfg 启动时 launcher 自动 `export TORCHDYNAMO_DISABLE=1`(L1 monkey-patch 与 dynamo 冲突)。

---

## 二、验证三件套(训练前 / 中 / 后)

**训练前**:① 清残留 dump/ckpt;② 若跑 monitor 确认 tensorboard 在(坑 1);③ 若跑 L1 确认 yaml 已 patch(坑 2)。

**训练中**:grep log 确认业务**真启动**(只看 wrapper applied 不够):
```
[hcu_probe-wrapper] train_step() wrapper applied via Feature mechanism   ← wrapper 装饰
[hcu_probe.dump] PrecisionDebugger initialized with <cfg>                 ← dump 真启动
[hcu_probe.monitor] TrainerMonitorV2 initialized and started             ← monitor 真启动
msprobe ends successfully / [hcu_probe.* stopped]                        ← 收尾干净
```
⚠️ 必须看到 `initialized` / `started` 才算真启动(只 `wrapper applied` 是假象 —— 见坑 1)。

**训练后**:查产物存在 + 数量正确 + 内容合理(见 SKILL Step 5 产物表)。docker 内 `chmod -R 755` 让 host 可读(0o750)。

---

## 三、L0 vs L1 覆盖差异(Megatron + TE)

- **L0(module 级)**:抓每个 module 的 forward 统计量,够定位"哪层不一致",不依赖 tensorboard、不需 yaml patch —— **生产推荐**。
- **L1(API 级)**:抓到 Functional/Tensor/Torch 算子级,更细;但需 yaml patch(坑 2/3)+ `TORCHDYNAMO_DISABLE`。Megatron 用 TE(TransformerEngine)融合算子时,部分计算在 TE kernel 内、L1 也抓不到 Python 层 API。

---

## 四、实测状态(诚实标注)

- **llama2-7B(TP1/PP2,8 卡)5 模式 PoC**:baseline / L0 / L1 / monitor / combined 均在 HCU 集群实跑通过;L1 注释 distributed 与否逐 iter 数值字字一致(坑 3 的依据)。
- **MoE/EP 模型**:坑 3 的 distributed 段约束是机制推断 + EP 时序原理,**未在 MoE 上逐项实测**。
- 具体实测数据表(loss 数值、各模式产物计数)在内部验证记录,非本 skill 范围;本 skill 给方法和坑,数据按你自己的 run 核对。

> 内部组件指纹类问题(特定集群环境相关)属环境特化,与接入无关,本 skill 不展开。
