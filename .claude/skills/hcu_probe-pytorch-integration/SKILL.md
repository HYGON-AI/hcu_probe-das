---
name: hcu_probe-pytorch-integration
description: 在原生 PyTorch / 自定义 train loop(无 Megatron/transformers/trl 专属胶水)接入 hcu_probe 做精度 dump + monitor 三通道监控,直接调 hcuprobe.integrations._common.lifecycle 的 5 个 hook,几行代码即可。当用户问"如何接入 hcu_probe 到原生 PyTorch / 自定义训练循环 / 自己写的 train loop / 怎么用 lifecycle hook / get_dump_lifecycle / get_monitor_lifecycle / PyTorch dump monitor / 没有 Feature 机制的框架怎么接" 时触发。
---

> ✅ **验证状态(2026-10,hcu_probe-das 26.2.0,BW 单卡,torch 2.10)**:`scripts/train_resnet_hcu_probe.py`(ResNet50 假数据 5 步)baseline / `--dump-cfg`(L0,step0~2 各 373 条)/ `--monitor-cfg`(3 通道 × 5 步 = 15 个 CSV)/ 两者同时开,均正常运行、无报错,日志出现 `PrecisionDebugger initialized` 与 `TrainerMonitorV2 initialized and started`。import 为 `hcuprobe.*`(whl 文件名 `hcu_probe-*.whl`)。

# hcu_probe-pytorch-integration(接入教学)

在**原生 PyTorch / 任意自定义 train loop**(没有 Megatron 那种 Feature 机制、没有 transformers Trainer callback)上接入 hcu_probe:**直接调 `hcuprobe.integrations._common.lifecycle` 的 5 个 hook,在你的训练循环里加几行即可。**

> **前置定位**:面向**训练已跑通的人**(环境/模型/数据/HCU 就绪),只讲 hcu_probe 接入,不含环境搭建。
>
> **本 skill 自包含**:完整接入范本 `scripts/train_resnet_hcu_probe.py`(纯 PyTorch + ResNet50,~120 行,照着改你的训练脚本);cfg 模板在 `config/`;坑 + 验证在 `references/runbook.md`。hcu_probe whl 本身带 `_common.lifecycle` 框架无关 API,直接 import 调用。

---

## 接入概述

PyTorch 这条路**比 megatron 更直接** —— 没有 Feature/wrapper/注册,就是在你训练脚本里 import lifecycle、按训练阶段调对应 hook:

| 阶段 | dump | monitor |
|---|---|---|
| 初始化(建完 model/optimizer) | `dump_lc.on_train_start(dump_cfg_path, seed, loss_output_dir)` | `mon_lc.on_setup_done(models, optimizer, monitor_cfg_path, grad_acc_steps, loss_output_dir)` |
| 每 step forward 前 | `dump_lc.on_step_start(model=model)` | — |
| 每 step 算完 loss/grad 后 | `dump_lc.on_step_end(step, loss_dict, grad_norm)` | `mon_lc.on_step_end(step, loss_dict, grad_norm)` |
| 训练结束 | `dump_lc.on_train_end()` | `mon_lc.on_train_end()` |

`dump_lc` = `get_dump_lifecycle()`、`mon_lc` = `get_monitor_lifecycle()`(都从 `hcuprobe.integrations._common.lifecycle` 拿,单例)。**不传 cfg = 不启,原训练**。

---

## Step 1: 把 5 个 hook 调用插进你的训练循环

完整范本 `scripts/train_resnet_hcu_probe.py`,照它改你的脚本。关键 6 处(从范本提炼):

```python
from hcuprobe.integrations._common.lifecycle import get_dump_lifecycle, get_monitor_lifecycle
from hcuprobe.integrations._common.config import resolve_loss_output_dir

dump_lc = get_dump_lifecycle()
mon_lc  = get_monitor_lifecycle()
loss_output_dir = resolve_loss_output_dir(dump_cfg_path=args.dump_cfg, monitor_cfg_path=args.monitor_cfg)

# ① 初始化后(model/optimizer 都建好了)
if args.dump_cfg:
    dump_lc.on_train_start(dump_cfg_path=args.dump_cfg, seed=args.seed, loss_output_dir=loss_output_dir)
if args.monitor_cfg:
    mon_lc.on_setup_done(models=model, optimizer=optimizer, monitor_cfg_path=args.monitor_cfg,
                         grad_acc_steps=1, loss_output_dir=loss_output_dir)

for step in range(steps):
    if args.dump_cfg:
        dump_lc.on_step_start(model=model)        # ② forward 前

    optimizer.zero_grad()
    out  = model(inputs)
    loss = criterion(out, labels)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # 只读 grad_norm 不裁
    optimizer.step()

    if args.dump_cfg:                              # ③ step 末
        dump_lc.on_step_end(step=step, loss_dict={"loss": float(loss.item())}, grad_norm=float(grad_norm))
    if args.monitor_cfg:                           # ④ step 末
        mon_lc.on_step_end(step=step, loss_dict={"loss": float(loss.item())}, grad_norm=float(grad_norm))

# ⑤ 收尾(monitor 先停 → dump 后停;LossTracker 在 on_train_end 内自动 close)
if args.monitor_cfg:
    mon_lc.on_train_end()
if args.dump_cfg:
    dump_lc.on_train_end()
```

**就这些** —— 不需要写 Feature 类、不需要注册、不改任何框架本体。`on_*` 的业务逻辑(`PrecisionDebugger` / `TrainerMonitorV2` init、rank 过滤、cfg 解析)全在 whl 的 lifecycle 里。

---

## Step 2: cfg 模板(在 `config/`)

| 文件 | 用途 |
|---|---|
| `dump_config_resnet_L0.json` | L0 module 级 statistics dump(改 `level` 为 `L1` 即 API 级)|
| `monitor_v2_config_resnet.json` | monitor 三通道(weight_grad/optimizer/param)|

改 `dump_path` / `output_dir` 为你要落盘的目录即用。约束(同所有框架):monitor cfg `patch_optimizer_step: false`、不开 `module` 通道;`optimizer` 通道要 Adam 这类有 state 的优化器才采得到(见 [`references/runbook.md`](references/runbook.md) 坑 3/4)。

---

## Step 3: 跑

训练脚本接好后,用 flag 控制启哪个(不传 = 原训练):

```bash
# baseline:不传 cfg
python train.py

# 只 dump
python train.py --dump-cfg config/dump_config_resnet_L0.json

# 只 monitor
python train.py --monitor-cfg config/monitor_v2_config_resnet.json

# 两个都开
python train.py --dump-cfg config/dump_config_resnet_L0.json --monitor-cfg config/monitor_v2_config_resnet.json
```

(flag 名是你训练脚本里 argparse 自定义的,范本用 `--dump-cfg` / `--monitor-cfg`;PyTorch 直接 argparse,不像 megatron 要穿启动脚本多层。)

---

## Step 4: 确认真启动 + 验产物

**⚠️ 只看训练 loss 正常 ≠ 采集成功**(缺 tensorboard 会静默 init 失败、只落 loss_tracking —— [`references/runbook.md`](references/runbook.md) 坑 1)。grep stdout 看 marker:

```
PrecisionDebugger initialized with <cfg>          ← dump 真起
TrainerMonitorV2 initialized and started          ← monitor 真起
```

产物:monitor csv 在 `output_dir/rank_0/{weight_grad,optimizer,param}_*.csv`;dump 在 `dump_path/step*/dump.json`;`loss_log.csv`。默认 `0o750`,host 读不到时容器内 `chmod -R 755`。

---

## Step 5: 采集之后 —— 离线分析(compare / 精度预检)

本 skill 只讲**采集**(dump + monitor)。dump 落盘后,用 hcu_probe 自带的**离线分析工具**做精度比对与预检——这些工具在 whl 里、不在本 skill,完整用法见主仓 `docs/offline_analysis/`:

- **两端精度比对**:`hcuprobe compare -tp <目标 dump 的 stepN 目录> -gp <golden dump 的 stepN 目录> -o <out>`,产出 `compare_result_*.xlsx`。**需要两份 dump**——同一段前向单跑只产一份,要比就用同一随机种子在两个 device(如 HCU 与 CPU,`HIP_VISIBLE_DEVICES=` 置空即回落 CPU)各 dump 一次再比。详见 `docs/offline_analysis/compare.md`。
- **API 精度预检**:`hcuprobe acc_check -api_info <L1 dump.json> -o <out>`(或 `multi_acc_check -n <进程> -d <卡号>`),逐算子在 HCU 与 CPU fp64 双跑比对,判 Pass/Warning/Error。**输入是 L1 dump**——把上面 dump cfg 的 `level` 改成 `L1`,跑时 `export TORCHDYNAMO_DISABLE=1`。详见 `docs/offline_analysis/precheck.md`。

> 一句话:采集在本 skill,compare / 精度预检 / overflow / graph_visualize 等在 `docs/offline_analysis/`。

---

## Quick Reference

| 项 | 内容 |
|---|---|
| 接入范本 | `scripts/train_resnet_hcu_probe.py`(纯 PyTorch + ResNet50,照改你的脚本)|
| 核心 API | `hcuprobe.integrations._common.lifecycle.{get_dump_lifecycle, get_monitor_lifecycle}` |
| 5 hook | `dump.on_train_start/on_step_start/on_step_end/on_train_end` + `monitor.on_setup_done/on_step_end/on_train_end` |
| cfg 模板 | 本 skill `config/`(L0 dump / monitor 三通道)|
| 坑 + 验证 + 实测状态 | [`references/runbook.md`](references/runbook.md) |

姊妹 skill:`hcu_probe-megatron-integration`(hcu_megatron Feature 机制)、verl 训推一致性(`hcu_probe-verl-fsdp-0.7-integration` / `hcu_probe-verl-fsdp-0.8-integration` / `hcu_probe-verl-megatron-0.8-integration`)。
