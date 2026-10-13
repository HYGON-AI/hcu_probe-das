# 原生 PyTorch 接入 hcu_probe 运行手册(坑 + 验证)

> 本文是 SKILL.md 的深挖层 —— 照 SKILL 接通即可,**排错 / 确认验证状态**时读。

---

## 坑(实跑暴露)

### 坑 1 — tensorboard 隐性依赖(最迷惑)
hcu_probe 的 **pytorch 通用集成路径**(`_common.lifecycle` 的 dump + monitor 全部)**import tensorboard**;缺了 → `No module named 'tensorboard'` → `PrecisionDebugger` / `TrainerMonitorV2` init **静默失败**,只落 `loss_tracking`、没有 dump.json / monitor csv(训练 loss 仍正常打印 → **假象**)。
- 修:`pip install tensorboard`(版本与你的 torch ABI 兼容即可)。容器无外网时从同 image 已装的容器 `docker cp`。
- ⚠️ 这正是为什么**只看训练 loss 正常 ≠ 采集成功**,必须看 Step 验证里的 marker / csv 产物。

### 坑 2 — monitor 的 `output_dir` / dump 的 `dump_path` 父目录要可写
cfg 里的 `output_dir`(monitor)/ `dump_path`(dump)是相对**进程 cwd** 的相对路径(模板里是 `./resnet_monitor_v2` 等)。确认从哪个目录启动、产物会落那;容器 root 建的产物默认 `0o750`,host 读不到要容器内 `chmod -R 755`。

### 坑 3 — monitor 三通道约束(同所有框架)
monitor cfg **必设** `"patch_optimizer_step": false`(由训练循环里手动调 `on_step_end` 驱动 step),**不开** `monitors.module` 通道(易抓到 wrapper);三通道 = `weight_grad` / `optimizer`(`mv_distribution`)/ `param`,`ops=[min,max,mean,norm,nans]`。

### 坑 4 — optimizer 要有 state 才采得到
monitor 的 `optimizer` 通道采 `mv_distribution`(Adam 的 `exp_avg`/`exp_avg_sq`)—— 用 SGD 无动量时这个通道采不到东西(不是 bug,是优化器本身没 state)。样例用 Adam。

---

## 验证

**训练中**:grep stdout 确认**真启动**(只看 loss 正常不够,见坑 1):
```
PrecisionDebugger initialized with <cfg>          ← dump 真起
TrainerMonitorV2 initialized and started          ← monitor 真起
```
**训练后**:查产物存在 + 数量。以 ResNet 样例(单卡 5 step、both 模式)为例:
- monitor csv:`output_dir/rank_0/{weight_grad,optimizer,param}_*.csv`,单 rank × 3 通道 × step 数
- dump:`dump_path/step{0,1,2}/dump.json`(cfg 的 `step:[0,1,2]`)
- LossTracker:`loss_log.csv`,行数 = step 数

> 产物默认 `0o750`,host 读不到时容器内 `chmod -R 755 <产物目录>`。

---

## 实测状态(诚实标注)

- **ResNet50 / Adam / 单卡 / 随机数据**:dump(L0)+ monitor 三通道 + LossTracker 均实跑落盘(随机数据 5 step loss 快速过拟合,仅证明 hook 挂上 + 产物能写,不验收敛)。
- L1 dump / 多卡 / 其他模型:`_common.lifecycle` 机制相同、未逐项实测,接的人按需开 + 自验。
