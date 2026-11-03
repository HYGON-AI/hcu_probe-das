"""ResNet50 + 纯 PyTorch + hcu_probe whl 通用模型 PoC.

证明:hcu_probe whl 在通用 CV 模型(ResNet50)+ 通用框架(原生 PyTorch)
+ 通用优化器(Adam)上可用。无 Megatron / 无 transformers / 无 trl /
无 DeepSpeed,只用 hcuprobe.integrations._common.lifecycle 框架
无关 keyword-only API。

接入方式见本 skill(hcu_probe-pytorch-integration)的 SKILL.md;
dump 落盘后的离线比对 / 精度预检见主仓 docs/offline_analysis/。

跑法:
    python train_resnet_hcu_probe.py \
        --monitor-cfg ./monitor_v2_config_resnet.json \
        [--dump-cfg ./dump_config_resnet_L0.json] \
        [--steps 5] [--batch-size 4]

期望产物:
    monitor cfg.output_dir/rank_0/{weight_grad,optimizer,param}_*.csv  ← 三通道 csv
    dump_cfg.dump_path/step{0..2}/dump.json                            ← 可选 L0 dump
    {dump|monitor}.output_dir/loss_log.csv                             ← LossTracker
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from hcuprobe.integrations._common.lifecycle import (
    get_dump_lifecycle,
    get_monitor_lifecycle,
)
from hcuprobe.integrations._common.config import resolve_loss_output_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--monitor-cfg", default=None, help="path to monitor v2 JSON")
    p.add_argument("--dump-cfg", default=None, help="path to L0/L1 dump JSON")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="resnet50",
                   choices=["resnet18", "resnet34", "resnet50", "resnet101"],
                   help="torchvision ResNet variant")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1) 设备 + 随机种子
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[resnet-poc] device = {device}, model = {args.model}")

    # ------------------------------------------------------------------
    # 2) 模型 + Adam optimizer + CrossEntropy loss
    #    - Adam 有 exp_avg / exp_avg_sq state,monitor.optimizer.mv_distribution 才能采到东西
    #    - 不加载预训练权重(weights=None),不依赖外网,不依赖磁盘 ckpt
    # ------------------------------------------------------------------
    model_ctor = getattr(models, args.model)
    model = model_ctor(weights=None).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 3) Fake 数据
    #    PoC 不要求收敛,仅证明 hook 能挂上 + csv 能写。固定输入避免抖动。
    # ------------------------------------------------------------------
    fake_input = torch.randn(args.batch_size, 3, 224, 224, device=device)
    fake_label = torch.randint(0, 1000, (args.batch_size,), device=device)

    # ------------------------------------------------------------------
    # 4) hcu_probe lifecycle init
    # ------------------------------------------------------------------
    dump_lc = get_dump_lifecycle()
    mon_lc = get_monitor_lifecycle()

    loss_output_dir = resolve_loss_output_dir(
        dump_cfg_path=args.dump_cfg,
        monitor_cfg_path=args.monitor_cfg,
    )

    if args.dump_cfg:
        dump_lc.on_train_start(
            dump_cfg_path=args.dump_cfg,
            seed=args.seed,
            loss_output_dir=loss_output_dir,
        )

    if args.monitor_cfg:
        mon_lc.on_setup_done(
            models=model,
            optimizer=optimizer,
            monitor_cfg_path=args.monitor_cfg,
            grad_acc_steps=1,
            loss_output_dir=loss_output_dir,
        )

    # ------------------------------------------------------------------
    # 5) 训练循环
    # ------------------------------------------------------------------
    model.train()
    for step in range(args.steps):
        if args.dump_cfg:
            dump_lc.on_step_start(model=model)

        optimizer.zero_grad()
        out = model(fake_input)
        loss = criterion(out, fake_label)
        loss.backward()

        # grad norm:不裁剪,只读数(传 inf 给 max_norm 让 clip_grad_norm_ 退化为度量)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        optimizer.step()

        loss_val = float(loss.item())
        gn_val = float(gn)
        print(f"[resnet-poc] step {step}: loss = {loss_val:.6f} | grad_norm = {gn_val:.4f}")

        if args.dump_cfg:
            dump_lc.on_step_end(
                step=step,
                loss_dict={"loss": loss_val},
                grad_norm=gn_val,
            )
        if args.monitor_cfg:
            mon_lc.on_step_end(
                step=step,
                loss_dict={"loss": loss_val},
                grad_norm=gn_val,
            )

    # ------------------------------------------------------------------
    # 6) cleanup(顺序:monitor 先停 → dump 后停;LossTracker 在两者 on_train_end 内自动 close)
    # ------------------------------------------------------------------
    if args.monitor_cfg:
        mon_lc.on_train_end()
    if args.dump_cfg:
        dump_lc.on_train_end()

    print("[resnet-poc] done.")


if __name__ == "__main__":
    main()
