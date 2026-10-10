# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MulanPSL-2.0

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from hcuprobe.pytorch import PrecisionDebugger, seed_all

class ModuleOP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=8, out_features=4)

    def forward(self, x):
        x1 = self.linear(x)
        return F.relu(x1)

if __name__ == "__main__":
    # 设置随机数种子
    seed_all()
    # 实例化
    os.chdir(Path(__file__).resolve().parent)
    debugger = PrecisionDebugger(config_path=str(Path(__file__).resolve().parent / "config.json"))
    # 模型定义
    model = ModuleOP()
    # 开启数据采集
    debugger.start(model=model)
    x = torch.randn(10, 8)
    out = model(x)
    loss = out.sum()
    loss.backward()
    # 停止并结束当前 step 的数据采集
    debugger.stop()
    debugger.step()