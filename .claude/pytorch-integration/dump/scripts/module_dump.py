# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MulanPSL-2.0

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from hcuprobe.pytorch import PrecisionDebugger, module_dump, module_dump_end

class ModuleOP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=8, out_features=4)
        self.linear_2 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = self.linear_2(x1)
        r1 = F.relu(x2)
        return r1

if __name__ == "__main__":
    # 创建 hcuprobe
    os.chdir(Path(__file__).resolve().parent)
    debugger = PrecisionDebugger(config_path=str(Path(__file__).resolve().parent / "config.json"))
    debugger.start()
    # 模型定义
    model = ModuleOP()
    x = torch.randn(10, 8)
    # 开启模块级精度数据dump
    module_dump(model, "MyModuleOP")
    out = model(x)
    module_dump_end()
    loss = out.sum()
    loss.backward()
    # 停止并结束当前 step 的数据采集
    debugger.stop()
    debugger.step()