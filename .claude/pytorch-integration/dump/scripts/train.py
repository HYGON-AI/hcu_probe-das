# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MulanPSL-2.0

import torch
import os
from pathlib import Path

from hcuprobe.pytorch import PrecisionDebugger
from model import ModuleOP

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    debugger = PrecisionDebugger(config_path=str(Path(__file__).resolve().parent / "config.json"))
    model = ModuleOP()
    x = torch.randn(10, 8)
    out = model(x)
    loss = out.sum()
    loss.backward()