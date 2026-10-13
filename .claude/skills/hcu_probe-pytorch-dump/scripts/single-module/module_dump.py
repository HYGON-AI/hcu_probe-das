# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MulanPSL-2.0

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

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
    model = ModuleOP()
    x = torch.randn(10, 8)
    out = model(x)
    loss = out.sum()
    loss.backward()