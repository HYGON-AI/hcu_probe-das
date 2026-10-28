# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MulanPSL-2.0

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

class ModuleOP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        _, _, H, W = x.shape
        x = F.upsample_nearest(x, size=(H * 2, W))
        x = self.linear(x.permute(0, 2, 3, 1))
        return x.permute(0, 3, 1, 2)

def main():


    device = torch.device("cuda")

    N, C, H, W = 4, 8, 32, 32
    x_data = torch.randn(12, C, H, W)
    y_data = torch.randn(12, C, H * 2, W)

    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=N, shuffle=False)

    model = ModuleOP().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for step, (x, y) in enumerate(dataloader):
        x = x.to(device).contiguous(memory_format=torch.channels_last)
        y = y.to(device).contiguous(memory_format=torch.channels_last)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()