import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from hcuprobe.pytorch import PrecisionDebugger, seed_all

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
    # 1. 固定随机种子
    seed_all()
    device = torch.device("cuda")

    # 2. 构造固定数据
    N, C, H, W = 4, 8, 32, 32
    x_data = torch.randn(12, C, H, W)
    y_data = torch.randn(12, C, H * 2, W)

    # 3. 构造数据
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=N, shuffle=False)

    # 4. 创建模型
    model = ModuleOP().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 5. 创建 hcuprobe
    os.chdir(Path(__file__).resolve().parent)
    debugger = PrecisionDebugger(config_path=str(Path(__file__).resolve().parent / "config.json"))

    # 6. 训练
    model.train()
    for step, (x, y) in enumerate(dataloader):
        x = x.to(device).contiguous(memory_format=torch.channels_last)
        y = y.to(device).contiguous(memory_format=torch.channels_last)
        debugger.start(model=model)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        debugger.stop()
        debugger.step()
    
if __name__ == "__main__":
    main()