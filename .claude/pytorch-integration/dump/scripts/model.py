import torch
import torch.nn as nn
import torch.nn.functional as F

from hcuprobe.pytorch import PrecisionDebugger

# 定义网络
class ModuleOP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=8, out_features=4)
        self.linear_2 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        PrecisionDebugger.start()
        x1 = self.linear_1(x)
        PrecisionDebugger.stop()
        x2 = self.linear_2(x1)
        r1 = F.relu(x2)
        return r1