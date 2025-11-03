# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from msprobe.pytorch import TrainerMon
from msprobe.pytorch.common import seed_all
from msprobe.pytorch.hook_module.api_register import get_api_register

get_api_register().restore_all_api()

device = torch.device('cpu')
dtype_float32 = torch.float32
seed_all(mode=True)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(784, 10, dtype=dtype_float32)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x).type(dtype_float32))


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn(16, 784, dtype=dtype_float32, requires_grad=True)
        self.labels = torch.randint(low=0, high=9, size=(16,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx].to(device), self.labels[idx].to(device)


def monitor_demo(config: str = "./config/monitor_config.json"):
    net = Model().to(device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    hooker = TrainerMon(
        config,
        params_have_main_grad=False
    )
    hooker.set_monitor(
        model=net,
        grad_acc_steps=1,
        optimizer=optimizer
    )

    train_ds = ToyDataset()
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=10)

    for (inputs, labels) in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()

    hooker.summary_writer.close()
