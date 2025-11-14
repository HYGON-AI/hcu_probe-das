# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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

from mindspore import Tensor
import torch


def create_msa_tensor(data, dtype=None):
    return Tensor(data, dtype)


tensor_tensor = torch.tensor
setattr(torch, 'tensor', create_msa_tensor)


def reset_torch_tensor():
    setattr(torch, 'tensor', tensor_tensor)
