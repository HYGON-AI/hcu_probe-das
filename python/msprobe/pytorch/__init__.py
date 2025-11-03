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
from .compare.distributed_compare import compare_distributed
from .compare.pt_compare import compare
from .common.utils import seed_all
torch_version_above_or_equal_2 = torch.__version__.split('+')[0] >= '2.0'
if torch_version_above_or_equal_2:
    from msprobe.pytorch.monitor.module_hook import TrainerMon
