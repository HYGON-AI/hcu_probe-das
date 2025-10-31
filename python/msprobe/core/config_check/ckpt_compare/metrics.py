# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

import numpy as np

from msprobe.core.common.log import logger
from msprobe.core.compare.npy_compare import CompareOps



def in_different_shape(a, b):
    if a.shape != b.shape:
        logger.warning(f"a, b are in different shape. a: {a.shape}, b: {b.shape}")
        return True
    return False


def l2_distance(a, b):
    if a is None or b is None:
        return None
    if in_different_shape(a, b):
        return None
    return np.linalg.norm(a - b).item()


def cos_sim(a, b):
    if a is None or b is None:
        return None

    if in_different_shape(a, b):
        return None
    if a.ndim > 0:
        a = a.flatten().squeeze()
        b = b.flatten().squeeze()

    num = a.dot(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 and b_norm == 0:
        return 1.
    if a_norm == 0 or b_norm == 0:
        logger.warning(f'One tensor norm is zero.')
        return None

    sim = num / (a_norm * b_norm)

    return sim.item()


def numel(a, b):
    n1 = a.size
    n2 = b.size
    if n1 != n2:
        logger.warning('parameters have different number of element')
        return (n1, n2)
    return n1


def shape(a, b):
    if in_different_shape(a, b):
        return [list(a.shape), list(b.shape)]
    return list(a.shape)


METRIC_FUNC = {
    'l2': l2_distance, 
    'cos': cos_sim, 
    'numel': numel, 
    'shape': shape
    }
