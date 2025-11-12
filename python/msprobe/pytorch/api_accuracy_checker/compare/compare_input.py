#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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


class CompareInput:
    """
    A class to encapsulate the input data required for comparison operations.

    Attributes:
        bench_output (np.ndarray): The benchmark output values.
        device_output (np.ndarray): The device output values.
        compare_column (class): A clasee to store and update comparison metrics.
        dtype (type, optional): The data type of the outputs. Defaults to None.
        rel_err_orign (float or array-like, optional): The original relative error values. Defaults to None.

    Methods:
        __init__(bench_output, device_output, compare_column, dtype, rel_err_orign): 
        Initializes an instance of CompareInput.
    """
    def __init__(self, bench_output, device_output, compare_column, dtype=None, rel_err_orign=None):
        self.bench_output = bench_output
        self.device_output = device_output
        if not isinstance(bench_output, np.ndarray) or not isinstance(device_output, np.ndarray):
            raise TypeError("The input should be numpy array")
        self.compare_column = compare_column
        self.dtype = dtype
        self.rel_err_orign = rel_err_orign


class PrecisionCompareInput:
    def __init__(self, row_npu, row_gpu, dtype, compare_column):
        self.row_npu = row_npu
        self.row_gpu = row_gpu
        self.dtype = dtype
        self.compare_column = compare_column
