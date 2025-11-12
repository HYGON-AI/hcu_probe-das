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

from msprobe.pytorch.api_accuracy_checker.compare.algorithm import compare_bool_tensor
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare


class BinaryCompare(BaseCompare):
    """
    Binary comparison class for comparing boolean tensors.

    This class is designed to compare the output of a binary operation between a benchmark and a device.
    It calculates the error rate of the comparison and provides a simple metric for assessing the accuracy.

    Attributes:
        bench_output (np.ndarray): The output from the benchmark.
        device_output (np.ndarray): The output from the device.
        compare_column (object): The column object to store comparison results.
        dtype (torch.dtype): The data type of the outputs.

    Methods:
        _compute_metrics(): Computes the comparison metrics, specifically the error rate.

    Note:
        This class assumes that the input data is an instance of InputData containing the benchmark output,
        device output, comparison column, and data type. The outputs are expected to be boolean tensors.

    See Also:
        BaseCompare: The base class for comparison classes.
        compare_bool_tensor: The function used to compare boolean tensors.
    """
    def __init__(self, input_data):
        super(BinaryCompare, self).__init__(input_data)

    def _pre_compare(self):
        pass

    def _compute_metrics(self):
        """
        Computes the error rate metric for the comparison between benchmark and device outputs.

        This method calculates the proportion of mismatches between the benchmark output and the device output.
        It uses the `compare_bool_tensor` function to compare the two tensors and extract the error rate.

        Returns:
        dict: A dictionary containing the computed error rate metric.
            The dictionary has the following key:
            - "error_rate": The proportion of mismatches between the benchmark and device outputs.
        """
        error_rate, _, _ = compare_bool_tensor(self.bench_output, self.device_output)

        return {
            "error_rate": error_rate
        }
