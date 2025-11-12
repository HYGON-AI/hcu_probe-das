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

from msprobe.pytorch.api_accuracy_checker.compare.algorithm import check_inf_nan_value, check_norm_value, \
    check_small_value
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import StandardConfig
from msprobe.core.common.const import CompareConst



class AbsolutethdCompare(BaseCompare):
    """
    Absolute threshold compare class.

    This class is used to compare the absolute threshold of benchmark outputs and device outputs.
    It calculates various metrics such as inf_nan_error_ratio, rel_err_ratio, and abs_err_ratio
    to determine the accuracy of the device output compared to the benchmark output.

    Attributes:
        bench_output (np.ndarray): The output from the benchmark.
        device_output (np.ndarray): The output from the device.
        dtype (torch.dtype): The data type of the outputs.
        abs_bench (np.ndarray): The absolute value of the benchmark output.
        abs_bench_with_eps (np.ndarray): The absolute value of the benchmark output with epsilon.
        both_finite_mask (np.ndarray): A mask indicating where both outputs are finite.
        inf_nan_mask (np.ndarray): A mask indicating where either output is infinite or NaN.
        rtol (float): The relative tolerance for comparison.
        rel_err (np.ndarray): The relative error between the benchmark and device outputs.
        small_value (float): The small value threshold for comparison.
        small_value_atol (float): The absolute tolerance for small values.
        small_value_mask (np.ndarray): A mask indicating where values are small.
        normal_value_mask (np.ndarray): A mask indicating where values are normal.

    Methods:
        _get_rtol(): Gets the relative tolerance based on the data type.
        _get_rel_err(abs_bench_with_eps): Calculates the relative error.
        _get_normal_value_mask(small_value_mask): Gets the mask for normal values.
        _pre_compare(): Prepares the comparison by calculating various metrics.
        _compute_metrics(): Computes the comparison metrics.

    Note:
        This class assumes that the input data is a dictionary containing 'bench_output', 'device_output', 
        'compare_column' and 'dtype'.
        The 'dtype' should be a PyTorch data type.

    See Also:
        BaseCompare: The base class for comparison classes.
        StandardConfig: The class containing standard configuration values.
    """
    def __init__(self, input_data):
        super(AbsolutethdCompare, self).__init__(input_data)
        self.compare_algorithm = CompareConst.ABSOLUTE_THRESHOLD

    def _get_rtol(self):
        return StandardConfig.get_rtol(self.dtype)

    def _pre_compare(self):
        """
        Prepares the comparison by calculating various metrics.

        This method performs the following steps:
            1. Calculates the absolute benchmark values and their epsilon-adjusted versions.
            2. Determines masks for finite and infinite/NaN values in the outputs.
            3. Computes the absolute error between benchmark and device outputs.
            4. Retrieves the relative tolerance based on the data type.
            5. Calculates the relative error using the absolute error and epsilon-adjusted benchmark values.
            6. Determines the small value threshold and its absolute tolerance.
            7. Creates a mask for small values based on the benchmark values and finite mask.
            8. Creates a mask for normal values by excluding small values from the finite mask.
        """
        self.abs_bench, self.abs_bench_with_eps = self.stat_abs_bench_with_eps()
        self.both_finite_mask, self.inf_nan_mask = self.stat_finite_and_infinite_mask()
        self.abs_err = self.stat_abs_error()
        self.rtol = self._get_rtol()
        self.rel_err = self._get_rel_err(self.abs_err, self.abs_bench_with_eps)
        self.small_value, self.small_value_atol = self.get_small_value_threshold()
        self.small_value_mask = self.stat_small_value_mask(self.abs_bench, self.both_finite_mask, self.small_value)
        self.normal_value_mask = self._get_normal_value_mask(self.both_finite_mask, self.small_value_mask)

    def _compute_metrics(self):
        inf_nan_error_ratio = check_inf_nan_value(self.inf_nan_mask, self.bench_output, self.device_output, self.dtype,
                                                  self.rtol)
        rel_err_ratio = check_norm_value(self.normal_value_mask, self.rel_err, self.rtol)
        abs_err_ratio = check_small_value(self.abs_err, self.small_value_mask, self.small_value_atol)
        return {
            "inf_nan_error_ratio": inf_nan_error_ratio,
            "rel_err_ratio": rel_err_ratio,
            "abs_err_ratio": abs_err_ratio
        }
