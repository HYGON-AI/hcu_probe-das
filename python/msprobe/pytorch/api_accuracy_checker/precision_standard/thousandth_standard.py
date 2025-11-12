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

from msprobe.pytorch.api_accuracy_checker.compare.algorithm import get_rel_err_ratio
from msprobe.core.common.const import CompareConst
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare


class ThousandthStdCompare(BaseCompare):
    """
    Thousandth standard comparison class for calculating accuracy metrics.
    
    A subclass of BaseCompare, specifically designed to compare the relative error 
    between benchmark and device outputs, focusing on errors within a thousandth (0.001) threshold.

    Attributes:
        rel_err_orign (float or array-like): The original relative error values to be compared.
        compare_column (object): An object to store and update comparison metrics.

    Methods:
        _compute_metrics(): Computes the relative error metrics, specifically the thousandth error ratio.
    """
    def __init__(self, input_data):
        super().__init__(input_data)
        self.rel_err_orign = input_data.rel_err_orign
        self.compare_column = input_data.compare_column

    def _pre_compare(self):
        pass

    def _compute_metrics(self):
        """
        Computes the relative error metrics for the comparison, specifically focusing on errors within a thousandth 
        (0.001) threshold.

        This method calculates the proportion of relative errors that are within the thousandth threshold.
        It uses the `get_rel_err_ratio` function to determine the ratio of relative errors that are less than or 
        equal to the
        specified threshold defined in `CompareConst.THOUSAND_RATIO_THRESHOLD`.

        Returns:
        dict: A dictionary containing the computed relative error metric.
            The dictionary has the following key:
            - 'rel_err_thousandth': The proportion of relative errors within the thousandth threshold.
        """
        rel_err_thousandth, _ = get_rel_err_ratio(self.rel_err_orign, CompareConst.THOUSAND_RATIO_THRESHOLD)

        return {
            'rel_err_thousandth': rel_err_thousandth
        }
