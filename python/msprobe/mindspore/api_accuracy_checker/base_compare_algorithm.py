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

from abc import ABC, abstractmethod

import mindspore
import numpy as np
import torch
from msprobe.core.common.const import CompareConst
from msprobe.core.common.exceptions import ApiAccuracyCheckerException
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.common.const import MsCompareConst


class CompareResult:
    def __init__(self, compare_value, pass_status, err_msg):
        self.compare_value = compare_value
        self.pass_status = pass_status
        self.err_msg = err_msg


class BaseCompareAlgorithm(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.compare_algorithm_name = None
        self.err_msg_mapping = {
            CompareConst.COSINE: {
                CompareConst.PASS: "",
                CompareConst.ERROR: f"cosine similarity is less than threshold: {CompareConst.COS_THRESHOLD} ",
                CompareConst.SKIP: "two inputs are not valid for computing cosine similarity, skip comparing ",
            },
            CompareConst.MAX_ABS_ERR: {
                CompareConst.PASS: "",
                CompareConst.ERROR: "max absolute difference is greater than " \
                                    f"threshold: {CompareConst.MAX_ABS_ERR_THRESHOLD} ",
                CompareConst.SKIP: "two inputs are not valid for computing max absolute difference, skip comparing ",
            },
            CompareConst.MAX_RELATIVE_ERR: {
                CompareConst.PASS: "",
                CompareConst.ERROR: "",
                CompareConst.SKIP: "",
            },
        }

    def __call__(self, bench_compute_element, tested_compute_element):
        '''
        Args:
            bench_compute_element: ComputeElement
            tested_compute_element: ComputeElement

        Return:
            compare_result: CompareResult
        '''
        if self.check_validity(bench_compute_element, tested_compute_element):
            compare_value = self.run_compare(bench_compute_element, tested_compute_element)
            pass_status = self.check_pass(compare_value)
        else:
            logger.warning(f"not suitable for computing {self.compare_algorithm_name}, skip this.")
            compare_value = None
            pass_status = CompareConst.SKIP

        err_msg = self.err_msg_mapping.get(self.compare_algorithm_name).get(pass_status)

        compare_result = CompareResult(compare_value, pass_status, err_msg)
        return compare_result

    @staticmethod
    def convert_to_np_float64_ndarray(tensor):
        if isinstance(tensor, mindspore.Tensor):
            ndarray = tensor.astype(mindspore.float64).numpy()
        elif isinstance(tensor, torch.Tensor):
            ndarray = tensor.to(torch.float64, copy=True).numpy()
        else:
            err_msg = "BaseCompareAlgorithm.convert_to_np_float64_ndarray failed: " \
                      "input is not mindspore.Tensor or torch.Tensor"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))
        return ndarray

    @staticmethod
    def check_two_tensor(bench_compute_element, tested_compute_element):
        bench_parameter = bench_compute_element.get_parameter()
        tested_parameter = tested_compute_element.get_parameter()

        bench_is_tensor = isinstance(bench_parameter, (mindspore.Tensor, torch.Tensor))
        tested_is_tensor = isinstance(tested_parameter, (mindspore.Tensor, torch.Tensor))
        shape_same = bench_compute_element.get_shape() == tested_compute_element.get_shape()
        return bench_is_tensor and tested_is_tensor and shape_same

    @abstractmethod
    def check_validity(self, bench_compute_element, tested_compute_element):
        '''
        Args:
            bench_compute_element: ComputeElement
            tested_compute_element: ComputeElement

        Return:
            check_res: boolean
        '''
        raise NotImplementedError

    @abstractmethod
    def run_compare(self, bench_compute_element, tested_compute_element):
        '''
        Args:
            bench_compute_element: ComputeElement
            tested_compute_element: ComputeElement

        Return:
            compare_value: float/int
        '''
        raise NotImplementedError

    @abstractmethod
    def check_pass(self, compare_value):
        '''
        Args:
            compare_value: float/int

        Return:
            pass_status: str
        '''
        raise NotImplementedError


class CosineSimilarityCompareAlgorithm(BaseCompareAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.compare_algorithm_name = CompareConst.COSINE

    def check_validity(self, bench_compute_element, tested_compute_element):
        return self.check_two_tensor(bench_compute_element, tested_compute_element)

    def run_compare(self, bench_compute_element, tested_compute_element):
        bench_ndarray = self.convert_to_np_float64_ndarray(bench_compute_element.get_parameter())
        tested_ndarray = self.convert_to_np_float64_ndarray(tested_compute_element.get_parameter())

        bench_norm = np.linalg.norm(bench_ndarray)
        tested_norm = np.linalg.norm(tested_ndarray)
        dot_product = np.dot(bench_ndarray.flatten(), tested_ndarray.flatten())
        cosine_similarity = (MsCompareConst.EPSILON + dot_product) / (MsCompareConst.EPSILON + bench_norm * tested_norm)
        return cosine_similarity

    def check_pass(self, compare_value):
        if compare_value > CompareConst.COS_THRESHOLD:
            return CompareConst.PASS
        else:
            return CompareConst.ERROR


class MaxAbsoluteDiffCompareAlgorithm(BaseCompareAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.compare_algorithm_name = CompareConst.MAX_ABS_ERR

    def check_validity(self, bench_compute_element, tested_compute_element):
        return self.check_two_tensor(bench_compute_element, tested_compute_element)

    def run_compare(self, bench_compute_element, tested_compute_element):
        bench_ndarray = self.convert_to_np_float64_ndarray(bench_compute_element.get_parameter())
        tested_ndarray = self.convert_to_np_float64_ndarray(tested_compute_element.get_parameter())

        max_absolute_diff = np.max(np.abs(bench_ndarray - tested_ndarray))
        return max_absolute_diff

    def check_pass(self, compare_value):
        if compare_value < CompareConst.MAX_ABS_ERR_THRESHOLD:
            return CompareConst.PASS
        else:
            return CompareConst.ERROR


class MaxRelativeDiffCompareAlgorithm(BaseCompareAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.compare_algorithm_name = CompareConst.MAX_RELATIVE_ERR

    def check_validity(self, bench_compute_element, tested_compute_element):
        return self.check_two_tensor(bench_compute_element, tested_compute_element)

    def run_compare(self, bench_compute_element, tested_compute_element):
        bench_ndarray = self.convert_to_np_float64_ndarray(bench_compute_element.get_parameter())
        tested_ndarray = self.convert_to_np_float64_ndarray(tested_compute_element.get_parameter())

        abs_diff = np.abs(bench_ndarray - tested_ndarray)
        bench_ndarray_nonzero = np.abs(bench_ndarray) + (bench_ndarray == 0) * MsCompareConst.EPSILON
        max_relative_diff = np.max(abs_diff / bench_ndarray_nonzero)
        return max_relative_diff

    def check_pass(self, compare_value):
        if compare_value < CompareConst.MAX_RELATIVE_ERR_THRESHOLD:
            return CompareConst.PASS
        else:
            return CompareConst.ERROR


compare_algorithms = {
    CompareConst.COSINE: CosineSimilarityCompareAlgorithm(),
    CompareConst.MAX_ABS_ERR: MaxAbsoluteDiffCompareAlgorithm(),
    CompareConst.MAX_RELATIVE_ERR: MaxRelativeDiffCompareAlgorithm(),
}
