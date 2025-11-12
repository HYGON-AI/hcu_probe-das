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

from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union
)
import os
import numpy as np
import mindspore
from mindspore import ops
from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import ApiAccuracyCheckerException
from msprobe.mindspore.api_accuracy_checker.compute_element import ComputeElement
from msprobe.mindspore.api_accuracy_checker.type_mapping import float_dtype_str_list, torch_dtype_to_dtype_str
from msprobe.mindspore.api_accuracy_checker.utils import convert_to_tuple
from msprobe.mindspore.api_accuracy_checker.bench_functions.fusion_operator import fusion
from msprobe.mindspore.common.const import MsCompareConst
from msprobe.mindspore.common.log import logger


from msprobe.mindspore.api_accuracy_checker import torch_mindtorch_importer

from msprobe.mindspore.api_accuracy_checker.torch_mindtorch_importer import mindtorch
from msprobe.mindspore.api_accuracy_checker.torch_mindtorch_importer import mindtorch_tensor
from msprobe.mindspore.api_accuracy_checker.torch_mindtorch_importer import mindtorch_func
from msprobe.mindspore.api_accuracy_checker.torch_mindtorch_importer import mindtorch_dist

if torch_mindtorch_importer.is_valid_pt_mt_env:
    from msprobe.mindspore.api_accuracy_checker.torch_mindtorch_importer import torch
else:
    import torch

# 为了可读性，我们先给每种返回形态起个别名
ForwardResult = Tuple[
    List[ComputeElement],
    Tuple[Any, ...],
    Dict[str, Any],
    Tuple[Any, ...],
]

BackwardResultMT = Tuple[
    List[ComputeElement],
    Union[Any, Tuple[Any, ...]],
    Tuple[Any, ...],
]

PyTorchBackward = List[ComputeElement]


class ApiInputAggregation:
    def __init__(self, inputs, kwargs, gradient_inputs) -> None:
        """
        Args:
            inputs: List[ComputeElement]
            kwargs: dict{str: ComputeElement}
            gradient_inputs: Union[List[ComputeElement], None]
        """
        self.inputs = inputs
        self.kwargs = kwargs
        self.gradient_inputs = gradient_inputs


api_parent_module_mapping = {
    (MsCompareConst.MINT, Const.MS_FRAMEWORK): mindspore.mint,
    (MsCompareConst.MINT, Const.PT_FRAMEWORK): torch,
    (MsCompareConst.MINT_FUNCTIONAL, Const.MS_FRAMEWORK): mindspore.mint.nn.functional,
    (MsCompareConst.MINT_FUNCTIONAL, Const.PT_FRAMEWORK): torch.nn.functional,
    (MsCompareConst.TENSOR_API, Const.MS_FRAMEWORK): mindspore.Tensor,
    (MsCompareConst.TENSOR_API, Const.PT_FRAMEWORK): torch.Tensor,
    (MsCompareConst.MINDTORCH_TENSOR, Const.MT_FRAMEWORK): mindtorch_tensor,
    (MsCompareConst.MINDTORCH_TENSOR, Const.PT_FRAMEWORK): torch.Tensor,
    (MsCompareConst.MINDTORCH, Const.MT_FRAMEWORK): mindtorch,
    (MsCompareConst.MINDTORCH, Const.PT_FRAMEWORK): torch,
    (MsCompareConst.MINDTORCH_FUNC, Const.MT_FRAMEWORK): mindtorch_func,
    (MsCompareConst.MINDTORCH_FUNC, Const.PT_FRAMEWORK): torch.nn.functional,
    (MsCompareConst.MINDTORCH_DIST, Const.MT_FRAMEWORK): mindtorch_dist,
    (MsCompareConst.MINDTORCH_DIST, Const.PT_FRAMEWORK): torch.distributed,
    (MsCompareConst.FUNCTIONAL_API, Const.MS_FRAMEWORK): mindspore.ops,
    (MsCompareConst.FUSION_API, Const.PT_FRAMEWORK): fusion

}


api_parent_module_str_mapping = {
    (MsCompareConst.MINT, Const.MS_FRAMEWORK): "mindspore.mint",
    (MsCompareConst.MINT, Const.PT_FRAMEWORK): "torch",
    (MsCompareConst.MINT_FUNCTIONAL, Const.MS_FRAMEWORK): "mindspore.mint.nn.functional",
    (MsCompareConst.MINT_FUNCTIONAL, Const.PT_FRAMEWORK): "torch.nn.functional",
    (MsCompareConst.TENSOR_API, Const.MS_FRAMEWORK): "mindspore.Tensor",
    (MsCompareConst.TENSOR_API, Const.PT_FRAMEWORK): "torch.Tensor",
    (MsCompareConst.MINDTORCH_TENSOR, Const.MT_FRAMEWORK): "mindtorch_tensor",
    (MsCompareConst.MINDTORCH_TENSOR, Const.PT_FRAMEWORK): "torch.Tensor",
    (MsCompareConst.MINDTORCH, Const.MT_FRAMEWORK): "mindtorch",
    (MsCompareConst.MINDTORCH, Const.PT_FRAMEWORK): "torch",
    (MsCompareConst.MINDTORCH_FUNC, Const.MT_FRAMEWORK): "mindtorch_func",
    (MsCompareConst.MINDTORCH_FUNC, Const.PT_FRAMEWORK): "torch.nn.functional",
    (MsCompareConst.MINDTORCH_DIST, Const.MT_FRAMEWORK): "mindtorch_dist",
    (MsCompareConst.MINDTORCH_DIST, Const.PT_FRAMEWORK): "torch.distributed",
    (MsCompareConst.FUNCTIONAL_API, Const.MS_FRAMEWORK): "mindspore.ops",
    (MsCompareConst.FUSION_API, Const.PT_FRAMEWORK): "fusion"
}


class ApiRunner:
    def __call__(self, api_input_aggregation, api_name_str, forward_or_backward=Const.FORWARD,
                 api_platform=Const.MS_FRAMEWORK):
        '''
        Args:
            api_input_aggregation: ApiInputAggregation
            api_name_str: str, e.g. "MintFunctional.relu.0"
            forward_or_backward: str, Union["forward", "backward"]
            api_platform: str, Union["mindspore", "torch", "mindtorch"]

        Return:
            outputs: list[ComputeElement]

        Description:
            run mindspore.mint/torch api
        '''

        api_type_str, api_sub_name = self.get_info_from_name(api_name_str, api_platform)
        api_instance = self.get_api_instance(api_type_str, api_sub_name, api_platform)

        return self.run_api(api_instance, api_input_aggregation, forward_or_backward, api_platform)

    @staticmethod
    def get_info_from_name(api_name_str, api_platform=Const.MS_FRAMEWORK):
        """
        Args:
            api_name_str: str, the trimmed key of data dict in api_info.json. e.g. "MintFunctional.relu.0"
            api_platform: str, the platform for the API, which can be either "mindspore" or "mindtorch".
                      It specifies which framework is being used. Default is "mindspore".
        Return:
            api_type_str: str, Union["MintFunctional", "Mint", "Tensor", "Torch", "Functional"]
            api_sub_name: str, e.g. "relu"
        """
        api_name_list = api_name_str.split(Const.SEP)
        if len(api_name_list) != 3:
            err_msg = f"ApiRunner.get_info_from_name failed: api_name_str: {api_name_str} is not in defined format." \
                      f" Exception has been raised and will be captured/logged externally."
            logger.warning_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.WrongValue))
        api_type_str, api_sub_name = api_name_list[0], api_name_list[1]
        if api_type_str not in [MsCompareConst.MINT, MsCompareConst.MINT_FUNCTIONAL, MsCompareConst.TENSOR_API,
                                MsCompareConst.FUNCTIONAL_API] \
                and api_platform == Const.MS_FRAMEWORK:
            err_msg = f"ApiRunner.get_info_from_name failed: not mint, mint.nn.functional or Tensor api," \
                      f" api_name={api_name_str}. Exception has been raised and will be captured/logged externally."
            logger.warning_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.WrongValue))

        if api_type_str not in MsCompareConst.MT_VALID_API_TYPES and api_platform == Const.MT_FRAMEWORK:
            err_msg = f"ApiRunner.get_info_from_name failed: not torch, functional or Tensor api," \
                      f" api_name={api_name_str}. Exception has been raised and will be captured/logged externally."
            logger.warning_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.WrongValue))
        return api_type_str, api_sub_name

    @staticmethod
    def get_api_instance(api_type_str, api_sub_name, api_platform):
        """
        Args:
            api_type_str: str, Union["MintFunctional", "Mint", "Tensor", "Functional"]
            api_sub_name: str, e.g. "relu"
            api_platform: str: Union["mindspore", "pytorch"]

        Return:
            api_instance: function object

        Description:
            get mindspore.mint/torch api function
            mindspore.mint.{api_sub_name} <--> torch.{api_sub_name}
            mindspore.mint.nn.functional.{api_sub_name} <--> torch.nn.functional.{api_sub_name}
        """
        if api_sub_name in MsCompareConst.SUPPORTED_FUSION_LIST and api_platform == "pytorch":
            api_parent_module = api_parent_module_mapping.get((MsCompareConst.FUSION_API, api_platform))
            api_parent_module_str = api_parent_module_str_mapping.get((MsCompareConst.FUSION_API, api_platform))
        else:
            api_parent_module = api_parent_module_mapping.get((api_type_str, api_platform))
            api_parent_module_str = api_parent_module_str_mapping.get((api_type_str, api_platform))
        full_api_name = api_parent_module_str + Const.SEP + api_sub_name

        if not hasattr(api_parent_module, api_sub_name):
            err_msg = f"ApiRunner.get_api_instance failed: {full_api_name} is not found"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.ApiWrong))

        api_instance = getattr(api_parent_module, api_sub_name)
        if not callable(api_instance):
            err_msg = f"ApiRunner.get_api_instance failed: {full_api_name} is not callable"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.ApiWrong))

        return api_instance

    @staticmethod
    def run_api(
        api_instance,
        api_input_aggregation,
        forward_or_backward: str,
        api_platform: str,
    ) -> Union[ForwardResult, BackwardResultMT, PyTorchBackward]:
        inputs = tuple(compute_element.get_parameter(get_origin=False, tensor_platform=api_platform)
                       for compute_element in api_input_aggregation.inputs)
        kwargs = {key: value.get_parameter(get_origin=False, tensor_platform=api_platform)
                  for key, value in api_input_aggregation.kwargs.items()}
        gradient_inputs = api_input_aggregation.gradient_inputs

        if forward_or_backward == Const.FORWARD:
            forward_result = api_instance(*inputs, **kwargs)  # can be single tensor or tuple
            forward_result_tuple = convert_to_tuple(forward_result)
            res_compute_element_list = [ComputeElement(parameter=api_res) for api_res in forward_result_tuple]
            if api_platform == Const.MS_FRAMEWORK or api_platform == Const.MT_FRAMEWORK:
                return res_compute_element_list, inputs, kwargs, forward_result_tuple
        else:
            if gradient_inputs is None:
                err_msg = f"ApiRunner.run_api failed: run backward api but gradient_inputs is missing"
                logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.WrongValue))
            gradient_inputs = tuple(compute_element.get_parameter(get_origin=False, tensor_platform=api_platform)
                                    for compute_element in gradient_inputs)
            if api_platform == Const.MS_FRAMEWORK or api_platform == Const.MT_FRAMEWORK:
                if len(gradient_inputs) == 1:
                    gradient_inputs = gradient_inputs[0]

                def api_with_kwargs(*forward_inputs):
                    return api_instance(*forward_inputs, **kwargs)

                grad_func = ops.GradOperation(get_all=True, sens_param=True)(api_with_kwargs)
                backward_result = grad_func(*inputs, gradient_inputs)  # can be single tensor or tuple
                backward_result_tuple = convert_to_tuple(backward_result)
                res_compute_element_list = [ComputeElement(parameter=api_res) for api_res in backward_result_tuple]
                return res_compute_element_list, gradient_inputs, backward_result_tuple
            else:
                # set requires_grad
                requires_grad_index = []
                for index, tensor in enumerate(inputs):
                    if isinstance(tensor, torch.Tensor) and \
                            torch_dtype_to_dtype_str.get(tensor.dtype) in float_dtype_str_list:
                        setattr(tensor, "requires_grad", True)
                        requires_grad_index.append(index)
                forward_results = api_instance(*inputs, **kwargs)
                forward_results = convert_to_tuple(forward_results)
                for forward_res, gradient_in in zip(forward_results, gradient_inputs):
                    forward_res.backward(gradient_in)
                backward_result_list = []
                for index in requires_grad_index:
                    backward_result_list.append(getattr(inputs[index], "grad"))
                res_compute_element_list = [ComputeElement(parameter=api_res) for api_res in backward_result_list]

        return res_compute_element_list


api_runner = ApiRunner()
