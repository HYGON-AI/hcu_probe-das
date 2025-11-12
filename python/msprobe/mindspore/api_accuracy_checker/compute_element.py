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

import os

import mindspore
import numpy as np
from mindspore._c_expression import typing
from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import ApiAccuracyCheckerException
from msprobe.core.common.file_utils import load_npy
from msprobe.mindspore.api_accuracy_checker.type_mapping import (api_info_type_str_to_type,
                                                                 ms_dtype_to_dtype_str, torch_dtype_to_dtype_str,
                                                                 dtype_str_to_ms_dtype, dtype_str_to_np_dtype,
                                                                 dtype_str_to_mindtorch_dtype,
                                                                 dtype_str_to_torch_dtype, type_to_api_info_type_str,
                                                                 DEFAULT_CONSTRUCT_NP_FLOAT_DTYPE, TUPLE_TYPE_STR,
                                                                 MINDSPORE_TENSOR_TYPE_STR, MINDSPORE_DTYPE_TYPE_STR,
                                                                 SLICE_TYPE_STR, TORCH_DTYPE_TYPE_STR,
                                                                 float_dtype_str_list, int_dtype_str_list)
from msprobe.mindspore.api_accuracy_checker.utils import check_and_get_from_json_dict, global_context
from msprobe.mindspore.common.log import logger

import msprobe.mindspore.api_accuracy_checker.torch_mindtorch_importer as env_module


if env_module.is_valid_pt_mt_env:
    from msprobe.mindspore.api_accuracy_checker.torch_mindtorch_importer import mindtorch
    from msprobe.mindspore.api_accuracy_checker.torch_mindtorch_importer import torch
else:
    import torch


class MstensorMetaData:
    def __init__(self, dtype_str, npy_path, maximum, minimum, shape) -> None:
        self.dtype_str = dtype_str
        self.npy_path = npy_path
        self.maximum = maximum
        self.minimum = minimum
        self.shape = shape


class DtypeMetaData:
    def __init__(self, dtype_str) -> None:
        self.dtype_str = dtype_str


class ComputeElement:
    def __init__(self, compute_element_info=None, parameter=None):
        self.supported_parameter_type = tuple(type_to_api_info_type_str.keys()) + tuple([torch.Tensor, tuple])
        if parameter is not None:
            self._init_with_parameter(parameter)
        elif isinstance(compute_element_info, (list, dict)):
            self._init_from_compute_element_info(compute_element_info)
        elif compute_element_info is None:
            self._init_from_null_compute_element_info()
        else:
            logger.warning_log_with_exp(
                "ComputeElement.__init__ failed: not init with parameter or compute_element info is not (list, dict)."
                " Exception has been raised and will be captured/logged externally.",
                ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))

    @staticmethod
    def transfer_to_torch_tensor(ms_tensor):
        '''
        Args:
            ms_tensor: mindspore.Tensor
        Return:
            torch_tensor: torch.Tensor
        '''
        ms_dtype = ms_tensor.dtype
        dtype_str = ms_dtype_to_dtype_str.get(ms_dtype)
        if dtype_str not in dtype_str_to_torch_dtype:
            err_msg = f"ComputeElement.transfer_to_torch_tensor failed: no matching torch dtype" \
                      f" for {dtype_str}. Exception has been raised and will be captured/logged externally."
            logger.warning_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))
        else:
            torch_dtype = dtype_str_to_torch_dtype.get(dtype_str)

        if dtype_str in int_dtype_str_list:
            middle_dtype = mindspore.int64
        else:
            middle_dtype = mindspore.float64
        np_ndarray = ms_tensor.astype(middle_dtype).numpy()
        torch_tensor = torch.from_numpy(np_ndarray).to(torch_dtype)
        return torch_tensor

    @staticmethod
    def transfer_to_mindtorch_tensor(ms_tensor):
        """
        Args:
            ms_tensor: mindspore.Tensor
        Return:
            mindtorch_tensor: mindtorch.Tensor
        """

        ms_dtype = ms_tensor.dtype

        dtype_str = ms_dtype_to_dtype_str.get(ms_dtype)

        if dtype_str not in dtype_str_to_mindtorch_dtype:
            err_msg = f"ComputeElement.transfer_to_mindtorch_tensor failed: no matching mindtorch dtype for" \
                      f" {dtype_str}. Exception has been raised and will be captured/logged externally."
            logger.warning_log_with_exp(err_msg,
                                      ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))
        else:
            mindtorch_dtype = dtype_str_to_mindtorch_dtype.get(dtype_str)

        if dtype_str in int_dtype_str_list:
            middle_dtype = mindspore.int64
        else:
            middle_dtype = mindspore.float64

        np_ndarray = ms_tensor.astype(middle_dtype).numpy()

        mindtorch_tensor = mindtorch.from_numpy(np_ndarray).to(ms_dtype)

        return mindtorch_tensor

    @staticmethod
    def transfer_to_mindspore_tensor(torch_tensor):
        '''
        Args:
            torch_tensor: torch.Tensor

        Return:
            ms_tensor: mindspore.Tensor
        '''
        torch_dtype = torch_tensor.dtype
        dtype_str = torch_dtype_to_dtype_str.get(torch_dtype)
        if dtype_str not in dtype_str_to_ms_dtype:
            err_msg = \
                f"ComputeElement._transfer_to_mindspore_tensor failed: no matching mindspore dtype for {dtype_str}. " \
                f"Exception has been raised and will be captured/logged externally."
            logger.warning_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))
        else:
            ms_dtype = dtype_str_to_ms_dtype.get(dtype_str)

        if dtype_str in int_dtype_str_list:
            middle_dtype = torch.int64
        else:
            middle_dtype = torch.float64
        np_ndarray = torch_tensor.to(middle_dtype, copy=True).numpy()
        ms_tensor = mindspore.Tensor.from_numpy(np_ndarray).astype(ms_dtype)
        return ms_tensor

    @staticmethod
    def convert_inf_to_real_num(value, dtype_str):
        if value == float("inf"):
            np_dtype = dtype_str_to_np_dtype.get(dtype_str, DEFAULT_CONSTRUCT_NP_FLOAT_DTYPE)
            value = np.finfo(np_dtype).max
        elif value == float("-inf"):
            np_dtype = dtype_str_to_np_dtype.get(dtype_str, DEFAULT_CONSTRUCT_NP_FLOAT_DTYPE)
            value = np.finfo(np_dtype).min
        return value

    def get_parameter(self, get_origin=True, tensor_platform=Const.MS_FRAMEWORK):
        '''
        Args:
            get_origin: boolean
            tensor_platform: str, Union["mindspore", "pytorch"]

        Return:
            parameter: Union[int, float, str, slice, tuple, torch.Tensor, mindspore.Tensor]
        '''
        if self.parameter is None:
            return self.parameter
        if isinstance(self.parameter, tuple):
            return tuple([compute_element.get_parameter(get_origin=get_origin, tensor_platform=tensor_platform)
                          for compute_element in self.parameter])
        elif isinstance(self.parameter, self.supported_parameter_type):
            parameter_tmp = self.parameter
        elif isinstance(self.parameter, DtypeMetaData):
            if tensor_platform == Const.MS_FRAMEWORK:
                parameter_tmp = dtype_str_to_ms_dtype.get(self.parameter.dtype_str)
            elif tensor_platform == Const.PT_FRAMEWORK:
                parameter_tmp = dtype_str_to_torch_dtype.get(self.parameter.dtype_str)
            elif tensor_platform == Const.MT_FRAMEWORK:
                parameter_tmp = dtype_str_to_mindtorch_dtype.get(self.parameter.dtype_str)

        elif isinstance(self.parameter, MstensorMetaData):
            mstensor_meta_data = self.parameter
            ms_dtype = dtype_str_to_ms_dtype.get(mstensor_meta_data.dtype_str)
            if global_context.get_is_constructed():
                np_dtype = dtype_str_to_np_dtype.get(mstensor_meta_data.dtype_str, DEFAULT_CONSTRUCT_NP_FLOAT_DTYPE)
                ndarray = self._construct_ndarray(mstensor_meta_data.shape, mstensor_meta_data.maximum,
                                                  mstensor_meta_data.minimum, np_dtype)
            else:
                ndarray = load_npy(mstensor_meta_data.npy_path)
            parameter_tmp = mindspore.Tensor(ndarray, dtype=ms_dtype)
        else:
            err_msg = "ComputeElement.get_parameter failed: self.parameter type is not in " \
                      "(int, float, str, slice, bool, torch.Tensor, mindspore.Tensor, MstensorMetaData)." \
                      "Exception has been raised and will be captured/logged externally."
            logger.warning_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))

        # if necessary, do transfer
        if not get_origin and isinstance(parameter_tmp, mindspore.Tensor) and tensor_platform == Const.PT_FRAMEWORK:
            parameter = self.transfer_to_torch_tensor(parameter_tmp)
        elif not get_origin and isinstance(parameter_tmp, mindspore.Tensor) and tensor_platform == Const.MT_FRAMEWORK:
            parameter = self.transfer_to_mindtorch_tensor(parameter_tmp)
        elif not get_origin and isinstance(parameter_tmp, torch.Tensor) and tensor_platform == Const.MS_FRAMEWORK:
            parameter = self.transfer_to_mindspore_tensor(parameter_tmp)
        else:
            parameter = parameter_tmp

        return parameter

    def get_shape(self):
        return self.shape

    def get_dtype(self):
        return self.dtype_str

    def _construct_ndarray(self, shape, maximum, minimum, np_dtype):
        shape = tuple(shape)
        np.random.seed(42)
        if np_dtype == np.bool_:
            ndarray = np.random.rand(*shape) > 0.5
        else:
            maximum = self.convert_inf_to_real_num(maximum, np_dtype)
            minimum = self.convert_inf_to_real_num(minimum, np_dtype)
            ndarray = np.random.uniform(minimum, maximum, shape).astype(np_dtype)
        return ndarray

    def _init_from_null_compute_element_info(self):
        self.parameter = None
        self.shape = tuple()
        self.dtype = "None"

    def _init_from_compute_element_info(self, compute_element_info):
        '''
        Args:
            compute_element_info: Union[list, dict]

        Return:
            void

        init member attributes: self.shape, self.dtype_str, self.parameter
        '''
        if isinstance(compute_element_info, list):
            self.shape = tuple()
            self.dtype_str = TUPLE_TYPE_STR
            self.parameter = tuple([ComputeElement(compute_element_info=sub_info)
                                    for sub_info in compute_element_info])
        else:
            type_str = check_and_get_from_json_dict(compute_element_info, "type", "type field in api_info.json",
                                                    accepted_type=str, accepted_value=api_info_type_str_to_type.keys())
            self.shape = tuple()
            self.dtype_str = type_str
            if type_str == MINDSPORE_TENSOR_TYPE_STR:
                self._init_from_mstensor_compute_element_info(compute_element_info)
            else:
                value = check_and_get_from_json_dict(compute_element_info, "value", "value field in api_info.json")
                if type_str == MINDSPORE_DTYPE_TYPE_STR:
                    self.parameter = DtypeMetaData(value)
                elif type_str == SLICE_TYPE_STR:
                    self.parameter = slice(*tuple(value))
                else:  # type_str in ("str", "int", "float", "bool")
                    self.parameter = value

    def _init_from_mstensor_compute_element_info(self, compute_element_info):
        '''
            do not load real tensor, only record meta data
        '''
        dtype_str = check_and_get_from_json_dict(compute_element_info, "dtype", "dtype field in api_info.json",
                                                 accepted_type=str, accepted_value=dtype_str_to_ms_dtype.keys())
        shape = check_and_get_from_json_dict(compute_element_info, "shape", "shape field in api_info.json",
                                             accepted_type=(list,))
        if global_context.get_is_constructed():
            maximum = check_and_get_from_json_dict(compute_element_info, "Max", "Max field in api_info.json",
                                                   accepted_type=(int, float))
            minimum = check_and_get_from_json_dict(compute_element_info, "Min", "Min field in api_info.json",
                                                   accepted_type=(int, float))

            npy_path = None
        else:
            maximum, minimum = None, None
            data_name = check_and_get_from_json_dict(compute_element_info, "data_name",
                                                     "data_name field in api_info.json", accepted_type=(str,))
            npy_path = os.path.join(global_context.get_dump_data_dir(), data_name)
        mstensor_meta_data = MstensorMetaData(dtype_str, npy_path, maximum, minimum, shape)
        self.parameter = mstensor_meta_data
        self.dtype_str = dtype_str
        self.shape = tuple(shape)

    def _init_with_parameter(self, parameter):
        self.parameter = parameter
        self.shape = tuple()
        if not isinstance(parameter, self.supported_parameter_type):
            err_msg = "ComputeElement._init_with_parameter failed: " \
                      "parameter type is not in (int, float, str, slice, bool, torch.Tensor, mindspore.Tensor)." \
                      "Exception has been raised and will be captured/logged externally."
            logger.warning_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))
        if isinstance(parameter, mindspore.Tensor):
            self.shape = tuple(parameter.shape)
            self.dtype_str = ms_dtype_to_dtype_str.get(parameter.dtype)
        elif isinstance(parameter, torch.Tensor):
            self.shape = tuple(parameter.shape)
            self.dtype_str = torch_dtype_to_dtype_str.get(parameter.dtype)
        elif isinstance(parameter, typing.Type):
            self.dtype_str = MINDSPORE_DTYPE_TYPE_STR
            self.parameter = DtypeMetaData(ms_dtype_to_dtype_str.get(parameter))
        elif isinstance(parameter, torch.dtype):
            self.dtype_str = TORCH_DTYPE_TYPE_STR
            self.parameter = DtypeMetaData(torch_dtype_to_dtype_str.get(parameter))
        elif isinstance(parameter, tuple):
            self.dtype_str = TUPLE_TYPE_STR
            self.parameter = tuple([ComputeElement(parameter=param) for param in parameter])
        else:
            self.dtype_str = type_to_api_info_type_str.get(type(parameter))
