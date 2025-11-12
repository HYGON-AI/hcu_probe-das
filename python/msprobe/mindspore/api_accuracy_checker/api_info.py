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

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import ApiAccuracyCheckerException
from msprobe.core.common.utils import is_invalid_pattern
from msprobe.mindspore.api_accuracy_checker.compute_element import ComputeElement
from msprobe.mindspore.api_accuracy_checker.utils import check_and_get_from_json_dict
from msprobe.mindspore.common.log import logger


class ApiInfo:
    def __init__(self, api_name):
        if not isinstance(api_name, str):
            err_msg = "ApiInfo.__init__ failed: api_name is not a string"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.ParseJsonFailed))
        if is_invalid_pattern(api_name):
            err_msg = "ApiInfo.__init__ failed: api_name contain illegal character"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.ParseJsonFailed))
        self.api_name = api_name
        self.forward_info = None
        self.backward_info = None

    def load_forward_info(self, forward_info_dict):
        self.forward_info = forward_info_dict

    def load_backward_info(self, backward_info_dict):
        self.backward_info = backward_info_dict

    def check_forward_info(self):
        return self.forward_info is not None

    def check_backward_info(self):
        return self.backward_info is not None

    def get_compute_element_list(self, forward_or_backward, input_or_output):
        '''
        Args:
            forward_or_backward: str, Union["forward", "backward"]
            input_or_output: str,  Union["input", "output"]

        Return:
            compute_element_list: List[ComputeElement]
        '''
        mapping = {
            (Const.FORWARD, Const.INPUT): [self.forward_info, Const.INPUT_ARGS,
                                           f"input_args field of {self.api_name} forward api in api_info.json"],
            (Const.FORWARD, Const.OUTPUT): [self.forward_info, Const.OUTPUT,
                                            f"output field of {self.api_name} forward api in api_info.json"],
            (Const.BACKWARD, Const.INPUT): [self.backward_info, Const.INPUT,
                                            f"input field of {self.api_name} backward api in api_info.json"],
            (Const.BACKWARD, Const.OUTPUT): [self.backward_info, Const.OUTPUT,
                                             f"output field of {self.api_name} backward api in api_info.json"]
        }
        dict_instance, key, key_desc = mapping.get((forward_or_backward, input_or_output))
        compute_element_info_list = check_and_get_from_json_dict(dict_instance, key, key_desc, accepted_type=list)
        compute_element_list = [ComputeElement(compute_element_info=compute_element_info)
                                for compute_element_info in compute_element_info_list]
        return compute_element_list

    def get_kwargs(self):
        '''
        Return:
            kwargs_compute_element_dict: dict{str: ComputeElement}
        '''
        kwargs_dict = check_and_get_from_json_dict(self.forward_info, Const.INPUT_KWARGS,
                                                   "input_kwargs in api_info.json", accepted_type=dict)
        for key_str, compute_element_info in kwargs_dict.items():
            if not isinstance(key_str, str):
                err_msg = "ApiInfo.get_kwargs failed: compute_element_dict key is not a string"
                logger.error_log_with_exp(err_msg,
                                          ApiAccuracyCheckerException(ApiAccuracyCheckerException.ParseJsonFailed))
            if not (isinstance(compute_element_info, (list, dict)) or compute_element_info is None):
                err_msg = "ApiInfo.get_kwargs failed: compute_element_dict value is not a list, dict or null"
                logger.error_log_with_exp(err_msg,
                                          ApiAccuracyCheckerException(ApiAccuracyCheckerException.ParseJsonFailed))
        kwargs_compute_element_dict = {key_str: ComputeElement(compute_element_info=compute_element_info)
                                       for key_str, compute_element_info in kwargs_dict.items()}
        return kwargs_compute_element_dict
