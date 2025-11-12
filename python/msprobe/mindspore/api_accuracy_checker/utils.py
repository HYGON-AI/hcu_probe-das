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
from msprobe.mindspore.api_accuracy_checker.type_mapping import float_dtype_str_list
from msprobe.mindspore.common.log import logger


def check_and_get_from_json_dict(dict_instance, key, key_description, accepted_type=None, accepted_value=None):
    '''
    Args:
        dict_instance: dict, dict parsed from input json
        key: str
        key_description: str
        accepted_type: tuple
        accepted_value: Union[tuple, list]

    Return:
        value, the corresponding value of "key" in "dict_instance"

    Exception:
        raise ApiAccuracyCheckerException.ParseJsonFailed error when
        1. dict_instance is not a dict
        2. value is None
        3. value is not accepted type
        4. value is not accepted value
    '''
    if not isinstance(dict_instance, dict):
        error_info = "check_and_get_from_json_dict failed: input is not a dict"
        raise ApiAccuracyCheckerException(ApiAccuracyCheckerException.ParseJsonFailed, error_info)
    value = dict_instance.get(key)
    if value is None:
        error_info = f"check_and_get_from_json_dict failed: {key_description} is missing"
        raise ApiAccuracyCheckerException(ApiAccuracyCheckerException.ParseJsonFailed, error_info)
    elif accepted_type is not None and not isinstance(value, accepted_type):
        error_info = f"check_and_get_from_json_dict failed: {key_description} is not accepted type: {accepted_type}"
        raise ApiAccuracyCheckerException(ApiAccuracyCheckerException.ParseJsonFailed, error_info)
    elif accepted_value is not None and value not in accepted_value:
        error_info = f"check_and_get_from_json_dict failed: {key_description} is not accepted value: {accepted_value}"
        raise ApiAccuracyCheckerException(ApiAccuracyCheckerException.ParseJsonFailed, error_info)
    return value


def convert_to_tuple(args):
    if isinstance(args, (tuple, list)):
        return tuple(args)
    else:
        input_list = [args]
        return tuple(input_list)


def trim_output_compute_element_list(compute_element_list, forward_or_backward):
    '''
    Args:
        compute_element_list: List[ComputeElement]
        forward_or_backward: str, Union["forward", "backward"]
    '''
    trimmed_list = []
    for compute_element in compute_element_list:
        if compute_element.get_parameter() is None or \
                (forward_or_backward == Const.BACKWARD and compute_element.get_dtype() not in float_dtype_str_list):
            # trim case: 1. parameter is None. 2. backward output has non float parameter
            continue
        trimmed_list.append(compute_element)
    return trimmed_list


class GlobalContext:
    def __init__(self):
        self.is_constructed = True
        self.dump_data_dir = ""
        self.framework = Const.MS_FRAMEWORK

    def init(self, is_constructed, dump_data_dir, framework):
        self.is_constructed = is_constructed
        self.dump_data_dir = dump_data_dir
        self.framework = framework

    def get_dump_data_dir(self):
        return self.dump_data_dir

    def get_is_constructed(self):
        return self.is_constructed

    def get_framework(self):
        return self.framework


global_context = GlobalContext()
