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

from msprobe.core.common.exceptions import ParseJsonException
from msprobe.core.common.file_utils import load_json
from msprobe.core.common.log import logger


def parse_json_info_forward_backward(json_path):
    dump_json = load_json(json_path)

    real_data_path = dump_json.get("dump_data_dir")
    dump_data = dump_json.get("data")
    if dump_data is None:
        raise ParseJsonException(ParseJsonException.InvalidDumpJson, 
                                 "something wrong with dump, no data found in dump.json")
    if not dump_data:
        logger.warning("data field is empty, no overflow data found.")

    forward_data = {}
    backward_data = {}
    for data_name, data_item in dump_data.items():
        if "Module" in data_name:
            continue
        if "forward" in data_name:
            api_name = parse_data_name_with_pattern(data_name, "forward", json_path)
            forward_data.update({api_name: data_item})
        elif "backward" in data_name:
            api_name = parse_data_name_with_pattern(data_name, "backward", json_path)
            backward_data.update({api_name: data_item})
        else:
            raise ParseJsonException(ParseJsonException.UnexpectedNameStruct,
                                     f"{data_name} in file {json_path}.")

    return forward_data, backward_data, real_data_path


def parse_data_name_with_pattern(data_name, pattern, json_path):
    name_struct = data_name.split('.')
    if not name_struct[-1] == pattern:
        raise ParseJsonException(ParseJsonException.UnexpectedNameStruct, f"{data_name} in file {json_path}")
    api_name = '.'.join(name_struct[:-1])
    return api_name
