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
import json

import pandas as pd

from msprobe.core.common.file_utils import load_json, load_yaml, create_file_with_content, create_file_in_zip
from msprobe.core.config_check.checkers.base_checker import BaseChecker
from msprobe.core.config_check.config_checker import register_checker_item
from msprobe.core.config_check.utils.utils import config_checking_print, process_pass_check
from msprobe.core.common.const import Const


dirpath = os.path.dirname(__file__)
env_yaml_path = os.path.join(dirpath, "../resource/env.yaml")


def collect_env_data():
    result = {}
    for key, value in os.environ.items():
        result[key] = value
    return result


def get_device_type(env_json):
    for key in env_json.keys():
        if Const.ASCEND in key:
            return Const.NPU_LOWERCASE
    return Const.GPU_LOWERCASE


def compare_env_data(npu_path, bench_path):
    necessary_env = load_yaml(env_yaml_path)
    cmp_data = load_json(npu_path)
    cmp_type = get_device_type(cmp_data)
    bench_data = load_json(bench_path)
    bench_type = get_device_type(bench_data)
    data = []
    for _, value in necessary_env.items():
        cmp_env = value.get(cmp_type)
        bench_env = value.get(bench_type)
        if not bench_env and not cmp_env:
            continue
        elif cmp_env:
            cmp_env_name = cmp_env["name"]
            cmp_value = cmp_data.get(cmp_env_name, value[cmp_type]["default_value"])
            if not bench_env:
                data.append(["only cmp has this env", cmp_env["name"], "", cmp_value, Const.CONFIG_CHECK_WARNING])
                continue
            bench_env_name = bench_env["name"]
            bench_value = bench_data.get(bench_env_name, value[bench_type]["default_value"])
            if cmp_value != bench_value:
                data.append([bench_env_name, cmp_env_name, bench_value, cmp_value, Const.CONFIG_CHECK_ERROR])
        else:
            bench_env_name = bench_env["name"]
            bench_value = bench_data.get(bench_env_name) if bench_data.get(bench_env_name) else value[bench_type][
                "default_value"]
            data.append([bench_env_name, "only bench has this env", bench_value, "", Const.CONFIG_CHECK_WARNING])
    df = pd.DataFrame(data, columns=EnvArgsChecker.result_header)
    return df


@register_checker_item("env")
class EnvArgsChecker(BaseChecker):

    target_name_in_zip = "env"
    result_header = ["bench_env_name", "cmp_env_name", "bench_value", "cmp_value", "level"]

    @staticmethod
    def pack(pack_input):
        output_zip_path = pack_input.output_zip_path
        env_args_dict = collect_env_data()
        create_file_in_zip(output_zip_path, EnvArgsChecker.target_name_in_zip, json.dumps(env_args_dict, indent=4))
        config_checking_print(f"add env args to zip")

    @staticmethod
    def compare(bench_dir, cmp_dir, output_path, fmk):
        bench_env_data = os.path.join(bench_dir, EnvArgsChecker.target_name_in_zip)
        cmp_env_data = os.path.join(cmp_dir, EnvArgsChecker.target_name_in_zip)
        df = compare_env_data(bench_env_data, cmp_env_data)
        pass_check = process_pass_check(df['level'].values)
        return EnvArgsChecker.target_name_in_zip, pass_check, df
