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

from msprobe.core.common.file_utils import create_file_in_zip, os_walk_for_files, load_json
from msprobe.core.config_check.checkers.base_checker import BaseChecker
from msprobe.core.config_check.config_checker import register_checker_item, register_pre_forward_fun_list
from msprobe.core.config_check.utils.utils import config_checking_print, get_tensor_features
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.const import Const


def collect_weights_data(model):
    weights_data = {}
    for name, param in FmkAdp.named_parameters(model):
        if param.dtype != FmkAdp.dtype("float32"):
            param = param.float()
        weights_data[name] = get_tensor_features(param)
    return weights_data


def compare_weight_file(bench_file, cmp_file):
    bench_data = load_json(bench_file)
    cmp_data = load_json(cmp_file)

    results = []
    for weight_name in set(bench_data.keys()) | set(cmp_data.keys()):
        result = {
            "weight_name": weight_name,
            "equal": None,
            "max_relative_diff": None,
            "min_relative_diff": None,
            "mean_relative_diff": None,
            "norm_relative_diff": None
        }

        if weight_name not in bench_data:
            result["equal"] = "only cmp have"
            results.append(result)
            continue

        if weight_name not in cmp_data:
            result["equal"] = "only bench have"
            results.append(result)
            continue

        bench_vals = bench_data[weight_name]
        cmp_vals = cmp_data[weight_name]
        keys = ["max", "min", "mean", "norm"]
        equal = all([bench_vals[k] == cmp_vals[k] for k in keys])
        result["equal"] = equal

        for key in keys:
            diff_key = f"{key}_relative_diff"
            result[diff_key] = (abs(bench_vals[key] - cmp_vals[key]) / bench_vals[key]) \
            if bench_vals[key] != 0 else None

        results.append(result)

    return results


def compare_weight(bench_dir, cmp_dir):
    all_results = []
    bench_files_info = os_walk_for_files(bench_dir, 10)
    for info in bench_files_info:
        if not info["file"].endswith('.json'):
            continue
        bench_file = os.path.join(info["root"], info["file"])
        relative_path = os.path.relpath(info["root"], bench_dir)
        cmp_root = os.path.join(cmp_dir, relative_path)
        cmp_file = os.path.join(cmp_root, info["file"])

        path_list = relative_path.split(os.sep)
        if len(path_list) < 2:
            raise Exception("Can not compare weights because the extracted file has been corrupted!")
        step = int(path_list[0].replace("step", ""))
        rank = int(path_list[1].replace("rank", ""))

        if not os.path.exists(cmp_file):
            bench_data = load_json(bench_file)
            for weight_name in bench_data.keys():
                result = {
                    "step": step,
                    "rank": rank,
                    "weight_name": weight_name,
                    "equal": "only bench have",
                    "max_relative_diff": None,
                    "min_relative_diff": None,
                    "mean_relative_diff": None,
                    "norm_relative_diff": None
                }
                all_results.append(result)
        else:
            results = compare_weight_file(bench_file, cmp_file)
            for res in results:
                res["step"] = step
                res["rank"] = rank
                all_results.append(res)

    df = pd.DataFrame(all_results, columns=WeightsChecker.result_header)
    df = df.sort_values(by=['step', 'rank'], ascending=[True, True])
    return df


@register_checker_item("weights")
class WeightsChecker(BaseChecker):
    input_needed = "model"
    multi_rank = True

    target_name_in_zip = "weights"
    result_header = ["step", "rank", "weight_name", "equal", "max_relative_diff", 
                     "min_relative_diff", "mean_relative_diff", "norm_relative_diff"]

    @staticmethod
    def pack(pack_input):
        output_zip_path = pack_input.output_zip_path

        def collect_weights(model, args, kwargs, step):
            weights_data_dict = collect_weights_data(model)
            weights_data_filepath = os.path.join(WeightsChecker.target_name_in_zip, 
                                                 f"step{step}", f"rank{FmkAdp.get_rank_id()}", "weight.json")
            create_file_in_zip(output_zip_path, weights_data_filepath, json.dumps(weights_data_dict, indent=4))
            config_checking_print(f"add weights info to zip")
        register_pre_forward_fun_list(collect_weights)

    @staticmethod
    def compare(bench_dir, cmp_dir, output_path, fmk):
        bench_weight_pack_path = os.path.join(bench_dir, WeightsChecker.target_name_in_zip)
        cmp_weight_pack_path = os.path.join(cmp_dir, WeightsChecker.target_name_in_zip)
        df = compare_weight(bench_weight_pack_path, cmp_weight_pack_path)
        pass_check = Const.CONFIG_CHECK_PASS if False not in df['equal'].values else Const.CONFIG_CHECK_ERROR
        return WeightsChecker.target_name_in_zip, pass_check, df
