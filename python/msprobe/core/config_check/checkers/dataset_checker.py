# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
import json
import pandas as pd
from msprobe.core.common.file_utils import create_file_in_zip, load_json
from msprobe.core.config_check.checkers.base_checker import BaseChecker
from msprobe.core.config_check.config_checker import register_checker_item, register_pre_forward_fun_list
from msprobe.core.config_check.utils.utils import config_checking_print, get_tensor_features
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.const import Const


@recursion_depth_decorator("config_check: process_obj")
def process_obj(obj):
    if FmkAdp.is_tensor(obj):
        return get_tensor_features(obj)
    elif isinstance(obj, (tuple, list)):
        return {i: process_obj(x) for i, x in enumerate(obj)}
    elif isinstance(obj, dict):
        return {k: process_obj(v) for k, v in obj.items()}
    else:
        return ""


def parse_args_and_kargs(args, kwargs):
    processed_args = process_obj(args)
    processed_kargs = process_obj(kwargs)

    return {
        'args': processed_args,
        'kwargs': processed_kargs
    }


@recursion_depth_decorator("config_check: compare_dataset_dicts")
def compare_dataset_dicts(dict1, dict2, tag=''):
    results = []
    # 处理 dict1 中的键
    for key in dict1:
        new_tag = f"{tag}.{key}" if tag else key
        if key not in dict2:
            result = {'tag': new_tag, 'equal': False, 'status': 'delete'}
            results.append(result)
            continue
        value1 = dict1[key]
        value2 = dict2[key]
        if not isinstance(value1, dict):
            continue
        if set(value1.keys()) == {'max', 'min', 'mean', 'norm'}:
            equal = value1 == value2
            relative_diffs = {
                f"{k}_relative_diff": (abs(value1[k] - value2[k]) / value1[k]) if value1[k] != 0 else None
                for k in ['max', 'min', 'mean', 'norm']
            }
            result = {'tag': new_tag, 'equal': equal, 'status': 'unchanged'}
            result.update(relative_diffs)
            results.append(result)
        else:
            results.extend(compare_dataset_dicts(value1, value2, new_tag))
    # 处理 dict2 中独有的键
    for key in dict2:
        if key not in dict1:
            new_tag = f"{tag}.{key}" if tag else key
            result = {'tag': new_tag, 'equal': False, 'status': 'added'}
            results.append(result)
    return results


def compare_dataset(bench_dir, cmp_dir):
    all_results = []
    for step in os.listdir(bench_dir):
        step_path_bench = os.path.join(bench_dir, step)
        if not os.path.isdir(step_path_bench):
            continue
        step_path_cmp = os.path.join(cmp_dir, step)
        for rank in os.listdir(step_path_bench):
            rank_path_bench = os.path.join(step_path_bench, rank, 'dataset.json')
            rank_path_cmp = os.path.join(step_path_cmp, rank, 'dataset.json')
            if not os.path.isfile(rank_path_bench) or not os.path.isfile(rank_path_cmp):
                continue

            dict1 = load_json(rank_path_bench)
            dict2 = load_json(rank_path_cmp)
            results = compare_dataset_dicts(dict1, dict2)
            for result in results:
                result['step'] = int(step.replace("step", ""))
                result['rank'] = int(rank.replace("rank", ""))
            all_results.extend(results)

    df = pd.DataFrame(all_results, columns=DatasetChecker.result_header)
    df = df.sort_values(by=['step', 'rank'], ascending=[True, True])
    return df


@register_checker_item("dataset")
class DatasetChecker(BaseChecker):
    input_needed = "model"
    multi_rank = True

    target_name_in_zip = "dataset"
    result_header = ['step', 'rank', 'tag', 'equal', 'max_relative_diff',
                     'min_relative_diff', 'mean_relative_diff', 'norm_relative_diff']

    @staticmethod
    def pack(pack_input):
        output_zip_path = pack_input.output_zip_path

        def collect_input(model, args, kwargs, step):
            features = parse_args_and_kargs(args, kwargs)
            dataset_filepath = os.path.join(DatasetChecker.target_name_in_zip, 
                                            f"step{step}", f"rank{FmkAdp.get_rank_id()}", "dataset.json")
            create_file_in_zip(output_zip_path, dataset_filepath, json.dumps(features, indent=4))
            config_checking_print(f"add first dataset input features to zip")
            
        register_pre_forward_fun_list(collect_input)

    @staticmethod
    def compare(bench_dir, cmp_dir, output_path):
        bench_dataset_pack_path = os.path.join(bench_dir, DatasetChecker.target_name_in_zip)
        cmp_dataset_pack_path = os.path.join(cmp_dir, DatasetChecker.target_name_in_zip)

        df = compare_dataset(bench_dataset_pack_path, cmp_dataset_pack_path)
        pass_check = Const.CONFIG_CHECK_PASS if False not in df['equal'].values else Const.CONFIG_CHECK_ERROR
        return DatasetChecker.target_name_in_zip, pass_check, df
