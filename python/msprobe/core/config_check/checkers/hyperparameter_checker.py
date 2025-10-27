# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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
from difflib import SequenceMatcher

from typing import Union, List, Dict, Any
import pandas as pd

from python.msprobe.core.common.utils import check_extern_input_list
from python.msprobe.core.config_check.checkers.base_checker import BaseChecker
from python.msprobe.core.config_check.config_checker import register_checker_item
from python.msprobe.core.config_check.utils.utils import compare_dict, config_checking_print, update_dict, \
    process_pass_check
from python.msprobe.core.config_check.utils.hyperparameter_parser import ParserFactory
from python.msprobe.core.common.file_utils import check_file_or_directory_path, create_file_in_zip, load_json, load_yaml
from python.msprobe.core.common.const import Const


dirpath = os.path.dirname(__file__)
hyperparameters_path = os.path.join(dirpath, "../resource/hyperparameter.yaml")
parameter_name_mapping = load_yaml(os.path.realpath(hyperparameters_path))
hyperparameters_dict = {}


def refine_json_keys(json_dcit):
    new_dict = {}
    for key in json_dcit.keys():
        new_key = key.split(Const.SEP)[-1].replace("-", "_")
        new_dict[new_key] = key 
    return new_dict       


def to_str_if_number(value):
    if isinstance(value, (int, float)):
        return str(value)
    return value


@register_checker_item("hyperparameter")
class HyperparameterChecker(BaseChecker):
    target_name_in_zip = "hyperparameters"
    result_header = ["file_name", "bench_para", "cmp_para", "bench_value", "cmp_value", "matched_with", "level"]
    hyperparameters_file_list = ["hyperparameters_static.json", "hyperparameters_dynamic.json"]

    @staticmethod
    def pack(pack_input):
        shell_path = pack_input.shell_path
        output_zip_path = pack_input.output_zip_path

        if shell_path:
            check_extern_input_list(shell_path)

            hyperparameters = {}
            parser_factory = ParserFactory()
            for script_path in shell_path:
                if os.path.isfile(script_path):
                    check_file_or_directory_path(script_path)
                    parser = parser_factory.get_parser(os.path.splitext(script_path)[1])
                    update_dict(hyperparameters, parser.run(os.path.realpath(script_path)))
                else:
                    config_checking_print(f"Warning: Script path {script_path} is not a file.")
            if hyperparameters:
                create_file_in_zip(output_zip_path,
                                   os.path.join(HyperparameterChecker.target_name_in_zip,
                                                HyperparameterChecker.hyperparameters_file_list[0]),
                                   json.dumps(hyperparameters, indent=4))
                config_checking_print(f"add static hyperparameters args to zip")
            else:
                config_checking_print(f"Warning: Failed to extract hyperparameters from script {shell_path}")
        if hyperparameters_dict:
            create_file_in_zip(output_zip_path,
                               os.path.join(HyperparameterChecker.target_name_in_zip,
                                            HyperparameterChecker.hyperparameters_file_list[1]),
                               json.dumps(vars(hyperparameters_dict), default=lambda x: None, indent=4))
            config_checking_print(f"add dynamic hyperparameters args to zip")

    @staticmethod
    def compare(bench_dir, cmp_dir, output_path, fmk):
        all_diffs = []
        for file_name in HyperparameterChecker.hyperparameters_file_list:
            bench_model_dir = os.path.join(bench_dir, HyperparameterChecker.target_name_in_zip, file_name)
            cmp_model_dir = os.path.join(cmp_dir, HyperparameterChecker.target_name_in_zip, file_name)
            if os.path.isfile(bench_model_dir) and os.path.isfile(cmp_model_dir):
                bench_hyperparameters = load_json(bench_model_dir)
                cmp_hyperparameters = load_json(cmp_model_dir)
                all_diffs.extend(
                    HyperparameterChecker.compare_param(bench_hyperparameters, cmp_hyperparameters, file_name))
        df = pd.DataFrame(all_diffs, columns=HyperparameterChecker.result_header)
        pass_check = process_pass_check(df["level"].values)
        return HyperparameterChecker.target_name_in_zip, pass_check, df

    @staticmethod
    def compare_param(bench_params, cmp_params, file_name):
        all_diffs = []
        bench_params_refined = refine_json_keys(bench_params)
        cmp_params_refined = refine_json_keys(cmp_params)

        for bench_param_name in bench_params_refined.keys():
            matched_cmp_param_name, matched_with = HyperparameterChecker._fuzzy_match_parameter(bench_param_name,
                                                                                                cmp_params_refined)
            matched_cmp_param_name = cmp_params_refined.get(matched_cmp_param_name)
            bench_param_name = bench_params_refined.get(bench_param_name)
            bench_param_value = to_str_if_number(bench_params[bench_param_name])
            if matched_cmp_param_name:
                cmp_param_value = to_str_if_number(cmp_params[matched_cmp_param_name])
                if bench_param_value != cmp_param_value:
                    all_diffs.append(
                        [file_name, bench_param_name, matched_cmp_param_name, bench_param_value, cmp_param_value,
                         matched_with, Const.CONFIG_CHECK_ERROR])
                del cmp_params[matched_cmp_param_name]
            else:
                all_diffs.append(
                    [file_name, bench_param_name, "Only in benchmark", bench_param_value, "", "",
                     Const.CONFIG_CHECK_WARNING])
        for cmp_param_name, cmp_param_value in cmp_params.items():
            all_diffs.append(
                [file_name, "Only in comparison", cmp_param_name, "", cmp_param_value, "", Const.CONFIG_CHECK_WARNING])
        all_diffs.sort()
        return all_diffs

    @staticmethod
    def apply_patches(fmk):
        try:
            from megatron import training

            def collect_hyperparameter_wrapper(func):
                def wrapper(*args, **kwargs):
                    global hyperparameters_dict
                    result = func(*args, **kwargs)
                    if not hyperparameters_dict:
                        hyperparameters_dict = result
                    return result
                return wrapper
            training.get_args = collect_hyperparameter_wrapper(training.get_args)
        except ImportError:
            config_checking_print("No megatron find.")
        except Exception as e:
            config_checking_print(f"Patch megatron method failed, detail:{str(e)}")

    @staticmethod
    def _fuzzy_match_parameter(param_name: str, available_params: Dict[str, Any]):
        """
        Fuzzy matches a parameter name against available parameter names using predefined
        mappings and string similarity.
        """
        if param_name in available_params:
            return param_name, Const.MATCH_MODE_NAME

        canonical_name = None
        for standard_name, aliases in parameter_name_mapping.items():
            if param_name == standard_name or param_name in aliases:
                canonical_name = standard_name
                break

        if canonical_name:
            if canonical_name in available_params:
                return canonical_name, Const.MATCH_MODE_MAPPING
            for alias in parameter_name_mapping[canonical_name]:
                if alias in available_params:
                    config_checking_print(
                        f"Matched '{param_name}' to alias '{alias}' via canonical name '{canonical_name}'")
                    return alias, Const.MATCH_MODE_MAPPING

        best_match_name = None
        best_match_ratio = 0.8
        for available_param_name in available_params:
            ratio = SequenceMatcher(None, param_name.lower(), available_param_name.lower()).ratio()
            if ratio > best_match_ratio:
                best_match_ratio = ratio
                best_match_name = available_param_name

        if best_match_name:
            config_checking_print(
                f"Fuzzy matched parameter '{param_name}' to '{best_match_name}' (similarity: {best_match_ratio:.2f})")
            return best_match_name, f"{Const.MATCH_MODE_SIMILARITY}:{best_match_ratio}"

        return None, None
