# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import subprocess
import argparse
from glob import glob
from time import time
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import numpy as np

from msprobe.infer.utils.log import logger, msg_filter
from msprobe.infer.utils.constants import JSON_FILE_MAX_SIZE
from msprobe.infer.utils.file_open_check import ms_open
from msprobe.infer.utils.check.rule import Rule
from msprobe.infer.utils.util import filter_cmd
from msprobe.infer.utils.file_open_check import FileStat


@dataclass
class FusionParams:
    fusion_after_data_dir: str
    fusion_before_data_dir: str
    fusion_after_model: str
    fusion_before_model: str


def execute_command(cmd: list):
    if not isinstance(cmd, list):
        raise TypeError(f"Expected list, but got {type(cmd).__name__}.")
    cmd = filter_cmd(cmd)
    logger.info("[Run CMD]: %r." % ' '.join(cmd))
    subprocess.run(cmd, shell=False)
    return


def get_pass(params: FusionParams, output_path: str, fusion_node_switch: bool):
    fusion_after_data_dir = params.fusion_after_data_dir
    fusion_before_data_dir = params.fusion_before_data_dir
    fusion_after_model = params.fusion_after_model
    fusion_before_model = params.fusion_before_model
    formatted_date = datetime.fromtimestamp(int(time())).strftime("%Y%m%d_%H%M%S")
    msit_dir = os.path.join(output_path, f"msit_{formatted_date}")
    os.makedirs(msit_dir, mode=0o750)
    logger.info('Created directory %r.' % msit_dir)
    run_compare(fusion_after_data_dir, fusion_before_data_dir, fusion_after_model, fusion_before_model, msit_dir)
    try:
        cmp_res_file = glob(os.path.join(msit_dir, "result_*"))[0]
    except IndexError as e:
        raise FileNotFoundError(f"No result file found in directory: {msit_dir}.") from e
    logger.info(f"Result of the comparison: {cmp_res_file}.")
    out_csv_path = os.path.join(msit_dir, "fusion_pass_info.csv")
    fusion_pass_analysis(cmp_res_file, fusion_after_model, out_csv_path, fusion_node_switch)


def run_compare(fusion_after_data_dir, fusion_before_data_dir, fusion_after_model, fusion_before_model, output_dir):
    cann_path = os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
    msaccucmp = os.path.join(cann_path, "tools", "operator_cmp", "compare", "msaccucmp.py")
    cmd = ["python3", msaccucmp, "compare", "-m", fusion_after_data_dir, "-g", fusion_before_data_dir, "-f", \
           fusion_after_model, "-cf", fusion_before_model, "-out", output_dir]
    execute_command(cmd)


def fusion_pass_analysis(csv_result_path, fusion_json_path, out_csv_path, fusion_node_switch):
    op_info_dict = read_fusion_pass_from_json(fusion_json_path)
    if fusion_node_switch:
        write_fusion_to_csv(csv_result_path, op_info_dict, out_csv_path)
    else:
        write_output_to_csv(csv_result_path, op_info_dict, out_csv_path)


def get_single_op_info_from_op_list(op_list):
    op_info_dict = {}
    for op in op_list:
        pass_name = set()
        op_name = op.get("name", None)
        if not op_name:
            continue
        for attr in op['attr']:
            if attr['key'] == 'pass_name':
                pass_name.update(attr['value']['list']['s'])
            if attr['key'] == 'pass_name_ub':
                pass_name.add(attr['value']['s'])
        if pass_name:
            op_info_dict[op_name] = pass_name
    return op_info_dict


def read_fusion_pass_from_json(fusion_json_path):
    try:
        with ms_open(fusion_json_path, max_size=JSON_FILE_MAX_SIZE) as f:
            ge_fusion_data = json.load(f)
    except Exception as e:
        logger.error(f'load json failed, err:{e}')
        return {}

    graph = ge_fusion_data.get("graph")
    op_info_dict = {}
    for sub_graph in graph:
        op_list = sub_graph.get("op")
        sub_info_dict = get_single_op_info_from_op_list(op_list)
        op_info_dict.update(sub_info_dict)
    return op_info_dict


def write_fusion_to_csv(file_path, op_info_dict, out_csv):
    try:
        Rule.input_file().check(file_path, will_raise=True)
        df = pd.read_csv(file_path, keep_default_na=False)
    except Exception as e:
        logger.error(f'load input csv failed, err:{e}')
        return
        
    df = df[df['TensorIndex'].str.contains('output', na=False)]
    df_selected = df.drop(columns=['Index', 'Address.1', 'DataType.1', 'CompareFailReason'])
    filtered_rows = []
    for _, row in df_selected.iterrows():
        if row['NPUDump'] in op_info_dict.keys():
            row['PassName'] = op_info_dict[row['NPUDump']]
            filtered_rows.append(row)
    filtered_df = pd.DataFrame(filtered_rows)
    filtered_df['MatchError'] = np.where(filtered_df['CosineSimilarity'] == 'NaN', "Fusion node not match", "")
    filtered_df.to_csv(out_csv, index=False)
    return


def write_output_to_csv(file_path, op_info_dict, out_csv):
    try:
        Rule.input_file().check(file_path, will_raise=True)
        df = pd.read_csv(file_path, keep_default_na=False)
    except Exception as e:
        logger.error(f'load input csv failed, err:{e}')
        return

    df = df[df['TensorIndex'].str.contains('output', na=False)]
    df['PassName'] = df['NPUDump'].apply(lambda x: op_info_dict.get(x, None))
    df['MatchError'] = np.where(df['CosineSimilarity'] == 'NaN', "Node not match", "")
    df = df.drop(columns=['Index', 'Address.1', 'DataType.1', 'CompareFailReason'])
    df.to_csv(out_csv, index=False)
    return


def check_output_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except FileNotFoundError as ffe:
        raise argparse.ArgumentTypeError("output path %r does not exist." % msg_filter(path_value)) from ffe
    except PermissionError as pe:
        raise argparse.ArgumentTypeError("permission denied for output path %r." % msg_filter(path_value)) from pe
    except Exception as err:
        raise argparse.ArgumentTypeError(
            "an unexpected error occurred while checking the output path %r." % msg_filter(path_value)
            ) from err
    if not file_stat.is_basically_legal("write", strict_permission=False):
        raise argparse.ArgumentTypeError("output path %r cannot be written to." % msg_filter(path_value))
    return path_value


def check_input_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except FileNotFoundError as ffe:
        raise argparse.ArgumentTypeError("input path %r does not exist." % msg_filter(path_value)) from ffe
    except PermissionError as pe:
        raise argparse.ArgumentTypeError("permission denied for input path %r." % msg_filter(path_value)) from pe
    except Exception as err:
        raise argparse.ArgumentTypeError(
            "an unexpected error occurred while checking the input path %r." % msg_filter(path_value)
            ) from err
    if not file_stat.is_basically_legal('read', strict_permission=False):
        raise argparse.ArgumentTypeError("input path %r cannot be read." % msg_filter(path_value))
    return path_value


def main():
    parser = argparse.ArgumentParser(description="Compare precision data for fusion operators")
    parser.add_argument("-m", "--fusion-after-dir", dest="fusion_after_dir", type=check_input_path_legality,
                        help="Directory for precision data when fusion switch is enabled.", required=True)
    parser.add_argument("-g", "--fusion-before-dir", dest="fusion_before_dir", type=check_input_path_legality,
                        help="Directory for precision data when fusion switch is disabled.", required=True)
    parser.add_argument("-f", "--fusion-after-model", dest="fusion_after_model", type=check_input_path_legality, 
                        help="JSON file for the model with fusion enabled.", required=True)
    parser.add_argument("-cf", "--fusion-before-model", dest="fusion_before_model", type=check_input_path_legality, 
                        help="JSON file for the model with fusion disabled.", required=True)
    parser.add_argument("-o", "--output-path", dest="output_path", type=check_output_path_legality, 
                        help="Directory for output files.", default="./")
    parser.add_argument("-fn", "--fusion-node-switch", dest="fusion_node_switch", action="store_false",
                        default=True, help="Whether to output only fusion operators (default: True).")
    args = parser.parse_args()
    params = FusionParams(
    fusion_after_data_dir=args.fusion_after_dir,
    fusion_before_data_dir=args.fusion_before_dir,
    fusion_after_model=args.fusion_after_model,
    fusion_before_model=args.fusion_before_model
    )
    get_pass(params, args.output_path, args.fusion_node_switch)


if __name__ == "__main__":
    main()
