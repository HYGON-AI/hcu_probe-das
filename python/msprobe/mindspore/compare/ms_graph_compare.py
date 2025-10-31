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

import copy
import glob
import os
import re

import numpy as np
import pandas as pd
from msprobe.core.common.const import CompareConst, GraphMode, Const
from msprobe.core.common.file_utils import load_npy, read_csv, save_excel
from msprobe.core.common.log import logger
from msprobe.core.common.utils import add_time_with_xlsx, CompareException
from msprobe.core.compare.multiprocessing_compute import _ms_graph_handle_multi_process, check_accuracy
from msprobe.core.compare.npy_compare import npy_data_check, statistics_data_check, compare_ops_apply
from msprobe.mindspore.common.utils import convert_to_int, list_lowest_level_directories


class RowData:
    def __init__(self, mode):
        self.basic_data = copy.deepcopy(CompareConst.MS_GRAPH_BASE)
        self.npy_data = copy.deepcopy(CompareConst.MS_GRAPH_NPY)
        self.statistic_data = copy.deepcopy(CompareConst.MS_GRAPH_STATISTIC)
        self.csv = copy.deepcopy(CompareConst.MS_GRAPH_CSV)
        if mode == GraphMode.NPY_MODE:
            self.data = {**self.basic_data, **self.npy_data}
        else:
            self.data = {**self.basic_data, **self.statistic_data, **self.csv}

    def __call__(self):
        return self.data


def get_name_dict(name: str) -> dict:
    compare_pattern = re.compile(r'^([^.]+)\.([^.]+)\.([^.]+)\.([^.]+)\.(\d+(?:\.\d+)*)\.'
                                 r'((?:in|out)put(?:\.\d+)*)\.([^.]+)\.([^.]+)\.npy$')
    match = compare_pattern.match(name)
    if match:
        return {'op_type': match.group(1),
                'op_name': match.group(2),
                'task_id': match.group(3),
                'stream_id': match.group(4),
                'timestamp': match.group(5).split(Const.SEP)[0],
                'input_output_index': match.group(6),
                'slot': match.group(7),
                'format': match.group(8)}
    return {}


def npy_data_read(data_path, npy_file_list, mapping_dict):
    data_list = []
    compare_key_elements = ['op_name', 'task_id', 'input_output_index', 'slot']
    for data in npy_file_list:
        if data in mapping_dict:
            name_dict = get_name_dict(mapping_dict[data])
        else:
            name_dict = get_name_dict(data)
        if not name_dict:
            continue
        compare_key = Const.SEP.join([name_dict.get(element) for element in compare_key_elements])
        timestamp = convert_to_int(name_dict.get('timestamp'))

        data_list.append([os.path.join(data_path, data), compare_key, timestamp])
    return data_list


def statistic_data_read(statistic_file_list, statistic_file_path):
    data_list = []
    statistic_data_list = []
    header_index = {
        'Data Type': None, 'Shape': None,
        'Max Value': None, 'Min Value': None, 'Avg Value': None, 'L2Norm Value': None
    }
    for statistic_file in statistic_file_list:
        content = read_csv(statistic_file, as_pd=False)
        if not content:
            logger.error(f'Empty dump file: {statistic_file}')
            raise CompareException(f'Empty dump file: {statistic_file}')
        header = content[0]
        for index, value in enumerate(header):
            if value in header_index:
                header_index[value] = index
        statistic_data_list.extend(content[1:])

    for key in header_index.keys():
        if header_index[key] is None:
            logger.warning(f"Data_path {statistic_file_path} has no key {key}.")

    for data in statistic_data_list:
        '''
        13列分别是OpType, OpName, TaskId, StreamId, TimeStamp, IO, Slot, DataSize, 
        DataType, Shape, MaxValue, MinValue, L2NormValue
        '''
        if len(data) < 13:
            logger.error(f'Dump file {statistic_file_path} has been modified into incorrect format!')
            raise CompareException(f'Dump file {statistic_file_path} has been modified into incorrect format!')
        compare_key = f"{data[1]}.{data[2]}.{data[5]}.{data[6]}"  # OpName, TaskId, IO, Slot
        op_name = f"{compare_key}"
        timestamp = int(data[4])
        result_data = [op_name, compare_key, timestamp]
        for key in header_index.keys():
            if header_index[key] is None:
                result_data.append(np.nan)
            else:
                result_data.append(data[header_index[key]])
        csv_file = f"{statistic_file_path}"
        result_data.append(csv_file)
        data_list.append(result_data)
    return data_list


def generate_data_name(data_path):
    data_list = []

    mapping_path = os.path.join(data_path, "mapping.csv")
    statistic_path = os.path.join(data_path, "statistic.csv")
    npy_path = os.path.join(data_path, "*.npy")

    mapping_file_list = glob.glob(mapping_path)
    statistic_file_list = glob.glob(statistic_path)
    npy_file_list = glob.glob(npy_path)

    mapping_exist = bool(mapping_file_list)
    statistic_exist = bool(statistic_file_list)
    npy_exist = bool(npy_file_list)

    mapping_dict = {}
    if mapping_exist:
        for mapping_file in mapping_file_list:
            content = read_csv(mapping_file, False)
            for row in content[1:]:
                mapping_dict[row[0]] = row[1]

    if npy_exist:
        data_list = npy_data_read(data_path, npy_file_list, mapping_dict)

    elif statistic_exist:
        data_list = statistic_data_read(statistic_file_list, os.path.join(data_path, statistic_path))

    if npy_exist:
        mode = GraphMode.NPY_MODE
    elif statistic_exist:
        mode = GraphMode.STATISTIC_MODE
    else:
        mode = GraphMode.ERROR_MODE
        logger.error("Error mode.")
    return mode, data_list


def transform_special_string_into_float(data_frame):
    data_frame[data_frame == "null"] = '0'
    data_frame[data_frame == "False"] = '0'
    data_frame[data_frame == "True"] = '1'


class GraphMSComparator:
    def __init__(self, input_param, output_path):
        self.output_path = output_path
        self.base_npu_path = input_param.get('npu_path', None)
        self.base_bench_path = input_param.get('bench_path', None)
        rank_id_list = input_param.get('rank_id', [])
        step_id_list = input_param.get('step_id', [])
        if not isinstance(rank_id_list, list) or not isinstance(step_id_list, list):
            logger.error("'rank_id' and 'step_id' should both be lists, please check!")
            raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)
        self.rank_list = [convert_to_int(rank_id) for rank_id in rank_id_list]
        self.step_list = [convert_to_int(step_id) for step_id in step_id_list]
        # split by rank and step, generate rank step path
        self.npu_rank_step_dict = self.generate_rank_step_path(self.base_npu_path)
        self.bench_rank_step_dict = self.generate_rank_step_path(self.base_bench_path)
        self.common_rank_step = sorted(
            set(self.npu_rank_step_dict.keys()).intersection(self.bench_rank_step_dict.keys()))

    @staticmethod
    def compare_ops(compare_result_db, mode):

        def npy_mode_compute(row):
            result_dict = RowData(GraphMode.NPY_MODE)()

            def process_npy_file(file_path, name_prefix, result):
                if os.path.exists(file_path):
                    data = load_npy(file_path)
                    result[f'{name_prefix} Name'] = file_path
                    result[f'{name_prefix} Dtype'] = data.dtype
                    result[f'{name_prefix} Tensor Shape'] = data.shape
                    result[f'{name_prefix} max'] = np.max(data)
                    result[f'{name_prefix} min'] = np.min(data)
                    result[f'{name_prefix} mean'] = np.mean(data)
                    result[f'{name_prefix} l2norm'] = np.linalg.norm(data)
                    return data
                return ""

            n_value = process_npy_file(row[CompareConst.NPU_NAME], 'NPU', result_dict)
            b_value = process_npy_file(row[CompareConst.BENCH_NAME], 'Bench', result_dict)

            error_flag, error_message = npy_data_check(n_value, b_value)
            result_dict[CompareConst.ERROR_MESSAGE] = error_message

            if not error_flag:
                result_list, err_msg = compare_ops_apply(n_value, b_value, False, "")
                result_dict[CompareConst.COSINE] = result_list[0]
                result_dict[CompareConst.EUC_DIST] = result_list[1]
                result_dict[CompareConst.MAX_ABS_ERR] = result_list[2]
                result_dict[CompareConst.MAX_RELATIVE_ERR] = result_list[3]
                result_dict[CompareConst.ONE_THOUSANDTH_ERR_RATIO] = result_list[4]
                result_dict[CompareConst.FIVE_THOUSANDTHS_ERR_RATIO] = result_list[5]
                result_dict[CompareConst.ACCURACY] = check_accuracy(result_list[0], result_list[2])
                result_dict[CompareConst.ERROR_MESSAGE] = err_msg

            return pd.Series(result_dict)

        def statistic_mode_compute(row):
            result_dict = RowData('STATISTIC')()

            def update_result_dict(result, rows, prefix):
                result[f'{prefix} Name'] = rows[f'{prefix} Name']
                result[f'{prefix} Dtype'] = rows[f'{prefix} Dtype']
                result[f'{prefix} Tensor Shape'] = rows[f'{prefix} Tensor Shape']
                result[f'{prefix} max'] = np.float32(rows[f'{prefix} max'])
                result[f'{prefix} min'] = np.float32(rows[f'{prefix} min'])
                result[f'{prefix} mean'] = np.float32(rows[f'{prefix} mean'])
                result[f'{prefix} l2norm'] = np.float32(rows[f'{prefix} l2norm'])
                result[f'{prefix} CSV File'] = rows[f'{prefix} CSV File']

            def calculate_relative_error(numerator, denominator):
                """Calculates relative error, handling division by zero and NaN."""
                if denominator != 0:
                    result = numerator / denominator
                    if not np.isnan(result):
                        return str(abs(result * 100)) + "%"
                    else:
                        return CompareConst.NAN
                return CompareConst.N_A

            # 使用示例
            update_result_dict(result_dict, row, 'NPU')
            update_result_dict(result_dict, row, 'Bench')
            error_flag, error_message = statistics_data_check(result_dict)
            result_dict[CompareConst.ERROR_MESSAGE] += error_message
            if not error_flag:
                metrics = [
                    (CompareConst.MAX_DIFF, CompareConst.NPU_MAX, CompareConst.BENCH_MAX),
                    (CompareConst.MIN_DIFF, CompareConst.NPU_MIN, CompareConst.BENCH_MIN),
                    (CompareConst.MEAN_DIFF, CompareConst.NPU_MEAN, CompareConst.BENCH_MEAN),
                    (CompareConst.NORM_DIFF, CompareConst.NPU_NORM, CompareConst.BENCH_NORM),
                ]
                relative_error_metrics = [
                    (CompareConst.MAX_RELATIVE_ERR, CompareConst.MAX_DIFF, CompareConst.BENCH_MAX),
                    (CompareConst.MIN_RELATIVE_ERR, CompareConst.MIN_DIFF, CompareConst.BENCH_MIN),
                    (CompareConst.MEAN_RELATIVE_ERR, CompareConst.MEAN_DIFF, CompareConst.BENCH_MEAN),
                    (CompareConst.NORM_RELATIVE_ERR, CompareConst.NORM_DIFF, CompareConst.BENCH_NORM),
                ]

                for diff_metric, npu_metric, bench_metric in metrics:
                    result_dict[diff_metric] = result_dict[npu_metric] - result_dict[bench_metric]

                for rel_metric, diff_metric, bench_metric in relative_error_metrics:
                    result_dict[rel_metric] = calculate_relative_error(result_dict[diff_metric],
                                                                       result_dict[bench_metric])

                magnitude_diff = result_dict[CompareConst.MAX_DIFF] / (
                        max(result_dict[CompareConst.NPU_MAX], result_dict[CompareConst.BENCH_MAX]) + 1e-10)
                if np.isnan(result_dict[CompareConst.NPU_MAX]) and np.isnan(result_dict[CompareConst.BENCH_MAX]):
                    magnitude_diff = 0
                result_dict[CompareConst.ACCURACY] = CompareConst.YES if \
                    magnitude_diff <= CompareConst.MAGNITUDE else CompareConst.NO

            return pd.Series(result_dict)

        if mode == GraphMode.NPY_MODE:
            compare_result_db = compare_result_db.apply(npy_mode_compute, axis=1)
        else:
            compare_result_db = compare_result_db.apply(statistic_mode_compute, axis=1)
        return compare_result_db

    def compare_core(self):
        logger.info("Please check whether the input data belongs to you. If not, there may be security risks.")

        for rank_id, step_id in self.common_rank_step:
            compare_result_df, mode = self.compare_process(rank_id, step_id)
            if isinstance(compare_result_df, list):
                is_empty = not compare_result_df
            elif isinstance(compare_result_df, pd.DataFrame):
                is_empty = compare_result_df.empty
            else:
                is_empty = True
            if is_empty or not mode:
                continue
            compare_result_df = self.do_multi_process(compare_result_df, mode)
            compare_result_name = add_time_with_xlsx(f"compare_result_{str(rank_id)}_{str(step_id)}")
            compare_result_path = os.path.join(os.path.realpath(self.output_path), f"{compare_result_name}")
            save_excel(compare_result_path, compare_result_df)
            logger.info(f"Compare rank: {rank_id} step: {step_id} finish. Compare result: {compare_result_path}.")

    def compare_process(self, rank_id, step_id):
        # generate data_path
        npu_data_path_list = self.npu_rank_step_dict.get((rank_id, step_id))
        bench_data_path_list = self.bench_rank_step_dict.get((rank_id, step_id))
        if not npu_data_path_list or not npu_data_path_list:
            return [], ''

        # generate file name
        npu_mode = GraphMode.ERROR_MODE
        bench_mode = GraphMode.ERROR_MODE
        npu_data_list = []
        bench_data_list = []
        for npu_data_path in npu_data_path_list:
            npu_mode, data_list = generate_data_name(npu_data_path)
            npu_data_list.extend(data_list)
        for bench_data_path in bench_data_path_list:
            bench_mode, data_list = generate_data_name(bench_data_path)
            bench_data_list.extend(data_list)

        if npu_mode == GraphMode.ERROR_MODE or bench_mode == GraphMode.ERROR_MODE:
            logger.warning(f"Data path: npu_data_path or bench_data_path does not exist.")
            return [], ''
        if npu_mode != bench_mode:
            logger.error(f"NPU mode {npu_mode} not equal to MATCH mode {bench_mode}.")
            return [], ''

        if npu_mode == 'NPY_MODE':
            npu_data_df = pd.DataFrame(npu_data_list, columns=[CompareConst.NPU_NAME, 'Compare Key', 'TimeStamp'])
            bench_data_df = pd.DataFrame(bench_data_list, columns=[CompareConst.BENCH_NAME, 'Compare Key', 'TimeStamp'])
        else:
            npu_data_df = pd.DataFrame(npu_data_list,
                                       columns=[CompareConst.NPU_NAME, 'Compare Key', 'TimeStamp',
                                                CompareConst.NPU_DTYPE, CompareConst.NPU_SHAPE,
                                                CompareConst.NPU_MAX, CompareConst.NPU_MIN,
                                                CompareConst.NPU_MEAN, CompareConst.NPU_NORM,
                                                CompareConst.NPU_CSV_FILE])
            bench_data_df = pd.DataFrame(bench_data_list,
                                         columns=[CompareConst.BENCH_NAME, 'Compare Key', 'TimeStamp',
                                                  CompareConst.BENCH_DTYPE, CompareConst.BENCH_SHAPE,
                                                  CompareConst.BENCH_MAX, CompareConst.BENCH_MIN,
                                                  CompareConst.BENCH_MEAN, CompareConst.BENCH_NORM,
                                                  CompareConst.BENCH_CSV_FILE])

            npu_float_type = [CompareConst.NPU_MAX, CompareConst.NPU_MIN, CompareConst.NPU_MEAN, CompareConst.NPU_NORM]
            npu_float_data_df = npu_data_df[npu_float_type].astype(str)
            transform_special_string_into_float(npu_float_data_df)
            npu_data_df[npu_float_type] = npu_float_data_df.astype(float)

            bench_float_type = [
                CompareConst.BENCH_MAX, CompareConst.BENCH_MIN,
                CompareConst.BENCH_MEAN, CompareConst.BENCH_NORM
            ]
            bench_float_data_df = bench_data_df[bench_float_type].astype(str)
            transform_special_string_into_float(bench_float_data_df)
            bench_data_df[bench_float_type] = bench_float_data_df.astype(float)

        npu_data_df['Local Index'] = npu_data_df.sort_values('TimeStamp').groupby('Compare Key').cumcount()
        bench_data_df['Local Index'] = bench_data_df.sort_values('TimeStamp').groupby('Compare Key').cumcount()

        compare_result_df = pd.merge(npu_data_df, bench_data_df, on=['Compare Key', 'Local Index'], how='outer')

        compare_result_df[CompareConst.NPU_NAME] = compare_result_df[CompareConst.NPU_NAME].fillna('')
        compare_result_df[CompareConst.BENCH_NAME] = compare_result_df[CompareConst.BENCH_NAME].fillna('')

        return compare_result_df, npu_mode

    def generate_rank_step_path(self, base_path):

        def generate_rank_step_id(path_with_rank_step):
            split_path = path_with_rank_step.split("/")
            rank_id = -1
            if "rank_" in path_with_rank_step:
                # KBK mode
                if len(split_path) > 4:
                    rank_id = convert_to_int(split_path[-4].split("_")[-1])
                step_id = convert_to_int(split_path[-1])
            else:
                if len(split_path) > 4:
                    rank_id = convert_to_int(split_path[-4])
                if rank_id == -1 and len(split_path) > 3:
                    rank_id = convert_to_int(split_path[-3])
                step_id = convert_to_int(split_path[-1])
            return rank_id, step_id

        base_path = os.path.abspath(base_path)
        lowest_level = list_lowest_level_directories(base_path)

        rank_step_path_dict = {}
        for dir_path in lowest_level:
            rank_id, step_id = generate_rank_step_id(dir_path)
            if rank_id == -1 or step_id == -1:
                continue
            if self.rank_list and rank_id not in self.rank_list:
                continue
            if self.step_list and step_id not in self.step_list:
                continue
            rank_step_key = (rank_id, step_id)
            if rank_step_key in rank_step_path_dict:
                rank_step_path_dict[rank_step_key].append(dir_path)
            else:
                rank_step_path_dict[rank_step_key] = [dir_path]
        return dict(sorted(rank_step_path_dict.items()))

    def do_multi_process(self, result_df, mode):
        try:
            result_df = _ms_graph_handle_multi_process(self.compare_ops, result_df, mode)
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e
        return result_df
