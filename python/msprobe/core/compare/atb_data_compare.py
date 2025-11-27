# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import array
import os
import time

import pandas as pd

try:
    import torch
except ImportError:
    torch_available = False
else:
    torch_available = True

from msprobe.core.common.const import Const, FileCheckConst, CompareConst
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.file_utils import (
    check_and_get_real_path,
    check_if_valid_dir_pattern_path,
    read_csv,
    save_excel,
    create_directory,
    get_file_content_bytes
)
from msprobe.core.common.log import logger
from msprobe.core.compare.npy_compare import get_error_flag_and_msg, compare_ops_apply


MIN_ROW_COUNT_OF_CSV = 2
COL_COUNT_OF_CSV = 15
ROW_NUMBER_OF_FIRST_STAT = 1
DEVICE_AND_PID_COL_NUM = 0
EXECUTION_COUNT_COL_NUM = 1
OP_NAME_COL_NUM = 2
OP_ID_COL_NUM = 4
IO_TYPE_COL_NUM = 5
TENSOR_INDEX_COL_NUM = 6
DTYPE_COL_NUM = 7
SHAPE_COL_NUM = 9
MAX_VALUE_COL_NUM = 10
MIN_VALUE_COL_NUM = 11
MEAN_VALUE_COL_NUM = 12
NORM_VALUE_COL_NUM = 13
TENSOR_PATH_COL_NUM = 14

ATTR_VERSION = "$Version"
ATTR_END = "$End"
ATTR_OBJECT_LENGTH = "$Object.Length"
ATTR_OBJECT_COUNT = "$Object.Count"
ATTR_OBJECT_PREFIX = "$Object."

COMMON_HEADER = [
    CompareConst.ATB_DATA_NAME, CompareConst.DEVICE_AND_PID,
    CompareConst.EXECUTION_COUNT, CompareConst.DATA_TYPE, CompareConst.DATA_SHAPE
]

COMMON_HEADER_STATS = [Const.MAX, Const.MIN, Const.MEAN, Const.NORM]

STATS_SPECIAL_HEADER = [
    'Max Diff', 'Min Diff', 'Mean Diff', 'Norm Diff',
    'Relative Err of Max(%)', 'Relative Err of Min(%)',
    'Relative Err of Mean(%)', 'Relative Err of Norm(%)'
]

TENSOR_SPECIAL_HEADER = [
    'Cosine', 'Euc Distance', 'Max Absolute Err', 'Max Relative Err',
    'One Thousandth Err Ratio', 'Five Thousandth Err Ratio'
]


def compare_atb_mode(args):
    """
    Entry point used by the CLI to trigger ATB model accuracy compare.
    """
    args.golden_path = check_and_get_real_path(args.golden_path, FileCheckConst.READ_ABLE)
    args.target_path = check_and_get_real_path(args.target_path, FileCheckConst.READ_ABLE)
    if os.path.isfile(args.golden_path) != os.path.isfile(args.target_path):
        logger.error('The golden_path and target_path should be both files or directories')
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR)
    check_if_valid_dir_pattern_path(args.output_path)
    args.output_path = check_and_get_real_path(args.output_path, FileCheckConst.WRITE_ABLE, must_exist=False)

    if os.path.isfile(args.golden_path):
        logger.warning('Comparison between ATB data files is not yet supported')
        return

    golden_stats_path = check_and_get_real_path(os.path.join(args.golden_path, 'statistic.csv'),
                                                FileCheckConst.READ_ABLE, file_type=FileCheckConst.CSV_SUFFIX)
    target_stats_path = check_and_get_real_path(os.path.join(args.target_path, 'statistic.csv'),
                                                FileCheckConst.READ_ABLE, file_type=FileCheckConst.CSV_SUFFIX)
    golden_csv_content = read_csv(golden_stats_path, as_pd=False)
    target_csv_content = read_csv(target_stats_path, as_pd=False)
    golden_comparison_mode = get_comparison_mode(golden_csv_content)
    target_comparison_mode = get_comparison_mode(target_csv_content)
    if (golden_comparison_mode != target_comparison_mode or
       golden_comparison_mode == CompareConst.UNKNOWN_COMPARISION_MODE):
        logger.error('Unvalid ATB dump data')
        return

    if golden_comparison_mode == CompareConst.TENSOR_COMPARISION_MODE and not torch_available:
        logger.error('Unable to compare ATB Tensor without torch. Please install with \"pip install torch\"')
        return

    golden_stats_map = get_stats_map(golden_csv_content, args.golden_path)
    target_stats_map = get_stats_map(target_csv_content, args.target_path)
    comparison_result_map = {}
    for col_name in COMMON_HEADER:
        comparison_result_map[f'Target {col_name}'] = []
        comparison_result_map[f'Golden {col_name}'] = []

    if golden_comparison_mode == CompareConst.STATS_COMPARISION_MODE:
        for col_name in STATS_SPECIAL_HEADER:
            comparison_result_map[col_name] = []
        for col_name in COMMON_HEADER_STATS:
            comparison_result_map[f'Target {col_name}'] = []
            comparison_result_map[f'Golden {col_name}'] = []
        cal_comparison_metrics_by_stats(golden_stats_map, target_stats_map, comparison_result_map)
    else:
        for col_name in TENSOR_SPECIAL_HEADER:
            comparison_result_map[col_name] = []
        for col_name in COMMON_HEADER_STATS:
            comparison_result_map[f'Target {col_name}'] = []
            comparison_result_map[f'Golden {col_name}'] = []
        cal_comparison_metrics_by_tensor(golden_stats_map, target_stats_map, comparison_result_map)

    if comparison_result_map.get(f'Target {CompareConst.ATB_DATA_NAME}'):
        create_directory(args.output_path)
        file_name = (f'atb_{golden_comparison_mode}_' +
                     f'compare_result_{time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))}.xlsx')
        file_path = os.path.join(args.output_path, file_name)
        save_excel(file_path, pd.DataFrame(comparison_result_map))
        logger.info(f'Complete ATB dump data comparison. Comparison result has been saved at {file_path}')


def get_comparison_mode(csv_content):
    if len(csv_content) < MIN_ROW_COUNT_OF_CSV or not csv_content[ROW_NUMBER_OF_FIRST_STAT]:
        return CompareConst.UNKNOWN_COMPARISION_MODE

    if csv_content[ROW_NUMBER_OF_FIRST_STAT][-1] == CompareConst.N_A:
        return CompareConst.STATS_COMPARISION_MODE

    return CompareConst.TENSOR_COMPARISION_MODE


def get_stats_map(csv_content, root_path):
    stats_map = dict()

    if len(csv_content) < MIN_ROW_COUNT_OF_CSV:
        return stats_map

    csv_content.pop(0)

    for tensor_stat in csv_content:
        if len(tensor_stat) != COL_COUNT_OF_CSV:
            continue

        single_names = tensor_stat[OP_NAME_COL_NUM].split(Const.SCOPE_SEPARATOR)
        single_ids = tensor_stat[OP_ID_COL_NUM].split(Const.REPLACEMENT_CHARACTER)
        if (len(single_names) != len(single_ids)):
            continue
        data_name = ''
        for index, name in enumerate(single_names):
            data_name += f'{single_ids[index]}_{name}{Const.SCOPE_SEPARATOR}'
        data_name += f'{tensor_stat[IO_TYPE_COL_NUM]}.{tensor_stat[TENSOR_INDEX_COL_NUM]}'

        tensor_path = tensor_stat[TENSOR_PATH_COL_NUM]
        if tensor_path != CompareConst.N_A:
            tensor_path = tensor_path[tensor_path.rfind('atb_dump_data'):]
            tensor_path_vec = tensor_path.split(Const.SCOPE_SEPARATOR)
            if len(tensor_path_vec) < 5:
                continue
            tensor_path = os.path.join(root_path, Const.SCOPE_SEPARATOR.join(tensor_path_vec[4:]))

        single_stat_map = {
            CompareConst.ATB_DATA_NAME: data_name,
            CompareConst.DEVICE_AND_PID: tensor_stat[DEVICE_AND_PID_COL_NUM],
            CompareConst.EXECUTION_COUNT: tensor_stat[EXECUTION_COUNT_COL_NUM],
            CompareConst.DATA_TYPE: tensor_stat[DTYPE_COL_NUM],
            CompareConst.DATA_SHAPE: tensor_stat[SHAPE_COL_NUM],
            Const.MAX: tensor_stat[MAX_VALUE_COL_NUM],
            Const.MIN: tensor_stat[MIN_VALUE_COL_NUM],
            Const.MEAN: tensor_stat[MEAN_VALUE_COL_NUM],
            Const.NORM: tensor_stat[NORM_VALUE_COL_NUM],
            CompareConst.TENSOR_PATH: tensor_path
        }

        stats_map[data_name] = single_stat_map

    return stats_map


def convert_str_to_float(float_str):
    float_value = 0
    try:
        float_value = float(float_str)
    except ValueError:
        return False, float_value

    return True, float_value


def convert_str_to_int(int_str, default_value=0):
    int_value = default_value
    try:
        int_value = int(int_str)
    except ValueError:
        return False, int_value

    return True, int_value


def cal_comparison_metrics_by_stats(golden_stats_map, target_stats_map, comparison_result_map):
    for data_name in target_stats_map:
        if data_name not in golden_stats_map:
            continue

        for col_name in COMMON_HEADER:
            comparison_result_map[f'Target {col_name}'].append(target_stats_map[data_name][col_name])
            comparison_result_map[f'Golden {col_name}'].append(golden_stats_map[data_name][col_name])

        for metric in Const.SUMMARY_METRICS_LIST:
            comparison_result_map[f'Target {metric}'].append(target_stats_map[data_name][metric])
            comparison_result_map[f'Golden {metric}'].append(golden_stats_map[data_name][metric])
            target_crt_ret, target_crt_value = convert_str_to_float(target_stats_map[data_name][metric])
            golden_crt_ret, golden_crt_value = convert_str_to_float(golden_stats_map[data_name][metric])
            if not (target_crt_ret and golden_crt_ret):
                comparison_result_map[f'{metric} Diff'].append(CompareConst.N_A)
                comparison_result_map[f'Relative Err of {metric}(%)'].append(CompareConst.N_A)
                continue

            diff_value = target_crt_value - golden_crt_value
            comparison_result_map[f'{metric} Diff'].append(str(diff_value))
            if diff_value == 0:
                comparison_result_map[f'Relative Err of {metric}(%)'].append('0.0')
                continue
            if golden_crt_value == 0:
                comparison_result_map[f'Relative Err of {metric}(%)'].append(CompareConst.N_A)
                continue
            comparison_result_map[f'Relative Err of {metric}(%)'].append(
                str(abs(diff_value / golden_crt_value * 100)))

    return comparison_result_map


def cal_comparison_metrics_by_tensor(golden_stats_map, target_stats_map, comparison_result_map):
    for data_name in target_stats_map:
        if data_name not in golden_stats_map:
            continue

        for col_name in COMMON_HEADER:
            comparison_result_map[f'Target {col_name}'].append(target_stats_map[data_name][col_name])
            comparison_result_map[f'Golden {col_name}'].append(golden_stats_map[data_name][col_name])

        target_bin = TensorBinFile(target_stats_map[data_name][CompareConst.TENSOR_PATH])
        golden_bin = TensorBinFile(golden_stats_map[data_name][CompareConst.TENSOR_PATH])
        target_tensor = target_bin.get_data()
        golden_tensor = golden_bin.get_data()
        target_tensor_stats = cal_single_tensor_stats(target_tensor, target_bin.is_valid)
        golden_tensor_stats = cal_single_tensor_stats(golden_tensor, golden_bin.is_valid)

        for col_name in COMMON_HEADER_STATS:
            comparison_result_map[f'Target {col_name}'].append(target_tensor_stats[col_name])
            comparison_result_map[f'Golden {col_name}'].append(golden_tensor_stats[col_name])

        comparison_metrics = [CompareConst.N_A] * len(TENSOR_SPECIAL_HEADER)
        if target_bin.is_valid and golden_bin.is_valid:
            comparison_metrics = compare_single_tensor(golden_tensor, target_tensor)

        for index, col_name in enumerate(TENSOR_SPECIAL_HEADER):
            comparison_result_map[col_name].append(comparison_metrics[index])

    return comparison_result_map


def cal_single_tensor_stats(tensor, is_valid=True):
    tensor_stats = {}
    for stat in COMMON_HEADER_STATS:
        tensor_stats[stat] = CompareConst.N_A

    if not is_valid or not tensor.numel() or not tensor.shape:
        return tensor_stats

    if tensor.dtype == torch.bool:
        tensor_stats[Const.MAX] = str(torch.any(tensor))
        tensor_stats[Const.MIN] = str(torch.all(tensor))
        return tensor_stats

    if torch.is_complex(tensor):
        mean_value = str(tensor.mean().item())
        norm_value = str(tensor.norm().item())
        tensor_stats[Const.MEAN] = CompareConst.N_A if mean_value == 'nan' else mean_value
        tensor_stats[Const.NORM] = CompareConst.N_A if norm_value == 'nan' else norm_value

    if not torch.is_complex(tensor) and (tensor.dtype == torch.float64 or not tensor.is_floating_point()):
        tensor = tensor.float()

    max_value = CompareConst.N_A
    min_value = CompareConst.N_A

    if not torch.is_complex(tensor):
        max_value = str(tensor.max().item())
        min_value = str(tensor.min().item())
    mean_value = str(tensor.mean().item())
    norm_value = str(tensor.norm().item())
    tensor_stats[Const.MAX] = CompareConst.N_A if max_value == 'nan' else max_value
    tensor_stats[Const.MIN] = CompareConst.N_A if min_value == 'nan' else min_value
    tensor_stats[Const.MEAN] = CompareConst.N_A if mean_value == 'nan' else mean_value
    tensor_stats[Const.NORM] = CompareConst.N_A if norm_value == 'nan' else norm_value

    return tensor_stats


def compare_single_tensor(golden_tensor, target_tensor):
    result_list = [CompareConst.N_A] * len(TENSOR_SPECIAL_HEADER)

    if golden_tensor.dtype == torch.bfloat16:
        golden_tensor = golden_tensor.to(torch.float32)
    golden_value = golden_tensor.numpy()
    if target_tensor.dtype == torch.bfloat16:
        target_tensor = target_tensor.to(torch.float32)
    target_value = target_tensor.numpy()

    target_value, golden_value, error_flag, _ = get_error_flag_and_msg(target_value, golden_value)
    if error_flag:
        return result_list

    result_list, _ = compare_ops_apply(target_value, golden_value, False, '')
    result_list = [str(ret) if str(ret) != CompareConst.NAN else CompareConst.N_A for ret in result_list]
    return result_list


class TensorBinFile:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.real_data_path = ''
        self.dtype = 0
        self.format = 0
        self.dims = []
        self.dtype_dict = {
            0: torch.float32,
            1: torch.float16,
            2: torch.int8,
            3: torch.int32,
            9: torch.int64,
            12: torch.bool,
            27: torch.bfloat16
        }
        self.is_valid = True
        self._parse_bin_file()

    def get_data(self):
        tensor = torch.zeros(1)
        if not self.is_valid or self.real_data_path:
            return tensor
        if self.dtype not in self.dtype_dict:
            logger.warning(f'Unsupported dtype: {self.dtype}')
            self.is_valid = False
            return tensor
        dtype = self.dtype_dict.get(self.dtype)
        try:
            tensor = torch.frombuffer(array.array('b', self.obj_buffer), dtype=dtype)
            tensor = tensor.view(self.dims)
        except Exception:
            logger.warning(f'Can not convert {self.file_path} to PyTorch Tensor')
            self.is_valid = False
            return torch.zeros(1)
        return tensor

    def _parse_bin_file(self):
        try:
            file_data = get_file_content_bytes(self.file_path)
        except FileCheckException:
            self.is_valid = False
            return

        begin_offset = 0
        for i, byte in enumerate(file_data):
            if byte != ord("\n"):
                continue
            line = file_data[begin_offset: i].decode("utf-8")
            begin_offset = i + 1
            fields = line.split("=")
            if len(fields) != 2:
                self.is_valid = False
                return

            attr_name = fields[0]
            attr_value = fields[1]
            if attr_name == ATTR_END:
                self.obj_buffer = file_data[i + 1:]
                return

            if not attr_name.startswith("$"):
                self._parse_user_attr(attr_name, attr_value)

    def _parse_user_attr(self, attr_name, attr_value):
        ret = True
        if attr_name == "data":
            self.real_data_path = attr_value
            ret = False
        elif attr_name == "dtype":
            ret, self.dtype = convert_str_to_int(attr_value, -1)
        elif attr_name == "format":
            ret, self.format = convert_str_to_int(attr_value, -1)
        elif attr_name == "dims":
            self.dims = attr_value.split(",")
            self.dims = [convert_str_to_int(dim, -1)[1] for dim in self.dims]
            if -1 in self.dims:
                ret = False
        if not ret:
            self.is_valid = False
