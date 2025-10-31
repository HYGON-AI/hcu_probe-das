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
import re
import multiprocessing
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from msprobe.core.common.file_utils import check_file_or_directory_path, create_directory, load_npy, save_excel
from msprobe.core.common.log import logger
from msprobe.core.common.utils import check_process_num


@dataclass
class CompareResult:
    max_abs_error: float
    max_relative_error: float
    same_percentage: float
    first_mismatch_index: int
    percentage_within_thousandth: float
    percentage_within_hundredth: float


class SingleComparator:
    result_header = [
        'step', 
        'rank', 
        'micro_step', 
        'id', 
        'shape1', 
        'shape2', 
        '相同元素百分比(%)', 
        '首个不匹配元素索引', 
        '最大绝对误差', 
        '最大相对误差', 
        '误差在千分之一内元素占比(%)', 
        '误差在百分之一内元素占比(%)'
    ]

    @classmethod
    def compare(cls, dir1, dir2, output_path="./msprobe_compare_output", num_processes=8):
        data_dir1 = os.path.join(dir1, "data")
        data_dir2 = os.path.join(dir2, "data")
        check_file_or_directory_path(data_dir1, isdir=True)
        check_file_or_directory_path(data_dir2, isdir=True)
        check_process_num(num_processes)
        # 确保输出目录存在，如果不存在则创建
        if not os.path.exists(output_path):
            create_directory(output_path)
        cls.compare_data(data_dir1, data_dir2, output_path, num_processes)

    @classmethod
    def compare_arrays(cls, array1, array2) -> CompareResult:
        """
        比较两个NumPy数组，计算最大绝对误差、最大相对误差和相同元素的百分比
        """
        # 计算每个维度上的最小尺寸
        if array1.ndim != array2.ndim:
            array1 = array1.flatten()
            array2 = array2.flatten()
        min_shape = [min(s1, s2) for s1, s2 in zip(array1.shape, array2.shape)]
        # 截取数组到相同的形状
        sliced_array1 = array1[tuple(slice(0, s) for s in min_shape)]
        sliced_array2 = array2[tuple(slice(0, s) for s in min_shape)]

        abs_error = np.abs(sliced_array1 - sliced_array2)
        max_abs_error = np.max(abs_error)
        
        # 计算相对误差，处理分母为零的情况
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_error = np.abs(sliced_array1 - sliced_array2) / \
                np.maximum(np.abs(sliced_array1), np.abs(sliced_array2))
        relative_error = np.nan_to_num(relative_error)
        max_relative_error = np.max(relative_error)

        same_elements = np.sum(sliced_array1 == sliced_array2)
        total_elements = sliced_array1.size
        same_percentage = (same_elements / total_elements) * 100

        # 展平数组
        flat_array1 = sliced_array1.flatten()
        flat_array2 = sliced_array2.flatten()

        # 计算从第几个元素开始对不上
        mismatch_indices = np.nonzero(flat_array1 != flat_array2)[0]
        first_mismatch_index = mismatch_indices[0] if mismatch_indices.size > 0 else None

        # 计算误差在千分之一内的元素占比
        threshold = 0.001 * np.maximum(np.abs(sliced_array1), np.abs(sliced_array2))
        error_within_thousandth = np.sum(abs_error <= threshold)
        percentage_within_thousandth = (error_within_thousandth / total_elements) * 100

        # 计算误差在百分之一内的元素占比
        threshold = 0.01 * np.maximum(np.abs(sliced_array1), np.abs(sliced_array2))
        error_within_hundredth = np.sum(abs_error <= threshold)
        percentage_within_hundredth = (error_within_hundredth / total_elements) * 100

        return CompareResult(
            max_abs_error,
            max_relative_error,
            same_percentage,
            first_mismatch_index,
            percentage_within_thousandth,
            percentage_within_hundredth
        )

    @classmethod
    def get_steps(cls, tag_path):
        for step_folder in os.listdir(tag_path):
            if step_folder.startswith('step'):
                try:
                    step = int(step_folder[4:])
                except Exception as e:
                    raise RuntimeError(f"parse step number error") from e
                yield step, os.path.join(tag_path, step_folder)

    @classmethod
    def get_ranks(cls, step_path):
        for rank_folder in os.listdir(step_path):
            if rank_folder.startswith('rank'):
                try:
                    rank = int(rank_folder[4:])
                except Exception as e:
                    raise RuntimeError(f"parse rank number error") from e
                yield rank, os.path.join(step_path, rank_folder)

    @classmethod
    def get_micro_steps(cls, rank_path):
        for micro_step_folder in os.listdir(rank_path):
            if micro_step_folder.startswith('micro_step'):
                try:
                    micro_step = int(micro_step_folder[10:])
                except Exception as e:
                    raise RuntimeError(f"parse nicro_step number error") from e
                yield micro_step, os.path.join(rank_path, micro_step_folder)
            else:
                yield 0, rank_path

    @classmethod
    def get_arrays(cls, micro_step_path):
        for file in os.listdir(micro_step_path):
            if file.endswith('.npy'):
                try:
                    parts = file.rsplit('.', 2)
                    if len(parts) > 1 and parts[-2].isdigit():
                        array_id = int(parts[-2])
                    else:
                        array_id = 0
                except ValueError:
                    array_id = 0
                yield array_id, os.path.join(micro_step_path, file)

    @classmethod
    def get_array_paths(cls, dir_path):
        """
        获取目录中所有符合结构的NumPy数组文件路径
        """
        array_paths = {}
        if not os.path.exists(dir_path):
            return array_paths
        for tag in os.listdir(dir_path):
            tag_path = os.path.join(dir_path, tag)
            if not os.path.isdir(tag_path):
                continue
            for step, step_path in cls.get_steps(tag_path):
                for rank, rank_path in cls.get_ranks(step_path):
                    for item in os.listdir(rank_path):
                        next_path = os.path.join(rank_path, item)
                        if re.match(r"micro_step(\d+)", item):
                            micro_step = re.match(r"micro_step(\d+)", item).group(1)
                            for array_id, array_path in cls.get_arrays(next_path):
                                array_paths.setdefault(tag, []).append(
                                    (step, rank, int(micro_step), array_id, array_path))
                        elif re.match(r"\w{1,100}_(\d{1,100})\.npy", item):
                            array_id = re.match(r"\w{1,100}_(\d{1,100})\.npy", item).group(1)
                            array_paths.setdefault(tag, []).append((step, rank, 0, int(array_id), next_path))
                        else:
                            array_paths.setdefault(tag, []).append((step, rank, 0, 0, next_path))
        return array_paths

    @classmethod
    def compare_single_tag(cls, tag, array_paths1, array_paths2, output_dir):
        data = []
        paths1 = array_paths1.get(tag, [])
        paths2 = array_paths2.get(tag, [])
        path_dict1 = {(step, rank, micro_step, array_id): path for step, rank, micro_step, array_id, path in paths1}
        path_dict2 = {(step, rank, micro_step, array_id): path for step, rank, micro_step, array_id, path in paths2}
        common_keys = set(path_dict1.keys()) & set(path_dict2.keys())
        for key in common_keys:
            try:
                array1 = load_npy(path_dict1[key])
                array2 = load_npy(path_dict2[key])
                result = cls.compare_arrays(array1, array2)
                step, rank, micro_step, array_id = key
                data.append([
                    step, rank, micro_step, array_id,
                    list(array1.shape), list(array2.shape),
                    result.same_percentage,
                    result.first_mismatch_index,
                    result.max_abs_error,
                    result.max_relative_error,
                    result.percentage_within_thousandth,
                    result.percentage_within_hundredth
                ])
            except Exception as e:
                logger.error(f"Error comparing {path_dict1[key]} and {path_dict2[key]}: {e}")

        try:
            df = pd.DataFrame(data, columns=SingleComparator.result_header)
            df = df.sort_values(by=['step', 'rank', 'micro_step', 'id'])
            # 构建输出文件的完整路径
            output_file_path = os.path.join(output_dir, f'{tag}.xlsx')
            save_excel(output_file_path, df)
        except Exception as e:
            logger.error(f"Error processing tag {tag}: {e}")

    @classmethod
    def compare_data(cls, dir1, dir2, output_dir, num_processes=8):
        """
        比较两个目录中的NumPy数组文件，并将结果保存到指定目录的Excel文件中
        """

        array_paths1 = cls.get_array_paths(dir1)
        array_paths2 = cls.get_array_paths(dir2)

        all_tags = set(array_paths1.keys()) | set(array_paths2.keys())

        with multiprocessing.Pool(processes=num_processes) as pool:
            args = [(tag, array_paths1, array_paths2, output_dir) for tag in all_tags]
            try:
                results = pool.starmap_async(cls.compare_single_tag, args)
                with tqdm(total=len(all_tags), desc="Processing data") as pbar:
                    while not results.ready():
                        pbar.n = len(all_tags) - results._number_left
                        pbar.refresh()
                    results.wait()
                    results.get()
            except Exception as e:
                logger.error(f"Multiprocessing error: {e}")
            finally:
                pool.close()
                pool.join()
