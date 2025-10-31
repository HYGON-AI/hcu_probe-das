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
from typing import List

from tqdm import tqdm
import matplotlib.pyplot as plt

from msprobe.core.common.file_utils import create_directory, check_file_or_directory_path
from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import remove_path, load_npy, write_csv, read_csv
from msprobe.core.grad_probe.constant import GradConst
from msprobe.core.grad_probe.utils import plt_savefig


class GradComparator:

    @staticmethod
    def _get_grad_weight_order(path1, path2):
        for summary_file in os.listdir(path1):
            if not summary_file.endswith(".csv"):
                continue
            if not os.path.exists(os.path.join(path2, summary_file)):
                continue
            summary_csv = read_csv(os.path.join(path1, summary_file))
            return summary_csv["param_name"]
        raise RuntimeError("no matched grad_summary.csv for comparison, please dump data in same configuration")
    
    @staticmethod
    def _get_name_matched_grad_file(param_name, grad_files):
        for grad_file in grad_files:
            if param_name == grad_file[:grad_file.rfind('.')]:
                return grad_file
        raise RuntimeError("no matched grad_file for comparison, please dump data in same configuration")

    @classmethod
    def compare_distributed(cls, path1: str, path2: str, output_dir: str):
        check_file_or_directory_path(path1, isdir=True)
        check_file_or_directory_path(path2, isdir=True)
        ranks = cls._get_matched_dirs(path1, path2, "rank")
        logger.info(f"the following ranks will be compared: {ranks}")
        if not ranks:
            raise RuntimeError("no matched ranks for comparison, please dump data in same configuration")
        if not os.path.isdir(output_dir):
            create_directory(output_dir)
        for rank in tqdm(ranks, desc="rank"):
            logger.info(f"now comparing rank {rank}:")
            cls.compare(os.path.join(path1, f"rank{rank}"),
                        os.path.join(path2, f"rank{rank}"),
                        os.path.join(output_dir, f"rank{rank}"))

    @classmethod
    def compare(cls, path1: str, path2: str, output_dir: str):
        steps = cls._get_matched_dirs(path1, path2, "step")
        if not steps:
            raise RuntimeError("no matched steps for comparison, please dump data in same configuration")
        similarities = cls._calculate_separated_similarities(path1, path2, steps)
        if not os.path.isdir(output_dir):
            create_directory(output_dir)
        cls._save_similarities(similarities, steps, output_dir)

    @classmethod
    def _get_matched_dirs(cls, path1: str, path2: str, dir_prefix):
        check_file_or_directory_path(path1, isdir=True)
        check_file_or_directory_path(path2, isdir=True)
        dirs = []
        for dir_name in os.listdir(path1):
            index = dir_name.replace(dir_prefix, "", 1)
            if not dir_name.startswith(dir_prefix) or not index.isdigit():
                continue

            folder2 = os.path.join(path2, dir_name)
            if not os.path.isdir(folder2):
                continue
            dirs.append(int(index))
        dirs = sorted(dirs)
        return dirs

    @classmethod
    def _save_similarities(cls, similarities: List[float], steps: List[int], output_dir: str):
        if not similarities:
            raise ValueError(f"length of similarities is 0")
        result = [['step'] + [str(step) for step in steps]]
        for key, value in tqdm(similarities.items(), desc="save similarities (by param)"):
            if len(value) != len(steps):
                raise RuntimeError(f"similarities length of {key}:{len(value)} not equal steps:{len(steps)}")
            plt.plot(steps, value)
            plt.xlabel('steps')
            plt.ylabel('similarities')
            plt.title(f'{key}_similarities')
            picture_dir = os.path.join(output_dir, "similarities_picture")
            if not os.path.isdir(picture_dir):
                create_directory(picture_dir)
            fig_save_path = os.path.join(picture_dir, f"{key}_similarities.png")

            plt_savefig(fig_save_path)
            plt.close()

            result.append([key] + value)
        result_csv_path = os.path.join(output_dir, "similarities.csv")
        if os.path.exists(result_csv_path):
            logger.warning(f"{result_csv_path} will be deleted")
            remove_path(result_csv_path)
        write_csv(result, result_csv_path)

    @classmethod
    def _calculate_separated_similarities(cls, path1, path2, steps):
        similarities = {}
        logger.info(f"{len(steps)} steps will be compared")
        grad_weight_order = cls._get_grad_weight_order(path1, path2)
        for step in tqdm(steps, desc="calculate similarities (by step)"):
            grad_files = cls._get_matched_grad_files(path1, path2, step)
            same_count_summary = 0
            total_count_summary = 0
            for grad_name in grad_weight_order:
                grad_file = cls._get_name_matched_grad_file(grad_name, grad_files)
                grad1 = os.path.join(path1, f"step{step}", grad_file)
                grad2 = os.path.join(path2, f"step{step}", grad_file)
                same_count, total_count = cls._calculate_similarity(grad1, grad2)
                same_count_summary += same_count
                total_count_summary += total_count
                idx = grad_file.rfind(".")
                param_name = grad_file[:idx]
                if param_name not in similarities:
                    similarities[param_name] = []
                if total_count == 0:
                    similarities[param_name].append(0)
                else:
                    similarities[param_name].append(same_count / total_count)
            if GradConst.SUMMARY not in similarities:
                similarities[GradConst.SUMMARY] = []
            if total_count_summary == 0:
                similarities[GradConst.SUMMARY].append(0)
            else:
                similarities[GradConst.SUMMARY].append(same_count_summary / total_count_summary)
        return similarities

    @classmethod
    def _get_matched_grad_files(cls, path1: str, path2: str, step: int):
        path1 = os.path.join(path1, f"step{step}")
        path2 = os.path.join(path2, f"step{step}")
        check_file_or_directory_path(path1, isdir=True)
        check_file_or_directory_path(path2, isdir=True)
        grad_files = []
        for grad_file in os.listdir(path1):
            splits = grad_file.split('.')
            if len(splits) < 1 or splits[-1] not in GradConst.GRAD_FILE_SUFFIX:
                continue
            folder2 = os.path.join(path2, grad_file)
            if not os.path.exists(folder2):
                continue
            grad_files.append(grad_file)
        return sorted(grad_files)

    @classmethod
    def _calculate_similarity(cls, grad_file1: str, grad_file2: str):
        npy1, npy2 = cls._load_grad_files(grad_file1, grad_file2)
        same_count = (npy1 == npy2).sum()
        total_count = npy1.size
        return same_count, total_count

    @classmethod
    def _load_grad_files(cls, grad_file1: str, grad_file2: str):
        grad1 = load_npy(grad_file1)
        grad2 = load_npy(grad_file2)
        if grad1.shape != grad2.shape:
            raise RuntimeError(f"tensor shape is not equal: {grad_file1}, {grad_file2}")
        if grad1.dtype != bool:
            raise TypeError(f"tensor type is not bool: {grad_file1}")
        if grad2.dtype != bool:
            raise TypeError(f"tensor type is not bool: {grad_file2}")
        return grad1, grad2


