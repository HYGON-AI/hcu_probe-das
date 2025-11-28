# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tqdm import tqdm

from msprobe.core.common.const import Const, Data2DBConst
from msprobe.core.common.file_utils import (
    check_file_or_directory_path,
    create_directory,
    load_json,
    recursive_chmod,
    remove_path,
)
from msprobe.core.common.log import logger
from msprobe.core.dump.dump2db.db_utils import DumpDB


def validate_micro_step(micro_step):
    """
    校验 micro_step 值是否在有效范围内
    """
    if micro_step is not None:
        if type(micro_step) is not int:
            raise ValueError("Micro step must be an integer")
        if micro_step < Data2DBConst.MIN_MICRO_STEP or micro_step > Data2DBConst.MAX_MICRO_STEP:
            raise ValueError(f"Micro step must be between"
                             f"{Data2DBConst.MIN_MICRO_STEP} and {Data2DBConst.MAX_MICRO_STEP}")


def validate_step_partition(step_partition: int) -> None:
    if type(step_partition) is not int:
        raise TypeError("step_partition must be integer")
    if not Data2DBConst.MIN_PARTITION <= step_partition <= Data2DBConst.MAX_PARTITION:
        raise ValueError(
            f"step_partition must be between {Data2DBConst.MIN_PARTITION} ",
            f"and {Data2DBConst.MAX_PARTITION}, got {step_partition}"
        )


def load_mapping(mapping_path):
    if mapping_path and isinstance(mapping_path, str):
        return load_json(mapping_path)
    else:
        return {}


@dataclass
class TensorProcessingParams:
    """Tensor处理函数的参数载体"""
    tensor_data: Dict[str, Any]
    target_prefix: str
    vpp_stage: int
    micro_step: int
    table_name: str
    step: int
    rank: int
    metric_id: int
    metric_type: Optional[str]


class DumpRecordBuilder:
    def __init__(self, db: DumpDB, data_dir, mapping, micro_step):
        self.db = db
        self.step_partition = db.step_partition
        self.data_dir = data_dir
        self.mapping = mapping
        self.micro_step = micro_step if micro_step else None

    @staticmethod
    def extract_target_info(full_key):
        """
        从完整的key中提取target信息
        格式: Module.layerN.operation.M
        """
        vpp_stage = 0  # 暂不提取vpp信息
        micro_step = 0
        target_prefix = full_key

        parts = full_key.split(Const.SEP)
        if parts and parts[-1].isdigit():
            micro_step = int(parts[-1])
            # 重新构建target_prefix（去掉最后一个数字部分）
            target_prefix = Const.SEP.join(parts[:-1])

        return target_prefix, vpp_stage, micro_step

    @staticmethod
    def parse_tensor_target(metric_type, tensor_type, tensor_idx):
        """根据tensor类型和索引生成target后缀"""
        if metric_type in [Data2DBConst.FORWARD, Data2DBConst.RECOMPUTE]:
            if tensor_type == Const.INPUT_ARGS:
                return f".input.{tensor_idx}"
            elif tensor_type == Const.OUTPUT:
                return f".output.{tensor_idx}"
            elif tensor_type == Const.PARAMS:
                return f".parameters.{tensor_idx}"
        elif metric_type == Data2DBConst.BACKWARD:
            if tensor_type == Const.INPUT:
                return f".input.{tensor_idx}"
            elif tensor_type == Const.OUTPUT:
                return f".output.{tensor_idx}"
        return ""

    @staticmethod
    def process_tensor_value(value):
        """处理统计量值  将inf转换为float极值"""
        if value is None:
            return sys.float_info.max
        elif value == float("inf"):
            return sys.float_info.max - 1  # 最大float值
        elif value == float("-inf"):
            return sys.float_info.min + 1  # 最小float值
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return None

    def import_data(self):
        """导入数据"""
        logger.info("Scanning data directory...")

        # 扫描所有step和rank
        max_rank = 0
        min_step = float('inf')
        max_step = 0
        valid_ranks = defaultdict(list)

        # 首先扫描目录结构获取step和rank信息
        for step_dir in os.listdir(self.data_dir):
            if not step_dir.startswith('step'):
                continue
            try:
                step = int(step_dir[4:])  # 提取step数字
            except ValueError:
                continue
            min_step = min(min_step, step *
                           self.micro_step if self.micro_step else step)
            max_step = max(max_step, (step + 1) *
                           self.micro_step if self.micro_step else step)

            step_path = os.path.join(self.data_dir, step_dir)
            for rank_dir in os.listdir(step_path):
                if not rank_dir.startswith('rank'):
                    continue
                if rank_dir == "rank":
                    rank = 0
                else:
                    try:
                        rank = int(rank_dir[4:])  # 提取rank数字
                    except ValueError:
                        continue

                rank_path = os.path.join(step_path, rank_dir)
                json_path = os.path.join(rank_path, "dump.json")
                if os.path.exists(json_path):
                    valid_ranks[step].append((rank, json_path))
                    max_rank = max(max_rank, rank)

        if min_step == float('inf'):
            logger.warning(f"No valid step directories found in: {self.data_dir},"
                           f"looking for directories starting with 'step' (e.g., step0, step1")
            return

        # 更新全局统计信息
        self.db.update_global_stats(
            max_rank=max_rank, min_step=min_step, max_step=max_step)
        # 创建所有需要的分区表
        self.db.create_all_metric_tables(min_step, max_step)

        for step in tqdm(sorted(valid_ranks.keys()), desc="Processing steps"):
            step_dir = f"step{step}"
            step_path = os.path.join(self.data_dir, step_dir)

            check_file_or_directory_path(step_path, isdir=True)
            table_name_cache = self._get_step_table_name_cache(step)

            # 为当前step的所有有效rank创建进度条
            if not valid_ranks[step]:
                continue
            for rank, json_path in tqdm(valid_ranks[step], desc=f"Step {step} ranks", leave=False):
                self._process_dump_file(
                    json_path, table_name_cache, step, rank)
        self.db.extract_tags_from_processed_targets()

    def _get_step_table_name_cache(self, step):
        table_name_cache = defaultdict(dict)
        for metric_name in Data2DBConst.METRICS:
            metric_id = self.db.get_metric_id(metric_name)
            micro_step_part = self.micro_step if self.micro_step else 1
            for ms in range(micro_step_part):
                current_mstep = step * micro_step_part + ms
                partition_start = (
                    current_mstep // self.step_partition) * self.step_partition
                table_name = DumpDB.get_metric_table_name(metric_id, partition_start,
                                                          partition_start + self.step_partition - 1)
                table_name_cache[metric_id][current_mstep] = table_name
        return table_name_cache

    def _determine_metric_type(self, full_key, tensor_data):
        for key, value in self.mapping.items():
            full_key = full_key.replace(key, value)

        # 确定metric类型
        metric_type = None
        if f"{Const.SEP}{Const.FORWARD}{Const.SEP}" in full_key or \
                full_key.endswith(f"{Const.SEP}{Const.FORWARD}"):
            if tensor_data.get('is_recompute', False):
                metric_type = Data2DBConst.RECOMPUTE
            else:
                metric_type = Data2DBConst.FORWARD
            full_key = full_key.replace(
                f"{Const.SEP}{Const.FORWARD}{Const.SEP}", Const.SEP)  # module
            full_key = full_key.replace(
                f"{Const.SEP}{Const.FORWARD}", "")  # api
        elif f"{Const.SEP}{Const.BACKWARD}{Const.SEP}" in full_key or \
                full_key.endswith(f"{Const.SEP}{Const.BACKWARD}"):
            metric_type = Data2DBConst.BACKWARD
            full_key = full_key.replace(
                f"{Const.SEP}{Const.BACKWARD}{Const.SEP}", Const.SEP)
            full_key = full_key.replace(f"{Const.SEP}{Const.BACKWARD}", "")
        elif full_key.endswith(Const.PARAMS_GRAD):
            metric_type = Data2DBConst.PARAMETERS_GRAD
            full_key = full_key.replace(f"{Const.SEP}{Const.PARAMS_GRAD}", "")
        # fsdp
        elif len(full_key.split(Const.SEP)) >= 3:
            parts = full_key.split(Const.SEP)
            if parts[-2] == Const.PARAMS_GRAD:
                metric_type = Data2DBConst.PARAMETERS_GRAD
                full_key = Const.SEP.join(parts[:-2])

        return metric_type, full_key

    def _add_tensor_data(self, tensor, target_name, tensor_params: TensorProcessingParams, batch_data):
        """添加tensor数据方法"""

        cache_key = (target_name, tensor_params.vpp_stage,
                     tensor_params.micro_step)

        # 如果缓存中不存在，创建临时ID
        cache_id_dict = self.db.cache_targets(
            cache_key, tensor_params.metric_id)

        # 准备数据行, 这里id还是个临时id, 需要更新后读取, 第三个实际为{"id": 0}
        row_data = [tensor_params.rank, tensor_params.step, cache_id_dict]
        for stat in Data2DBConst.ORDERED_STAT:
            value = tensor.get(stat.capitalize(), None)
            row_data.append(DumpRecordBuilder.process_tensor_value(value))
        batch_data[tensor_params.table_name].append((row_data))

    def _process_forward_data(self, tensor_params: TensorProcessingParams, batch_data):
        """处理forward/recompute数据"""
        # 处理input_args
        if Const.INPUT_ARGS in tensor_params.tensor_data:
            for idx, tensor in enumerate(tensor_params.tensor_data[Const.INPUT_ARGS]):
                if isinstance(tensor, dict) and tensor.get(Const.TYPE) in Data2DBConst.SUPPORT_TYPE:
                    if tensor.get(Const.DTYPE) not in Data2DBConst.SUPPORT_DTYPE:
                        continue
                    target_suffix = DumpRecordBuilder.parse_tensor_target(
                        Data2DBConst.FORWARD, Const.INPUT_ARGS, idx)
                    target_name = tensor_params.target_prefix + target_suffix
                    self._add_tensor_data(
                        tensor, target_name, tensor_params, batch_data)

        # 处理output
        if Const.OUTPUT in tensor_params.tensor_data:
            for idx, tensor in enumerate(tensor_params.tensor_data[Const.OUTPUT]):
                if isinstance(tensor, dict) and tensor.get(Const.TYPE) in Data2DBConst.SUPPORT_TYPE:
                    if tensor.get(Const.DTYPE) not in Data2DBConst.SUPPORT_DTYPE:
                        continue
                    target_suffix = DumpRecordBuilder.parse_tensor_target(
                        Data2DBConst.FORWARD, Const.OUTPUT, idx)
                    target_name = tensor_params.target_prefix + target_suffix
                    self._add_tensor_data(
                        tensor, target_name, tensor_params, batch_data)

        # 处理parameters
        if Const.PARAMS in tensor_params.tensor_data:
            for param_name, param_tensors in tensor_params.tensor_data[Const.PARAMS].items():
                if not (isinstance(param_tensors, list) and param_tensors and isinstance(param_tensors[0], dict)):
                    continue
                if param_tensors[0].get(Const.TYPE) in Data2DBConst.SUPPORT_TYPE:
                    if param_tensors[0].get(Const.DTYPE) not in Data2DBConst.SUPPORT_DTYPE:
                        continue
                    target_suffix = DumpRecordBuilder.parse_tensor_target(
                        Data2DBConst.FORWARD, Const.PARAMS, param_name)
                    target_name = tensor_params.target_prefix + target_suffix
                    self._add_tensor_data(
                        param_tensors[0], target_name, tensor_params, batch_data)

    def _process_parameters_data(self, tensor_params: TensorProcessingParams, batch_data):
        """处理parameters数据"""
        for param_name, param_tensors in tensor_params.tensor_data.items():
            if not (isinstance(param_tensors, list) and param_tensors and isinstance(param_tensors[0], dict)):
                continue
            if param_tensors[0].get(Const.TYPE) in Data2DBConst.SUPPORT_TYPE:
                if param_tensors[0].get(Const.DTYPE) not in Data2DBConst.SUPPORT_DTYPE:
                    continue
                target_name = tensor_params.target_prefix + f".{param_name}"
                self._add_tensor_data(
                    param_tensors[0], target_name, tensor_params, batch_data)

    def _process_backward_data(self, tensor_params: TensorProcessingParams, batch_data):
        """处理backward数据"""
        # 处理input
        if Const.INPUT in tensor_params.tensor_data:
            for idx, tensor in enumerate(tensor_params.tensor_data[Const.INPUT]):
                if isinstance(tensor, dict) and tensor.get(Const.TYPE) in Data2DBConst.SUPPORT_TYPE:
                    if tensor.get(Const.DTYPE) not in Data2DBConst.SUPPORT_DTYPE:
                        continue
                    target_suffix = DumpRecordBuilder.parse_tensor_target(
                        Data2DBConst.BACKWARD, Const.INPUT, idx)
                    target_name = tensor_params.target_prefix + target_suffix
                    self._add_tensor_data(
                        tensor, target_name, tensor_params, batch_data)

        # 处理output
        if Const.OUTPUT in tensor_params.tensor_data:
            for idx, tensor in enumerate(tensor_params.tensor_data[Const.OUTPUT]):
                if isinstance(tensor, dict) and tensor.get(Const.TYPE) in Data2DBConst.SUPPORT_TYPE:
                    if tensor.get(Const.DTYPE) not in Data2DBConst.SUPPORT_DTYPE:
                        continue
                    target_suffix = DumpRecordBuilder.parse_tensor_target(
                        Data2DBConst.BACKWARD, Const.OUTPUT, idx)
                    target_name = tensor_params.target_prefix + target_suffix
                    self._add_tensor_data(
                        tensor, target_name, tensor_params, batch_data)

    def _process_dump_file(self, json_path, table_name_cache, step, rank):
        """处理单个dump.json文件"""
        data = load_json(json_path)
        if 'data' not in data or not isinstance(data['data'], dict):
            return

        batch_data = defaultdict(list)
        # 预先计算所有metric_type的table_name

        for i, (ori_key, tensor_data) in enumerate(data['data'].items()):
            if not isinstance(ori_key, str) or not isinstance(tensor_data, dict):
                continue
            metric_type, full_key = self._determine_metric_type(
                ori_key, tensor_data)
            if not metric_type:
                continue

            # 使用缓存获取metric_id和stats
            metric_id = self.db.get_metric_id(metric_type)
            target_prefix, vpp_stage, mstep = DumpRecordBuilder.extract_target_info(
                full_key)
            if self.micro_step:
                current_mstep = step * self.micro_step + mstep
                mstep = 0
            else:
                current_mstep = step
            # table_name_cache 是 defaultdict(dict) 实例
            table_name = table_name_cache[metric_id].get(current_mstep)
            if not table_name:
                logger.warning(
                    f"Key '{ori_key}': index exceeds micro_step range {self.micro_step}, record skipped.")
                continue

            tensor_params = TensorProcessingParams(
                tensor_data=tensor_data,
                target_prefix=target_prefix,
                vpp_stage=vpp_stage,
                micro_step=mstep,
                table_name=table_name,
                step=current_mstep,
                rank=rank,
                metric_id=metric_id,
                metric_type=metric_type
            )

            # 处理不同类型的tensor数据
            if metric_type in [Data2DBConst.FORWARD, Data2DBConst.RECOMPUTE]:
                self._process_forward_data(tensor_params, batch_data)
            elif metric_type == Data2DBConst.PARAMETERS_GRAD:
                self._process_parameters_data(tensor_params, batch_data)
            elif metric_type == Data2DBConst.BACKWARD:
                self._process_backward_data(tensor_params, batch_data)

            if i % Data2DBConst.BATCH_SIZE == 0 and i > 0:
                self.db.batch_insert_targets()
                self.db.batch_insert_data(batch_data)
                batch_data = defaultdict(list)

        self.db.batch_insert_targets()
        self.db.batch_insert_data(batch_data)


def _data2db_service_parser(parser):
    parser.add_argument('--db', type=str, required=True,
                        help='Path to SQLite database file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dump output directory')
    parser.add_argument('--mapping', type=str, default=None,
                        help='Path to optional JSON mapping file')
    parser.add_argument('--micro_step', type=int, default=None,
                        help='Specific micro step value to split data in one step (must be between 1 and 10000)')

    parser.add_argument('--step_partition', type=int, default=50,
                        help='Partition size by step (default: 50)')


def _data2db_command(args):
    data_path = args.data

    check_file_or_directory_path(data_path, isdir=True, is_strict=True)
    create_directory(args.db)
    db_path = os.path.join(args.db, "monitor_metrics.db")

    if os.path.exists(db_path):
        logger.warning(f"Existing path {db_path} will be recovered")
        remove_path(db_path)

    micro_step = args.micro_step
    step_partition = args.step_partition
    validate_micro_step(micro_step)
    validate_step_partition(step_partition)
    mapping = load_mapping(args.mapping)
    db = DumpDB(db_path, step_partition)
    builder = DumpRecordBuilder(
        db, data_path, mapping=mapping, micro_step=micro_step)
    builder.import_data()

    recursive_chmod(args.db)
