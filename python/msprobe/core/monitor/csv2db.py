# Copyright (c) 2025-2026, Huawei Technologies Co., Ltd.
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

import datetime
import os
import re
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytz
from msprobe.core.common.const import MonitorConst
from msprobe.core.common.file_utils import (create_directory, read_csv,
                                            recursive_chmod, remove_path)
from msprobe.core.common.log import logger
from msprobe.core.common.utils import is_int
from msprobe.core.monitor.db_utils import MonitorDB, update_ordered_dict
from msprobe.core.monitor.utils import get_target_output_dir
from tqdm import tqdm

# Constants
all_data_type_list = [
    "actv", "actv_grad", "exp_avg", "exp_avg_sq",
    "grad_unreduced", "grad_reduced", "param_origin", "param_updated", "other"
]



@dataclass
class CSV2DBConfig:
    """Configuration for CSV to database conversion"""
    monitor_path: str
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    process_num: int = 1
    data_type_list: Optional[List[str]] = None
    output_dirpath: Optional[str] = None
    step_partition: int = 500


def validate_process_num(process_num: int) -> None:
    """Validate process number parameter"""
    if not is_int(process_num) or process_num <= 0:
        raise ValueError("process_num must be a positive integer")
    if process_num > MonitorConst.MAX_PROCESS_NUM:
        raise ValueError(f"Maximum supported process_num is {MonitorConst.MAX_PROCESS_NUM}")


def validate_step_partition(step_partition: int) -> None:
    if not isinstance(step_partition, int):
        raise TypeError("step_partition must be integer")
    if not MonitorConst.MIN_PARTITION <= step_partition <= MonitorConst.MAX_PARTITION:
        raise ValueError(
            f"step_partition must be between {MonitorConst.MIN_PARTITION} ",
            f"and {MonitorConst.MAX_PARTITION}, got {step_partition}"
        )


def validate_data_type_list(data_type_list: Optional[List[str]]) -> None:
    """Validate data type list parameter"""
    if data_type_list is None or not data_type_list:
        logger.info(f"Using default data types: {all_data_type_list}")
        return

    if not isinstance(data_type_list, list):
        raise ValueError("data_type_list must be a list")

    invalid_types = [t for t in data_type_list if t not in all_data_type_list]
    if invalid_types:
        raise ValueError(f"Unsupported data types: {invalid_types}")


def get_info_from_filename(file_name, metric_list=None):
    metric_name = "_".join(file_name.split('_')[:-1])
    if metric_list and metric_name not in metric_list:
        return "", 0, 0
    match = re.match(f"{metric_name}{MonitorConst.CSV_FILE_PATTERN}", file_name)
    if not match:
        return "", 0, 0
    step_start, step_end = match.groups()
    return metric_name, step_start, step_end


def _pre_scan_single_rank(rank: int, files: List[str]) -> Dict:
    """Pre-scan files for a single rank to collect metadata"""
    metrics = set()
    min_step = None
    max_step = 0
    metric_stats = defaultdict(set)
    targets = OrderedDict()

    for file_path in files:
        file_name = os.path.basename(file_path)
        metric_name, step_start, step_end = get_info_from_filename(file_name)
        if not metric_name:
            continue
        step_start, step_end = int(step_start), int(step_end)

        metrics.add(metric_name)
        min_step = min(
            step_start if min_step is None else min_step, step_start)
        max_step = max(max_step, step_end)

        data = read_csv(file_path)
        stats = [k for k in data.keys() if k in MonitorConst.OP_MONVIS_SUPPORTED]
        metric_stats[metric_name].update(stats)

        for row_id, row in data.iterrows():
            try:
                name = row[MonitorConst.HEADER_NAME]
                vpp_stage = int(row['vpp_stage'])
                micro_step = int(row.get('micro_step', MonitorConst.DEFAULT_INT_VALUE))
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"CSV conversion failed | file={file_path}:{row_id+2} | error={str(e)}")
                continue
            target = (name, vpp_stage, micro_step)
            if target not in targets:
                targets[target] = None

    return {
        'max_rank': int(rank),
        'metrics': metrics,
        'min_step': min_step,
        'max_step': max_step,
        'metric_stats': metric_stats,
        'targets': list(targets.keys())
    }


def _pre_scan(monitor_db: MonitorDB, data_dirs: Dict[int, str], data_type_list: List[str], workers: int = 1):
    """Pre-scan all targets, metrics, and statistics"""
    logger.info("Scanning dimensions...")
    rank_files = defaultdict(list)

    # Collect files for each rank
    for rank, dir_path in data_dirs.items():
        files = os.listdir(dir_path)
        for file in files:
            metric_name, _, _ = get_info_from_filename(
                file, metric_list=data_type_list)
            if not metric_name:
                continue
            rank_files[rank].append(os.path.join(dir_path, file))

    # Parallel pre-scan
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_pre_scan_single_rank, rank, files): rank
            for rank, files in rank_files.items()
        }

        results = []
        with tqdm(total=len(futures), desc="Pre-scanning ranks") as pbar:
            for future in as_completed(futures):
                rank = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Error pre-scanning rank {rank}: {str(e)}")
                pbar.update(1)

    # Aggregate results
    targets = OrderedDict()
    metrics = set()
    min_step = None
    max_step = 0
    max_rank = 0
    metric_stats = defaultdict(set)

    for rank_result in results:
        max_rank = max(max_rank, rank_result['max_rank'])
        metrics.update(rank_result['metrics'])
        min_step = min(
            min_step if min_step is not None else rank_result['min_step'],
            rank_result['min_step']
        )
        max_step = max(max_step, rank_result['max_step'])

        for metric, stats in rank_result['metric_stats'].items():
            metric_stats[metric].update(stats)

        targets = update_ordered_dict(targets, rank_result['targets'])

    monitor_db.insert_dimensions(
        targets, metrics, metric_stats, min_step=min_step, max_step=max_step)
    monitor_db.update_global_stats(
        max_rank=max_rank, min_step=min_step, max_step=max_step)
    return rank_files


def process_single_rank(
    task: Tuple[int, List[str]],
    metric_id_dict: Dict[str, Tuple[int, List[str]]],
    target_dict: Dict[Tuple[str, int, int], int],
    step_partition_size: int,
    db_path: str
) -> int:
    """Process data import for a single rank"""
    rank, files = task
    db = MonitorDB(db_path, step_partition_size=step_partition_size)
    total_inserted = 0
    table_batches = defaultdict(list)

    for file in files:
        filename = os.path.basename(file)
        metric_name, _, _ = get_info_from_filename(filename)
        if not metric_name:
            continue
        metric_info = metric_id_dict.get(metric_name)
        if not metric_info:
            continue

        metric_id, stats = metric_info

        for row_id, row in read_csv(file).iterrows():
            try:
                # Parse row data
                name = row.get(MonitorConst.HEADER_NAME)
                vpp_stage = int(row['vpp_stage'])
                micro_step = int(row.get('micro_step', MonitorConst.DEFAULT_INT_VALUE))
                target_id = target_dict.get((name, vpp_stage, micro_step))
                if not target_id:
                    continue

                step = int(row['step'])
                table_name, _, _ = db.get_metric_table_name(metric_id, step)
                # Prepare row data
                row_data = [rank, step, target_id]
                row_data.extend(
                    float(row[stat]) if stat in row else None
                    for stat in stats
                )
            except (ValueError, KeyError) as e:
                logger.error(
                    f"CSV conversion failed | file={file}:{row_id+2} | error={str(e)}")
                continue

            table_batches[table_name].append(tuple(row_data))
            # Batch insert when threshold reached
            if len(table_batches[table_name]) >= MonitorConst.BATCH_SIZE:
                inserted = db.insert_rows(
                    table_name, table_batches[table_name])
                if inserted is not None:
                    total_inserted += inserted
                table_batches[table_name] = []

    # Insert remaining data
    for table_name, batch in table_batches.items():
        if batch:
            inserted = db.insert_rows(table_name, batch)
            if inserted is not None:
                total_inserted += inserted

    logger.info(f"Rank {rank} inserted {total_inserted} rows")
    return total_inserted


def import_data(monitor_db: MonitorDB, data_dirs: Dict[int, str], data_type_list: List[str], workers: int = 4) -> bool:
    """Main method to import data into database"""
    # 1. Pre-scan to get rank tasks
    monitor_db.init_schema()
    rank_tasks = _pre_scan(monitor_db, data_dirs, data_type_list, workers)
    if not rank_tasks:
        logger.error("No valid data files found during pre-scan")
        return False

    # 2. Get metric and target mappings
    try:
        metric_id_dict = monitor_db.get_metric_mapping()
        target_dict = monitor_db.get_target_mapping()
    except Exception as e:
        logger.error(f"Failed to get database mappings: {str(e)}")
        return False

    # 3. Process data for each rank in parallel
    total_files = sum(len(files) for files in rank_tasks.values())
    logger.info(f"Starting data import for {len(rank_tasks)} ranks,"
                f"{total_files} files..."
                )
    all_succeeded = True
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_single_rank,
                (rank, files),
                metric_id_dict,
                target_dict,
                monitor_db.step_partition_size,
                monitor_db.db_path): rank
            for rank, files in rank_tasks.items()
        }

        with tqdm(as_completed(futures), total=len(futures), desc="Import progress") as pbar:
            for future in pbar:
                rank = futures[future]
                try:
                    inserted = future.result()
                    pbar.set_postfix_str(
                        f"Rank {rank}: inserted {inserted} rows")
                except Exception as e:
                    logger.error(
                        f"Failed to process Rank {rank}: {str(e)}")
                    all_succeeded = False
    return all_succeeded


def csv2db(config: CSV2DBConfig) -> None:
    """Main function to convert CSV files to database"""
    validate_process_num(config.process_num)
    validate_step_partition(config.step_partition)
    validate_data_type_list(config.data_type_list)

    target_output_dirs = get_target_output_dir(
        config.monitor_path, config.time_start, config.time_end)

    if config.output_dirpath is None:
        local_tz = pytz.timezone("Asia/Shanghai")
        cur_time = datetime.datetime.now(local_tz).strftime("%b%d_%H-%M-%S")
        config.output_dirpath = os.path.join(
            config.monitor_path, f"{cur_time}-csv2db")

    create_directory(config.output_dirpath)
    db_path = os.path.join(config.output_dirpath, "monitor_metrics.db")

    if os.path.exists(db_path):
        remove_path(db_path)
        logger.warning(f"Existing path {db_path} will be recovered")

    db = MonitorDB(db_path, step_partition_size=config.step_partition)

    result = import_data(
        db,
        target_output_dirs,
        config.data_type_list if config.data_type_list else all_data_type_list,
        workers=config.process_num
    )
    recursive_chmod(config.output_dirpath)
    if result:
        logger.info(
            f"Data import completed. Output saved to: {config.output_dirpath}")
    else:
        logger.warning(
            f"Data import may be incomplete. Output directory: {config.output_dirpath} "
            f"(Some records might have failed)"
        )
