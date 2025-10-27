# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

from python.msprobe.core.common.file_utils import check_file_type, load_json, check_file_or_directory_path
from python.msprobe.core.common.const import FileCheckConst, Const
from python.msprobe.core.common.utils import CompareException
from python.msprobe.core.common.log import logger
from python.msprobe.core.compare.utils import get_paired_dirs


def compare_cli(args, depth=1):
    if depth > 2:
        logger.error("Recursive compare error, depth exceeds 2.")
        raise CompareException(CompareException.RECURSION_LIMIT_ERROR)

    if isinstance(args.input_path, dict):  # special for dyn-graph mix compare
        input_param = args.input_path
    else:
        input_param = load_json(args.input_path)
    if not isinstance(input_param, dict):
        logger.error("input_param should be dict, please check!")
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)

    npu_path = input_param.get("npu_path", None)
    bench_path = input_param.get("bench_path", None)
    if not npu_path:
        logger.error(f"Missing npu_path in input configuration file, please check!")
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    if not bench_path:
        logger.error(f"Missing bench_path in input configuration file, please check!")
        raise CompareException(CompareException.INVALID_PATH_ERROR)

    frame_name = args.framework
    auto_analyze = not args.compare_only

    if frame_name == Const.PT_FRAMEWORK:
        from python.msprobe.pytorch.compare.pt_compare import compare
        from python.msprobe.pytorch.compare.distributed_compare import compare_distributed
    else:
        from python.msprobe.mindspore.compare.ms_compare import ms_compare
        from python.msprobe.mindspore.compare.distributed_compare import ms_compare_distributed, ms_graph_compare
        from python.msprobe.mindspore.compare.common_dir_compare import common_dir_compare

    common_kwargs = {
        "auto_analyze": auto_analyze,
        "fuzzy_match": args.fuzzy_match,
        "highlight": args.highlight,
        "data_mapping": args.data_mapping,
        "diff_analyze": args.diff_analyze
    }

    if check_file_type(npu_path) == FileCheckConst.FILE and check_file_type(bench_path) == FileCheckConst.FILE:
        check_file_or_directory_path(npu_path)
        check_file_or_directory_path(bench_path)
        input_param["npu_json_path"] = input_param.pop("npu_path")
        input_param["bench_json_path"] = input_param.pop("bench_path")
        if "stack_path" not in input_param:
            logger.warning(f"Missing stack_path in the configuration file. "
                           f"Automatically detecting stack.json to determine whether to display NPU_Stack_Info.")
        else:
            input_param["stack_json_path"] = input_param.pop("stack_path")

        if frame_name == Const.PT_FRAMEWORK:
            kwargs = {**common_kwargs, "stack_mode": args.stack_mode}
            compare(input_param, args.output_path, **kwargs)
        else:
            kwargs = {
                **common_kwargs,
                "stack_mode": args.stack_mode,
                "cell_mapping": args.cell_mapping,
                "api_mapping": args.api_mapping,
                "layer_mapping": args.layer_mapping
            }
            ms_compare(input_param, args.output_path, **kwargs)
    elif check_file_type(npu_path) == FileCheckConst.DIR and check_file_type(bench_path) == FileCheckConst.DIR:
        check_file_or_directory_path(npu_path, isdir=True)
        check_file_or_directory_path(bench_path, isdir=True)

        if depth == 1:
            mix_compare_success = mix_compare(args, input_param, depth)
            if mix_compare_success:
                return

        kwargs = {
            **common_kwargs,
            "stack_mode": args.stack_mode,
            "is_print_compare_log": input_param.get("is_print_compare_log", True),
            "cell_mapping": args.cell_mapping,
            "api_mapping": args.api_mapping,
            "layer_mapping": args.layer_mapping
        }
        if input_param.get("rank_id") is not None:
            ms_graph_compare(input_param, args.output_path)
            return
        common = input_param.get("common", False)
        if isinstance(common, bool) and common:
            common_dir_compare(input_param, args.output_path)
            return

        if common_kwargs.get('diff_analyze', False):
            logger.info("Start finding first diff node......")
            from python.msprobe.core.compare.find_first.analyzer import DiffAnalyzer
            DiffAnalyzer(npu_path, bench_path, args.output_path, frame_name).analyze()
            return

        if frame_name == Const.PT_FRAMEWORK:
            compare_distributed(npu_path, bench_path, args.output_path, **kwargs)
        else:
            ms_compare_distributed(npu_path, bench_path, args.output_path, **kwargs)
    else:
        logger.error("The npu_path and bench_path need to be of the same type.")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)


def mix_compare(args, input_param, depth):
    npu_path = input_param.get("npu_path", None)
    bench_path = input_param.get("bench_path", None)

    npu_bench_same_dirs_set = set(get_paired_dirs(npu_path, bench_path))
    compare_cross_set = npu_bench_same_dirs_set & Const.MIX_DUMP_NAMES

    if compare_cross_set:
        logger.info("Start mix compare.")
        origin_output = args.output_path

        for folder_name in list(compare_cross_set):
            new_npu_path = os.path.join(npu_path, folder_name)
            new_bench_path = os.path.join(bench_path, folder_name)
            paired_steps = get_paired_dirs(new_npu_path, new_bench_path)

            for step_name in paired_steps:
                logger.info(f"[mix compare] Start comparing {folder_name}/{step_name}")
                npu_dir = os.path.join(new_npu_path, step_name)
                bench_dir = os.path.join(new_bench_path, step_name)
                args.input_path = {
                    "npu_path": npu_dir,
                    "bench_path": bench_dir,
                    "is_print_compare_log": input_param.get("is_print_compare_log", True)
                }
                args.output_path = os.path.join(origin_output, folder_name, step_name)
                compare_cli(args, depth + 1)
        return True
    return False
