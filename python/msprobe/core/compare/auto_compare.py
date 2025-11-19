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
from msprobe.core.common.file_utils import check_file_type, check_file_or_directory_path
from msprobe.core.common.const import FileCheckConst, Const
from msprobe.core.common.utils import CompareException
from msprobe.core.common.log import logger
from msprobe.core.compare.utils import get_paired_dirs
from msprobe.core.compare.utils import get_compare_framework
from msprobe.core.compare.utils import compare_distributed_inner


def compare_auto_mode(args, depth=1):
    """
    Auto-detect comparison mode based on input paths
    This is the original compare_cli logic
    """

    if depth > 2:
        logger.error("Recursive compare error, depth exceeds 2.")
        raise CompareException(CompareException.RECURSION_LIMIT_ERROR)

    common_kwargs = {
        "fuzzy_match": args.fuzzy_match,
        "data_mapping": args.data_mapping,
        "diff_analyze": args.diff_analyze
    }

    if (check_file_type(args.target_path) == FileCheckConst.FILE and
            check_file_type(args.golden_path) == FileCheckConst.FILE):

        check_file_or_directory_path(args.target_path)
        check_file_or_directory_path(args.golden_path)

        input_param = {"npu_path": args.target_path, "bench_path": args.golden_path}
        frame_name = get_compare_framework(input_param)

        if frame_name == Const.PT_FRAMEWORK:
            if args.cell_mapping is not None or args.api_mapping is not None:
                logger.error("Argument -cm or -am is not supported in PyTorch framework")
                raise Exception("Argument -cm or -am is not supported in PyTorch framework")
            kwargs = {**common_kwargs}
            from msprobe.pytorch.compare.pt_compare import compare
            compare(input_param, args.output_path, **kwargs)
        else:
            kwargs = {
                **common_kwargs,
                "cell_mapping": args.cell_mapping,
                "api_mapping": args.api_mapping,
                "layer_mapping": args.layer_mapping
            }
            from msprobe.mindspore.compare.ms_compare import ms_compare
            ms_compare(input_param, args.output_path, **kwargs)
    elif (check_file_type(args.target_path) == FileCheckConst.DIR and
          check_file_type(args.golden_path) == FileCheckConst.DIR):
        check_file_or_directory_path(args.target_path, isdir=True)
        check_file_or_directory_path(args.golden_path, isdir=True)

        if depth == 1:
            mix_compare_success = mix_compare(args, depth)
            if mix_compare_success:
                return

        kwargs = {
            **common_kwargs,
            "is_print_compare_log": True,
            "cell_mapping": args.cell_mapping,
            "api_mapping": args.api_mapping,
            "layer_mapping": args.layer_mapping
        }
        if args.rank is not None:
            from msprobe.mindspore.compare.distributed_compare import ms_graph_compare
            ms_graph_compare(args)
            return

        if common_kwargs.get('diff_analyze', False):
            logger.info("Start finding first diff node......")
            from msprobe.core.compare.find_first.analyzer import DiffAnalyzer
            DiffAnalyzer(args.target_path, args.golden_path, args.output_path).analyze()
            return

        compare_distributed_inner(args.target_path, args.golden_path, args.output_path, **kwargs)
    else:
        logger.error("The npu_path and bench_path need to be of the same type.")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)


def mix_compare(args, depth):
    npu_path = args.target_path
    bench_path = args.golden_path

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
                args.target_path = npu_dir
                args.golden_path = bench_dir
                args.output_path = os.path.join(origin_output, folder_name, step_name)
                compare_auto_mode(args, depth + 1)
        return True
    return False
