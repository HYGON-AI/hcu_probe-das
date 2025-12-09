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
import time
from copy import deepcopy
from multiprocessing import cpu_count, Pool
from msprobe.core.common.file_utils import (check_file_type, create_directory, FileChecker,
                                            check_file_or_directory_path, load_json)
from msprobe.core.common.const import FileCheckConst, Const
from msprobe.core.common.utils import CompareException, get_dump_mode
from msprobe.visualization.compare.graph_comparator import GraphComparator
from msprobe.visualization.utils import GraphConst, check_directory_content, SerializableArgs, load_parallel_param, \
    sort_rank_number_strings, validate_parallel_param, get_step_or_rank_int
from msprobe.visualization.builder.graph_builder import GraphBuilder, GraphExportConfig, GraphInfo, BuildGraphTaskInfo
from msprobe.core.common.log import logger
from msprobe.visualization.graph.node_colors import NodeColors
from msprobe.core.compare.layer_mapping import generate_api_mapping_by_layer_mapping
from msprobe.core.compare.utils import check_and_return_dir_contents
from msprobe.core.common.utils import detect_framework_by_dump_json
from msprobe.visualization.graph.distributed_analyzer import DistributedAnalyzer
from msprobe.visualization.builder.graph_merger import GraphMerger
from msprobe.visualization.db_utils import post_process_db

current_time = time.strftime("%Y%m%d%H%M%S")
build_output_db_name = f'build_{current_time}.vis.db'
compare_output_db_name = f'compare_{current_time}.vis.db'


def _compare_graph(graph_n: GraphInfo, graph_b: GraphInfo, input_param, args):
    dump_path_param = {
        'npu_path': graph_n.data_path,
        'bench_path': graph_b.data_path,
        'stack_path': graph_n.stack_path,
        'is_print_compare_log': input_param.get("is_print_compare_log", True)
    }
    mapping_dict = {}
    if args.layer_mapping:
        try:
            mapping_dict = generate_api_mapping_by_layer_mapping(graph_n.data_path, graph_b.data_path,
                                                                 args.layer_mapping)
        except Exception:
            logger.warning('The layer mapping file parsing failed, please check file format, mapping is not effective.')
    is_cross_framework = detect_framework_by_dump_json(graph_n.data_path) != \
                         detect_framework_by_dump_json(graph_b.data_path)
    if is_cross_framework and not args.layer_mapping:
        logger.error('The cross_frame graph comparison failed. '
                     'Please specify -lm or --layer_mapping when performing cross_frame graph comparison.')
        raise CompareException(CompareException.CROSS_FRAME_ERROR)

    graph_comparator = GraphComparator([graph_n.graph, graph_b.graph], dump_path_param, args, is_cross_framework,
                                       mapping_dict=mapping_dict)
    graph_comparator.compare()
    return graph_comparator


def _compare_graph_result(input_param, args):
    logger.info('Start building model graphs...')
    # 对两个数据进行构图
    graph_n = _build_graph_info(input_param.get('npu_path'), args)
    graph_b = _build_graph_info(input_param.get('bench_path'), args)
    logger.info('Model graphs built successfully, start comparing graphs...')
    # 基于graph、stack和data进行比较
    graph_comparator = _compare_graph(graph_n, graph_b, input_param, args)
    # 增加micro step标记
    micro_steps = graph_n.graph.paging_by_micro_step(graph_b.graph)
    # 开启溢出检测
    if args.overflow_check:
        graph_n.graph.overflow_check()
        graph_b.graph.overflow_check()

    return CompareGraphResult(graph_n.graph, graph_b.graph, graph_comparator, micro_steps)


def _export_compare_graph_result(args, result):
    graphs = [result.graph_n, result.graph_b]
    graph_comparator = result.graph_comparator
    micro_steps = result.micro_steps
    logger.info(f'Start exporting compare graph result, file name: {compare_output_db_name}...')
    output_db_path = os.path.join(args.output_path, compare_output_db_name)
    task = GraphConst.GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING.get(graph_comparator.ma.compare_mode)
    export_config = GraphExportConfig(graphs[0], graphs[1], graph_comparator.ma.get_tool_tip(),
                                      NodeColors.get_node_colors(graph_comparator.ma.compare_mode), micro_steps, task,
                                      args.overflow_check, graph_comparator.ma.compare_mode, result.step, result.rank,
                                      args.step_list if hasattr(args, 'step_list') else [0],
                                      args.rank_list if hasattr(args, 'rank_list') else [0])
    try:
        GraphBuilder.to_db(output_db_path, export_config)
        logger.info(f'Exporting compare graph result successfully, the result file is saved in {output_db_path}')
        return ''
    except RuntimeError as e:
        logger.error(f'Failed to export compare graph result, file: {compare_output_db_name}, error: {e}')
        return compare_output_db_name


def _build_graph_info(dump_path, args, graph=None):
    construct_path = FileChecker(os.path.join(dump_path, GraphConst.CONSTRUCT_FILE), FileCheckConst.FILE,
                                 FileCheckConst.READ_ABLE).common_check()
    data_path = FileChecker(os.path.join(dump_path, GraphConst.DUMP_FILE), FileCheckConst.FILE,
                            FileCheckConst.READ_ABLE).common_check()
    stack_path = FileChecker(os.path.join(dump_path, GraphConst.STACK_FILE), FileCheckConst.FILE,
                             FileCheckConst.READ_ABLE).common_check()
    if not graph:
        graph = GraphBuilder.build(construct_path, data_path, stack_path)
    return GraphInfo(graph, construct_path, data_path, stack_path)


def _build_graph_result(dump_path, args):
    logger.info('Start building model graphs...')
    graph = _build_graph_info(dump_path, args).graph
    # 增加micro step标记
    micro_steps = graph.paging_by_micro_step()
    # 开启溢出检测
    if args.overflow_check:
        graph.overflow_check()
    return BuildGraphResult(graph, micro_steps)


def _run_build_graph_compare(input_param, args, nr, br):
    logger.info(f'Start building graph for {nr}...')
    graph_n = _build_graph_info(input_param.get('npu_path'), args)
    graph_b = _build_graph_info(input_param.get('bench_path'), args)
    logger.info(f'Building graph for {nr} finished.')
    return BuildGraphTaskInfo(graph_n, graph_b, nr, br, current_time)


def _run_build_graph_single(dump_ranks_path, rank, step, args):
    logger.info(f'Start building graph for {rank}...')
    dump_path = os.path.join(dump_ranks_path, rank)
    result = _build_graph_result(dump_path, args)
    if rank != Const.RANK:
        result.rank = get_step_or_rank_int(rank, True)
    logger.info(f'Building graph for step: {step}, rank: {rank} finished.')
    return result


def _run_graph_compare(graph_task_info, input_param, args):
    logger.info(f'Start comparing data for {graph_task_info.npu_rank}...')
    graph_n = graph_task_info.graph_info_n
    graph_b = graph_task_info.graph_info_b
    nr = graph_task_info.npu_rank
    graph_comparator = _compare_graph(graph_n, graph_b, input_param, args)
    micro_steps = graph_n.graph.paging_by_micro_step(graph_b.graph)
    # 开启溢出检测
    if args.overflow_check:
        graph_n.graph.overflow_check()
        graph_b.graph.overflow_check()
    graph_result = CompareGraphResult(graph_n.graph, graph_b.graph, graph_comparator, micro_steps)
    if nr != Const.RANK:
        graph_result.rank = get_step_or_rank_int(nr, True)
    logger.info(f'Comparing data for {graph_task_info.npu_rank} finished.')
    return graph_result


def _export_build_graph_result(args, result):
    out_path = args.output_path
    graph = result.graph
    micro_steps = result.micro_steps
    overflow_check = args.overflow_check
    logger.info(f'Start exporting graph for {build_output_db_name}...')
    output_db_path = os.path.join(out_path, build_output_db_name)
    config = GraphExportConfig(graph, micro_steps=micro_steps, overflow_check=overflow_check, rank=result.rank,
                               step=result.step, rank_list=args.rank_list if hasattr(args, 'rank_list') else [0],
                               step_list=args.step_list if hasattr(args, 'step_list') else [0])
    try:
        GraphBuilder.to_db(output_db_path, config)
        logger.info(f'Model graph exported successfully, the result file is saved in {output_db_path}')
        return None
    except RuntimeError as e:
        logger.error(f'Failed to export model graph, file: {build_output_db_name}, error: {e}')
        return build_output_db_name


def is_real_data_compare(input_param, npu_ranks, bench_ranks):
    dump_rank_n = input_param.get('npu_path')
    dump_rank_b = input_param.get('bench_path')
    has_real_data = False
    for nr, br in zip(npu_ranks, bench_ranks):
        dump_path_param = {
            'npu_path': FileChecker(os.path.join(dump_rank_n, nr, GraphConst.DUMP_FILE), FileCheckConst.FILE,
                                    FileCheckConst.READ_ABLE).common_check(),
            'bench_path': FileChecker(os.path.join(dump_rank_b, br, GraphConst.DUMP_FILE), FileCheckConst.FILE,
                                      FileCheckConst.READ_ABLE).common_check()
        }
        has_real_data |= get_dump_mode(dump_path_param) == Const.ALL
    return has_real_data


def _mp_compare(input_param, serializable_args, nr, br):
    graph_task_info = _run_build_graph_compare(input_param, serializable_args, nr, br)
    return _run_graph_compare(graph_task_info, input_param, serializable_args)


def _compare_graph_ranks(input_param, args, step=None):
    with Pool(processes=max(int((cpu_count() + 1) // 4), 1)) as pool:
        def err_call(err):
            logger.error(f'Error occurred while comparing graph ranks: {err}')
            try:
                pool.close()
            except OSError as e:
                logger.error(f'Error occurred while terminating the pool: {e}')

        serializable_args = SerializableArgs(args)
        # 暂存所有rank的graph，用于匹配rank间的分布式节点
        compare_graph_results = _get_compare_graph_results(input_param, serializable_args, step, pool, err_call)

        serializable_args.rank_list = [result.rank for result in compare_graph_results]

        # 匹配rank间的分布式节点
        if len(compare_graph_results) > 1:
            DistributedAnalyzer({obj.rank: obj.graph_n for obj in compare_graph_results},
                                args.overflow_check).distributed_match()
            DistributedAnalyzer({obj.rank: obj.graph_b for obj in compare_graph_results},
                                args.overflow_check).distributed_match()

        export_res_task_list = []
        create_directory(args.output_path)
        for result in compare_graph_results:
            export_res_task_list.append(pool.apply_async(_export_compare_graph_result,
                                                         args=(serializable_args, result),
                                                         error_callback=err_call))
        export_res_list = [res.get() for res in export_res_task_list]
        if any(export_res_list):
            failed_names = list(filter(lambda x: x, export_res_list))
            logger.error(f'Unable to export compare graph results: {", ".join(failed_names)}.')
        else:
            logger.info('Successfully exported compare graph results.')


def _get_compare_graph_results(input_param, serializable_args, step, pool, err_call):
    dump_rank_n = input_param.get('npu_path')
    dump_rank_b = input_param.get('bench_path')
    npu_ranks = sort_rank_number_strings(check_and_return_dir_contents(dump_rank_n, Const.RANK))
    bench_ranks = sort_rank_number_strings(check_and_return_dir_contents(dump_rank_b, Const.RANK))
    if npu_ranks != bench_ranks:
        intersection_ranks = sort_rank_number_strings(list(set(npu_ranks) & set(bench_ranks)))
        if not intersection_ranks:
            logger.error('The ranks in the two runs are completely different. Unable to match the ranks.')
            raise CompareException(CompareException.INVALID_PATH_ERROR)
        npu_ranks = intersection_ranks
        bench_ranks = intersection_ranks
    compare_graph_results = []
    if is_real_data_compare(input_param, npu_ranks, bench_ranks):
        mp_task_dict = {}
        for nr, br in zip(npu_ranks, bench_ranks):
            input_param['npu_path'] = os.path.join(dump_rank_n, nr)
            input_param['bench_path'] = os.path.join(dump_rank_b, br)
            build_key = f'{step}_{nr}' if step else f'{nr}'
            input_param_copy = deepcopy(input_param)
            mp_task_dict[build_key] = pool.apply_async(_run_build_graph_compare,
                                                       args=(input_param_copy, serializable_args, nr, br),
                                                       error_callback=err_call)

        mp_res_dict = {k: v.get() for k, v in mp_task_dict.items()}
        for mp_res in mp_res_dict.values():
            compare_graph_results.append(_run_graph_compare(mp_res, input_param, serializable_args))
    else:
        compare_graph_tasks = []
        for nr, br in zip(npu_ranks, bench_ranks):
            input_param['npu_path'] = os.path.join(dump_rank_n, nr)
            input_param['bench_path'] = os.path.join(dump_rank_b, br)
            input_param_copy = deepcopy(input_param)
            compare_graph_tasks.append(pool.apply_async(_mp_compare,
                                                        args=(input_param_copy, serializable_args, nr, br),
                                                        error_callback=err_call))
        compare_graph_results = [task.get() for task in compare_graph_tasks]
    if step is not None:
        for result in compare_graph_results:
            result.step = get_step_or_rank_int(step)
    return compare_graph_results


def _compare_graph_steps(input_param, args):
    dump_step_n = input_param.get('npu_path')
    dump_step_b = input_param.get('bench_path')

    npu_steps = check_and_return_dir_contents(dump_step_n, Const.STEP)
    bench_steps = check_and_return_dir_contents(dump_step_b, Const.STEP)

    if npu_steps != bench_steps:
        intersection_steps = sort_rank_number_strings(list(set(npu_steps) & set(bench_steps)))

        if not intersection_steps:
            logger.error('The steps in the two runs are completely different. Unable to match the steps.')
            raise CompareException(CompareException.INVALID_PATH_ERROR)
        npu_steps = intersection_steps

    args.step_list = sorted([get_step_or_rank_int(step) for step in npu_steps])

    for folder_step in npu_steps:
        logger.info(f'Start processing data for {folder_step}...')
        input_param['npu_path'] = os.path.join(dump_step_n, folder_step)
        input_param['bench_path'] = os.path.join(dump_step_b, folder_step)

        _compare_graph_ranks(input_param, args, step=folder_step) if not args.parallel_merge \
            else _compare_graph_ranks_parallel(input_param, args, step=folder_step)


def _build_graph_ranks(args, step=None):
    dump_ranks_path = os.path.join(args.target_path, step) if step is not None else args.target_path
    ranks = sort_rank_number_strings(check_and_return_dir_contents(dump_ranks_path, Const.RANK))
    serializable_args = SerializableArgs(args)
    with Pool(processes=max(int((cpu_count() + 1) // 4), 1)) as pool:
        def err_call(err):
            logger.error(f'Error occurred while comparing graph ranks: {err}')
            try:
                pool.close()
            except OSError as e:
                logger.error(f'Error occurred while terminating the pool: {e}')

        build_graph_tasks = []
        for rank in ranks:
            build_graph_tasks.append(pool.apply_async(_run_build_graph_single,
                                                      args=(dump_ranks_path, rank, step, serializable_args),
                                                      error_callback=err_call))
        build_graph_results = [task.get() for task in build_graph_tasks]

        if step is not None:
            for result in build_graph_results:
                result.step = get_step_or_rank_int(step)

        if args.parallel_params:
            validate_parallel_param(args.parallel_params[0], dump_ranks_path)
            build_graph_results = GraphMerger(build_graph_results, args.parallel_params[0]).merge_graph()

        if len(build_graph_results) > 1 and not args.parallel_merge:
            DistributedAnalyzer({obj.rank: obj.graph for obj in build_graph_results},
                                args.overflow_check).distributed_match()

        create_directory(args.output_path)
        export_build_graph_tasks = []
        serializable_args.rank_list = [result.rank for result in build_graph_results]
        for result in build_graph_results:
            export_build_graph_tasks.append(pool.apply_async(_export_build_graph_result,
                                                             args=(serializable_args, result),
                                                             error_callback=err_call))
        export_build_graph_result = [task.get() for task in export_build_graph_tasks]
        if any(export_build_graph_result):
            failed_names = list(filter(lambda x: x, export_build_graph_result))
            logger.error(f'Unable to export build graph results: {failed_names}.')
        else:
            logger.info(f'Successfully exported build graph results.')


def _build_graph_steps(args):
    steps = sorted(check_and_return_dir_contents(args.target_path, Const.STEP))
    args.step_list = sorted([get_step_or_rank_int(step) for step in steps])

    for step in steps:
        logger.info(f'Start processing data for {step}...')
        _build_graph_ranks(args, step)


def _compare_and_export_graph(graph_task_info, input_param, args, step=None):
    result = _run_graph_compare(graph_task_info, input_param, args)
    if step is not None:
        result.step = get_step_or_rank_int(step)
    return _export_compare_graph_result(args, result)


def _compare_graph_ranks_parallel(input_param, args, step=None):
    args.fuzzy_match = True
    npu_path = input_param.get('npu_path')
    bench_path = input_param.get('bench_path')
    ranks_n = sort_rank_number_strings(check_and_return_dir_contents(npu_path, Const.RANK))
    ranks_b = sort_rank_number_strings(check_and_return_dir_contents(bench_path, Const.RANK))
    parallel_params = args.parallel_params
    if len(parallel_params) != 2:
        raise RuntimeError('Parallel params error in compare graph!')
    validate_parallel_param(parallel_params[0], npu_path)
    validate_parallel_param(parallel_params[1], bench_path, '[Bench]')
    serializable_args = SerializableArgs(args)

    with Pool(processes=max(int((cpu_count() + 1) // 4), 1)) as pool:
        def err_call(err):
            logger.error(f'Error occurred while comparing graph ranks: {err}')
            try:
                pool.close()
            except OSError as e:
                logger.error(f'Error occurred while terminating the pool: {e}')

        # 1.并行构图
        build_graph_tasks_n = []
        build_graph_tasks_b = []
        for rank in ranks_n:
            build_graph_tasks_n.append(pool.apply_async(_run_build_graph_single,
                                                        args=(npu_path, rank, step, serializable_args),
                                                        error_callback=err_call))
        for rank in ranks_b:
            build_graph_tasks_b.append(pool.apply_async(_run_build_graph_single,
                                                        args=(bench_path, rank, step, serializable_args),
                                                        error_callback=err_call))
        graph_results_n = [task.get() for task in build_graph_tasks_n]
        graph_results_b = [task.get() for task in build_graph_tasks_b]

        # 2.图合并
        build_graph_results_n = GraphMerger(graph_results_n, parallel_params[0]).merge_graph()
        build_graph_results_b = GraphMerger(graph_results_b, parallel_params[1], True).merge_graph()
        if len(build_graph_results_n) != len(build_graph_results_b):
            raise RuntimeError(f'Parallel merge failed because the dp of npu: {len(build_graph_results_n)} '
                               f'is inconsistent with that of bench: {len(build_graph_results_b)}!')
        serializable_args.rank_list = [result.rank for result in build_graph_results_n]
        # 3.并行图比对和输出
        export_res_task_list = []
        create_directory(args.output_path)
        for i, result_n in enumerate(build_graph_results_n):
            graph_n = result_n.graph
            graph_b = build_graph_results_b[i].graph
            graph_task_info = BuildGraphTaskInfo(
                _build_graph_info(os.path.join(npu_path, f'rank{graph_n.root.rank}'), args, graph_n),
                _build_graph_info(os.path.join(bench_path, f'rank{graph_b.root.rank}'), args, graph_b),
                f'rank{graph_n.root.rank}', f'rank{graph_b.root.rank}', current_time)
            export_res_task_list.append(pool.apply_async(_compare_and_export_graph,
                                                         args=(graph_task_info, input_param, serializable_args, step),
                                                         error_callback=err_call))
        export_res_list = [res.get() for res in export_res_task_list]
        if any(export_res_list):
            failed_names = list(filter(lambda x: x, export_res_list))
            logger.error(f'Unable to export compare graph results: {", ".join(failed_names)}.')
        else:
            logger.info('Successfully exported compare graph results.')


def _graph_service_parser(parser):
    # -------------------------- 基础必填参数 --------------------------
    parser.add_argument("-tp", "--target_path", dest="target_path", type=str,
                        help="<Required> The target path.", required=True)
    parser.add_argument("-o", "--output_path", dest="output_path", type=str,
                        help="<Required> The visualization task result out path.", required=True)
    # -------------------------- 基础可选参数 --------------------------
    parser.add_argument("-gp", "--golden_path", dest="golden_path", type=str,
                        help="<Optional> The golden path.", required=False)
    parser.add_argument("-lm", "--layer_mapping", dest="layer_mapping", type=str, nargs='?', const=True,
                        help="<Optional> The layer mapping file path.", required=False)
    parser.add_argument("-oc", "--overflow_check", dest="overflow_check", action="store_true",
                        help="<Optional> whether open overflow_check for graph.", required=False)
    parser.add_argument("-fm", "--fuzzy_match", dest="fuzzy_match", action="store_true",
                        help="<Optional> whether to perform a fuzzy match on the api name.", required=False)
    parser.add_argument("-log", "--is_print_compare_log", dest="is_print_compare_log", action="store_true",
                        help="<Optional> whether print compare log for visualization task.", required=False)

    # -------------------------- 不同并行切分策略合并可选参数 --------------------------
    group_n = parser.add_argument_group("Parallel Parameters, "
                                        "used for graph merging under different parallel partitioning strategies")

    group_n.add_argument("--rank_size", type=int, nargs='+', help="<Optional> The rank size of dump path.",
                         required=False)
    group_n.add_argument("--tp", type=int, nargs='+',
                         help="<Optional, but required if rank_size is not empty> The tp size of dump path.",
                         required=False)
    group_n.add_argument("--pp", type=int, nargs='+',
                         help="<Optional, but required if rank_size is not empty> The pp size of dump path.",
                         required=False)
    group_n.add_argument("--vpp", type=int, nargs='+', default=[1], help="<Optional> The vpp size of dump path.",
                         required=False)
    group_n.add_argument("--order", type=str, nargs='+', default=['tp-cp-ep-dp-pp'],
                         help="<Optional> The order of dump path.", required=False)


def _graph_service_command(args):
    npu_path = args.target_path
    bench_path = args.golden_path
    args.parallel_merge = True if args.rank_size else False
    args.parallel_params = load_parallel_param(args) if args.parallel_merge else None
    check_file_or_directory_path(npu_path, isdir=True)
    if bench_path:
        check_file_or_directory_path(bench_path, isdir=True)
    if check_file_type(npu_path) == FileCheckConst.DIR and not bench_path:
        content = check_directory_content(npu_path)
        output_db_path = os.path.join(args.output_path, build_output_db_name)
        if content == GraphConst.RANKS:
            _build_graph_ranks(args)
        elif content == GraphConst.STEPS:
            _build_graph_steps(args)
        else:
            result = _build_graph_result(npu_path, args)
            create_directory(args.output_path)
            file_name = _export_build_graph_result(args, result)
            if file_name:
                logger.error('Failed to export model build graph.')
    elif check_file_type(npu_path) == FileCheckConst.DIR and check_file_type(bench_path) == FileCheckConst.DIR:
        content_n = check_directory_content(npu_path)
        content_b = check_directory_content(bench_path)
        output_db_path = os.path.join(args.output_path, compare_output_db_name)
        if content_n != content_b:
            raise ValueError('The directory structures of npu_path and bench_path are inconsistent.')
        input_param = {
            'npu_path': args.target_path,
            'bench_path': args.golden_path,
            'is_print_compare_log': args.is_print_compare_log
        }
        if content_n == GraphConst.RANKS:
            if args.parallel_merge:
                _compare_graph_ranks_parallel(input_param, args)
            else:
                _compare_graph_ranks(input_param, args)
        elif content_n == GraphConst.STEPS:
            _compare_graph_steps(input_param, args)
        else:
            result = _compare_graph_result(input_param, args)
            create_directory(args.output_path)
            file_name = _export_compare_graph_result(args, result)
            if file_name:
                logger.error('Failed to export model compare graph.')
    else:
        logger.error("The npu_path or bench_path should be a folder.")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)
    # 所有数据输出db结束后，添加索引，修改权限
    post_process_db(output_db_path)


class CompareGraphResult:
    def __init__(self, graph_n, graph_b, graph_comparator, micro_steps, rank=0, step=0):
        self.graph_n = graph_n
        self.graph_b = graph_b
        self.graph_comparator = graph_comparator
        self.micro_steps = micro_steps
        self.rank = rank
        self.step = step


class BuildGraphResult:
    def __init__(self, graph, micro_steps=0, rank=0, step=0):
        self.graph = graph
        self.micro_steps = micro_steps
        self.rank = rank
        self.step = step
