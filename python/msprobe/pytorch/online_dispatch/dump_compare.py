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
import os
from datetime import datetime, timezone

import torch
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.core.common.file_utils import save_npy, save_json, remove_path
from msprobe.pytorch.common.log import logger


class DispatchRunParam:
    def __init__(self, debug_flag, device_id, root_npu_path, root_cpu_path, process_num, comparator):
        # static parameters are initialized by constructors, and dynamic parameters are constructed at run time
        self.debug_flag = debug_flag
        self.device_id = device_id
        self.root_npu_path = root_npu_path
        self.root_cpu_path = root_cpu_path
        self.process_num = process_num
        self.process_flag = False
        self.func_name = None
        self.func_namespace = None
        self.aten_api = None
        self.aten_api_overload_name = None
        self.single_api_index = None
        self.api_index = None
        self.dump_flag = None
        self.auto_dump_flag = None
        self.comparator = comparator


class DisPatchDataInfo:
    def __init__(self, cpu_args, cpu_kwargs, all_summary, func, npu_out_cpu, cpu_out, lock):
        self.cpu_args = cpu_args
        self.cpu_kwargs = cpu_kwargs
        self.all_summary = all_summary
        self.func = func
        self.npu_out_cpu = npu_out_cpu
        self.cpu_out = cpu_out
        self.lock = lock


class TimeStatistics:
    def __init__(self, name_tag, run_param, timeout=5):
        self.debug = run_param.debug_flag
        if self.debug:
            self.fun = run_param.func_name
            self.device = run_param.device_id
            self.process = run_param.process_num
            self.index = run_param.single_api_index
            self.tag = name_tag
            self.timeout = timeout
            self.time = None

    def __enter__(self):
        if self.debug:
            self.time = datetime.now(tz=timezone.utc)
            logger.info(f'Time[{self.tag}]-ENTER: Dev[{self.device}], Pid[{os.getpid()}], Fun[{self.fun}], ' \
                        f'Id[{self.index}]')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.debug:
            cost_time = datetime.now(tz=timezone.utc) - self.time
            time_cost = f'Time[{self.tag}]-EXIT: Dev[{self.device}], Pid[{os.getpid()}], Fun[{self.fun}], ' \
                        f'Id[{self.index}], time[{cost_time}]'
            hot_time_cost = "Hotspot " + time_cost

            if cost_time.total_seconds() > self.timeout:
                logger.info(hot_time_cost)
            else:
                logger.info(time_cost)


def support_basic_type(data):
    if isinstance(data, (bool, int, float, torch.Tensor)):
        return True
    return False


@recursion_depth_decorator("dump_data")
def dump_data(data, prefix, dump_path):
    if isinstance(data, (tuple, list)) and data:
        for i, item in enumerate(data):
            dump_data(item, "{}.{}".format(prefix, i), dump_path)
        return
    elif support_basic_type(data):
        if isinstance(data, torch.Tensor) and data.is_meta:
            return
        # dump data may greater than summary_list collect
        path = os.path.join(dump_path, f'{prefix}.npy')
        save_npy(data, path)


def save_temp_summary(api_index, single_api_summary, path, lock):
    summary_path = os.path.join(path, f'summary.json')
    lock.acquire()
    try:
        data = [api_index, single_api_summary]
        save_json(summary_path, data, mode='a')
    except Exception as e:
        logger.error(f'save temp summary error:{e}')
        try:
            remove_path(summary_path)
        except FileNotFoundError:
            logger.error(f'file not found:{summary_path}')
    finally:
        lock.release()


def dispatch_workflow(run_param: DispatchRunParam, data_info: DisPatchDataInfo):
    cpu_args, cpu_kwargs = data_info.cpu_args, data_info.cpu_kwargs
    all_summary, func = data_info.all_summary, data_info.func
    npu_out_cpu, cpu_out, lock = data_info.npu_out_cpu, data_info.cpu_out, data_info.lock
    single_api_summary = []

    prefix_input = f'{run_param.aten_api}_{run_param.single_api_index}_input'
    prefix_output = f'{run_param.aten_api}_{run_param.single_api_index}_output'

    accuracy_reached = False
    with TimeStatistics("COMPARE OUTPUT", run_param):
        run_param.comparator.compare_output(prefix_output, cpu_out, npu_out_cpu, None, None)

    # user set dump or auto mode will dump
    if run_param.dump_flag or (run_param.auto_dump_flag and not accuracy_reached):
        with TimeStatistics("DUMP INPUT", run_param):
            dump_data(cpu_args, prefix_input, run_param.root_npu_path)
            if len(cpu_kwargs) > 0:
                for k, v in cpu_kwargs.items():
                    kwargs_prefix_name = prefix_input + f'_{k}'
                    dump_data(v, kwargs_prefix_name, run_param.root_npu_path)

        with TimeStatistics("DUMP OUTPUT", run_param):
            dump_data(cpu_out, prefix_output, run_param.root_cpu_path)
            dump_data(npu_out_cpu, prefix_output, run_param.root_npu_path)

    if run_param.process_num == 0:
        all_summary[run_param.api_index - 1] = copy.deepcopy(single_api_summary)
    else:
        save_temp_summary(run_param.api_index - 1, single_api_summary, run_param.root_cpu_path, lock)


def get_torch_func(run_param):
    if hasattr(torch.ops, run_param.func_namespace):
        ops_func = getattr(torch.ops, run_param.func_namespace)
        if hasattr(ops_func, run_param.aten_api):
            ops_aten_func = getattr(ops_func, run_param.aten_api)
            if hasattr(ops_aten_func, run_param.aten_api_overload_name):
                ops_aten_overlaod_func = getattr(ops_aten_func, run_param.aten_api_overload_name)
                return ops_aten_overlaod_func
    return None


def dispatch_multiprocess(run_param, dispatch_data_info):
    torch_func = get_torch_func(run_param)
    if torch_func is None:
        logger.error(f'can not find suitable call api:{run_param.aten_api}')
    else:
        dispatch_data_info.func = torch_func
        dispatch_workflow(run_param, dispatch_data_info)


def error_call(err):
    logger.error(f'multiprocess {err}')
