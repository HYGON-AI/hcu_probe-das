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

import json
import os
import time
import multiprocessing
from multiprocessing import Pool, Lock

import torch
from torch.utils._python_dispatch import TorchDispatchMode

try:
    import torch_npu
except ImportError:
    is_npu = False
else:
    is_npu = True

from msprobe.core.common.file_utils import check_file_or_directory_path, load_yaml, FileOpen, create_directory
from msprobe.core.common.const import Const, CompareConst
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.online_dispatch.dump_compare import dispatch_workflow, dispatch_multiprocess, error_call, \
    TimeStatistics, DispatchRunParam, DisPatchDataInfo
from msprobe.pytorch.online_dispatch.utils import get_callstack, data_to_cpu, get_sys_info, DispatchException, \
    COMPARE_LOGO
from msprobe.pytorch.online_dispatch.compare import Comparator
from msprobe.core.common.utils import check_str_param, safe_get_value

child_global_lock = None
current_time = time.strftime("%Y%m%d%H%M%S")
RESULT_FILE_NAME = "accuracy_checking_result_" + current_time + ".csv"
DETAILS_FILE_NAME = "accuracy_checking_details_" + current_time + ".csv"


class PtdbgDispatch(TorchDispatchMode):
    def __init__(self, dump_mode=Const.OFF, api_list=None, debug=False, dump_path=None, tag=None, process_num=0):
        super(PtdbgDispatch, self).__init__()
        logger.info(COMPARE_LOGO)
        if not is_npu:
            logger.error("Please confirm you run environment installed torch_npu!")
            return
        if dump_path is None:
            logger.error("Please set dump_path when dump_mode is config!")
            raise DispatchException("Please set dump_path when dump_mode is config!")
        check_file_or_directory_path(dump_path, True)

        self.device_id = torch_npu._C._npu_getDevice()
        self.dump_mode = dump_mode
        self.dump_api_list = api_list or []
        self.debug_flag = debug
        self.api_index = 0
        self.single_api_index_dict = {}
        self.device_dump_path_cpu = None
        self.device_dump_path_npu = None
        self.all_summary = []
        self.call_stack_list = []
        self.process_num = process_num
        self.tag = tag
        self.check_param()
        self.filter_dump_api()
        dir_name = self.get_dir_name(tag)
        self.root_path = os.path.join(os.path.realpath(dump_path), dir_name)
        self.root_cpu_path = os.path.join(self.root_path, f'cpu')
        self.root_npu_path = os.path.join(self.root_path, f'npu')
        create_directory(self.root_cpu_path)
        create_directory(self.root_npu_path)

        self.result_csv_path = os.path.join(self.root_path, RESULT_FILE_NAME)
        self.detail_csv_path = os.path.join(self.root_path, DETAILS_FILE_NAME)
        self.comparator = Comparator(self.result_csv_path, self.detail_csv_path, False)

        self.aten_ops_blacklist = []
        self.npu_adjust_autograd = []
        yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "torch_ops_config.yaml")
        self.get_ops(yaml_path)

        self.lock = Lock() if process_num > 0 else None
        max_process_num = max(int((multiprocessing.cpu_count() + 1) // Const.CPU_QUARTER), 1)
        if process_num > max_process_num:
            logger.error(f"process_num should be less than or equal to {max_process_num}, but got {process_num}!")
            raise DispatchException(f'process_num should be less than or equal to {max_process_num}, '
                                    f'but got {process_num}!')
        if process_num > 0:
            self.pool = Pool(process_num, initializer=self._init_child_process, initargs=(self.lock,))
        if debug:
            logger.info(f'Main pid:{os.getpid()} device:{self.device_id} dump_list:{self.dump_api_list} '
                        f'dump_mode:{self.dump_mode} cpu_path[{self.root_cpu_path}], npu_path[{self.root_npu_path}], '
                        f'process[{process_num}]')

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

        if not is_npu:
            return
        logger.info(f'start write compare csv: Rank[{self.device_id}], Pid[{os.getpid()}]')

        if self.process_num > 0:
            self.pool.close()
            self.pool.join()
            summary_path = os.path.join(self.root_cpu_path, f'summary.json')
            if not os.path.exists(summary_path):
                logger.error("Please check train log, An exception may have occurred!")
                return
            check_file_or_directory_path(summary_path, False)
            with FileOpen(summary_path, "r") as fp_handle:
                while True:
                    json_line_data = fp_handle.readline()
                    if json_line_data == '\n':
                        continue
                    if len(json_line_data) == 0:
                        break
                    msg = json.loads(json_line_data)
                    if len(msg) < 2:
                        raise ValueError("JSON data does not contain enough elements. Expected at least 2 elements.")
                    self.all_summary[msg[0]] = msg[1]

        if self.debug_flag:
            input_num = 0
            output_num = 0
            total_num = 0

            for list_data in self.all_summary:
                for data in list_data:
                    logger.info(f'summary: Device[{self.device_id}], Pid[{os.getpid()}], Data[{data}]')
                    if "_input" in data[CompareConst.NPU_NAME]:
                        input_num = input_num + 1
                    if "_output" in data[CompareConst.NPU_NAME]:
                        output_num = output_num + 1
                    total_num = total_num + 1
            logger.info(f'Dispatch exit: Device[{self.device_id}], Pid[{os.getpid()} Input[{input_num}] '
                        f'Output[{output_num}] Total[{total_num}] API_Total[{self.api_index}]]')

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not is_npu:
            logger.error("Please confirm you run environment installed torch_npu!")
            return func(*args, **kwargs)

        func_name_split_list = func.__name__.split(".")
        aten_api = func_name_split_list[0]
        try:
            aten_api_overload_name = func_name_split_list[1]
        except IndexError:
            logger.error(f"Please check the func name {func.__name__}!")
            return func(*args, **kwargs)

        self.enable_autograd(aten_api)
        if aten_api in self.aten_ops_blacklist:
            npu_out = func(*args, **kwargs)
            return npu_out

        call_stack = get_callstack()
        self.call_stack_list.append(call_stack)

        self.lock.acquire() if self.process_num > 0 else None
        try:
            self.api_index += 1
            if aten_api not in self.single_api_index_dict:
                self.single_api_index_dict[aten_api] = 1
            else:
                self.single_api_index_dict[aten_api] += 1
        finally:
            self.lock.release() if self.process_num > 0 else None

        run_param = self.get_run_param(aten_api, func.__name__, aten_api_overload_name)

        if self.debug_flag:
            logger.info(f'Dispatch Info: Rank[{self.device_id}], Pid[{os.getpid()}], Func[{func.__name__}], '
                        f'Name[{run_param.aten_api}_{run_param.single_api_index}], '
                        f'Count[{self.api_index}], Sys[{get_sys_info()}]')

        cpu_args = []
        cpu_kwargs = []
        data_to_cpu(args, 0, cpu_args)
        data_to_cpu(kwargs, 0, cpu_kwargs)

        cpu_args = safe_get_value(cpu_args, 0, "cpu_args")
        cpu_kwargs = safe_get_value(cpu_kwargs, 0, "cpu_kwargs")

        with TimeStatistics("NPU RUN", run_param):
            npu_out = func(*args, **kwargs)
        npu_out_cpu = []
        data_to_cpu(npu_out, 0, npu_out_cpu)
        npu_out_cpu = safe_get_value(npu_out_cpu, 0, "npu_out_cpu")

        with TimeStatistics("CPU RUN", run_param):
            try:
                cpu_out = func(*cpu_args, **cpu_kwargs)
            except RuntimeError as e:
                self.lock.acquire() if self.process_num > 0 else None
                try:
                    self.api_index -= 1
                    self.single_api_index_dict[aten_api] -= 1
                finally:
                    self.lock.release() if self.process_num > 0 else None
                logger.warning(f"RuntimeError: {e}")
                logger.warning(f"This aten_api {aten_api} does not support running on cpu, so skip it.")
                return npu_out

        if isinstance(cpu_out, torch.Tensor) and cpu_out.dtype in [torch.bfloat16, torch.float16, torch.half]:
            cpu_out = cpu_out.float()

        if self.process_num == 0:
            self.all_summary.append([])
            data_info = DisPatchDataInfo(cpu_args, cpu_kwargs, self.all_summary, func, npu_out_cpu, cpu_out, self.lock)
            dispatch_workflow(run_param, data_info)
        else:
            self.lock.acquire()
            try:
                self.all_summary.append([])
            finally:
                self.lock.release()
            run_param.process_flag = True
            if self.check_fun(func, run_param):
                data_info = DisPatchDataInfo(cpu_args, cpu_kwargs, self.all_summary, None, npu_out_cpu, cpu_out,
                                             child_global_lock)
                self.pool.apply_async(func=dispatch_multiprocess, args=(run_param, data_info),
                                      error_callback=error_call)
            else:
                logger.error("can not get correct function please set process_num=0")
        return npu_out

    @staticmethod
    def check_fun(func, run_param):
        if hasattr(torch.ops.aten, run_param.aten_api):
            aten_func = getattr(torch.ops.aten, run_param.aten_api)
            if hasattr(aten_func, run_param.aten_api_overload_name):
                aten_overload_func = getattr(aten_func, run_param.aten_api_overload_name)
                if id(aten_overload_func) == id(func):
                    run_param.func_namespace = "aten"
                    return True
        return False

    @staticmethod
    def _init_child_process(lock):
        global child_global_lock
        child_global_lock = lock

    def get_dir_name(self, tag):
        # guarantee file uniqueness
        time.sleep(1)
        # 时间格式：年-月-日-时-分-秒-毫秒（精确到千分之一秒）
        time_now = time.strftime("%Y%m%d%H%M%S%f", time.localtime(time.time()))[:-3]  # 取前3位毫秒

        if tag is None or not isinstance(tag, str):
            logger.warning('There is not tag or the type of tag is not string.')
            # 目录名格式：msprobe_rank{设备ID}_{毫秒时间戳}
            dir_name = f'msprobe_rank{self.device_id}_{time_now}'
        else:
            dir_name = f'msprobe_{tag}_rank{self.device_id}_{time_now}'
        return dir_name

    def get_ops(self, file_path):
        yaml_file = load_yaml(file_path)
        self.aten_ops_blacklist = yaml_file.get('aten_ops_blacklist')
        self.npu_adjust_autograd = yaml_file.get('npu_adjust_autograd')

    def filter_dump_api(self):
        if self.dump_mode != Const.LIST or not self.dump_api_list:
            self.dump_api_list = []
            return
        aten_api_list = dir(torch.ops.aten)
        dump_api_list = []
        for aten_api in self.dump_api_list:
            if aten_api in aten_api_list:
                dump_api_list.append(aten_api)
            else:
                logger.warning(f'{aten_api} is not aten api will not dump, please refer to torch.ops.aten')
        self.dump_api_list = dump_api_list

    def get_run_param(self, aten_api, func_name, aten_api_overload_name):
        run_param = DispatchRunParam(self.debug_flag, self.device_id, self.root_npu_path, self.root_cpu_path,
                                     self.process_num, self.comparator)
        run_param.dump_flag, run_param.auto_dump_flag = self.get_dump_flag(aten_api)
        run_param.func_name = func_name
        run_param.aten_api = aten_api
        run_param.aten_api_overload_name = aten_api_overload_name
        run_param.single_api_index = self.single_api_index_dict[aten_api]
        run_param.api_index = self.api_index
        return run_param

    def get_dump_flag(self, aten_api):
        dump_flag = False
        auto_dump_flag = False
        if self.dump_mode == Const.ALL:
            dump_flag = True
        if self.dump_mode == Const.LIST and aten_api in self.dump_api_list:
            dump_flag = True
        if self.dump_mode == Const.AUTO:
            auto_dump_flag = True
        return dump_flag, auto_dump_flag

    def check_param(self):
        if self.dump_mode not in Const.ONLINE_DUMP_MODE:
            logger.error('The parameter "dump mode" can only be one of {}.'.format(Const.ONLINE_DUMP_MODE))
            raise DispatchException(DispatchException.INVALID_PARAMETER)
        if not isinstance(self.dump_api_list, list):
            logger.error('The type of parameter "api_list" can only be list.')
            raise DispatchException(DispatchException.INVALID_PARAMETER)
        if not all(isinstance(item, str) for item in self.dump_api_list):
            logger.error('The type of parameter in "api_list" can only be str.')
            raise DispatchException(DispatchException.INVALID_PARAMETER)
        if len(self.dump_api_list) > Const.STEP_RANK_MAXIMUM_VALUE:
            logger.error('The length of parameter "api_list" should not be greater '
                         f'than {Const.STEP_RANK_MAXIMUM_VALUE}.')
            raise DispatchException(DispatchException.INVALID_PARAMETER)
        for item in self.dump_api_list:
            check_str_param(item)
        if self.tag is not None:
            check_str_param(self.tag)
        if not isinstance(self.debug_flag, bool):
            logger.error('The type of parameter "debug" can only be bool.')
            raise DispatchException(DispatchException.INVALID_PARAMETER)
        if not isinstance(self.process_num, int) or self.process_num < 0:
            logger.error('The type of parameter "process_num" can only be int and it should not be less than 0.')
            raise DispatchException(DispatchException.INVALID_PARAMETER)

    def enable_autograd(self, aten_api):
        if aten_api in self.npu_adjust_autograd:
            torch._C._dispatch_tls_set_dispatch_key_excluded(torch._C.DispatchKey.AutogradFunctionality, False)
