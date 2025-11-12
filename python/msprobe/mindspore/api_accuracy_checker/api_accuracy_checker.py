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
from dataclasses import dataclass
from typing import Any, Optional
from tqdm import tqdm

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.file_utils import FileOpen, create_directory, write_csv, load_json, load_yaml
from msprobe.core.common.utils import add_time_as_suffix
from msprobe.mindspore.api_accuracy_checker.api_info import ApiInfo
from msprobe.mindspore.api_accuracy_checker.api_runner import api_runner, ApiInputAggregation
from msprobe.mindspore.api_accuracy_checker.base_compare_algorithm import compare_algorithms
from msprobe.mindspore.api_accuracy_checker.data_manager import DataManager
from msprobe.mindspore.api_accuracy_checker.utils import (check_and_get_from_json_dict, global_context,
                                                          trim_output_compute_element_list)
from msprobe.mindspore.common.const import MsCompareConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.api_accuracy_checker import torch_mindtorch_importer
from msprobe.core.data_dump.data_collector import build_data_collector
from msprobe.core.common.utils import Const, print_tools_ends_info, DumpPathAggregation
from msprobe.core.data_dump.data_processor.base import ModuleForwardInputsOutputs, ModuleBackwardInputsOutputs

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, MsCompareConst.SUPPORTED_API_LIST_FILE)


class BasicInfoAndStatus:
    def __init__(self, api_name, bench_dtype, tested_dtype, shape, status, err_msg) -> None:
        self.api_name = api_name
        self.bench_dtype = bench_dtype
        self.tested_dtype = tested_dtype
        self.shape = shape
        self.status = status
        self.err_msg = err_msg


class ResultCsvEntry:
    def __init__(self) -> None:
        self.forward_pass_status = None
        self.backward_pass_status = None
        self.forward_err_msg = ""
        self.backward_err_msg = ""
        self.overall_err_msg = None


class ProcessResultPacket:
    def __init__(self, process_status, result, err_msg) -> None:
        self.process_status = process_status
        self.result = result
        self.err_msg = err_msg


@dataclass
class Config:
    execution_mode: str
    dump_path: str
    task: str
    level: str
    scope: Optional[Any]
    list: Optional[Any]
    framework: str
    data_mode: str
    file_format: str
    dump_tensor_data_dir: str
    async_dump: bool
    summary_mode: Optional[Any] = None


class ApiAccuracyChecker:
    def __init__(self, args):
        self.api_infos = dict()
        self.data_manager = DataManager(args.out_path, args.result_csv_path)  # 在初始化时实例化 DataManager
        self.save_error_data = args.save_error_data
        if self.save_error_data:
            config, dump_path_aggregation = self.init_save_error_data(args)
            self.data_collector = build_data_collector(config)
            self.data_collector.update_dump_paths(dump_path_aggregation)

    @staticmethod
    def init_save_error_data(args):
        config = Config(
            execution_mode="pynative",
            dump_path=f"{args.out_path}",
            dump_tensor_data_dir=f"{args.out_path}",
            task="tensor",  # 任务类型,模拟保存tensor数据
            level="L1",  # 级别
            scope=None,  # 作用域 (None)
            list=None,  # API 列表 (None)
            framework=Const.MS_FRAMEWORK,  # 框架类型
            data_mode="all",
            file_format="npy",
            async_dump=False
        )

        dump_dir = f"{args.out_path}"
        dump_data_dir = os.path.join(dump_dir, "error_data")
        create_directory(dump_data_dir)
        dump_path_aggregation = DumpPathAggregation()
        dump_path_aggregation.dump_file_path = os.path.join(dump_dir, "dump.json")
        dump_path_aggregation.stack_file_path = os.path.join(dump_dir, "stack.json")
        dump_path_aggregation.dump_error_info_path = os.path.join(dump_dir, "dump_error_info.log")
        dump_path_aggregation.dump_tensor_data_dir = dump_data_dir
        return config, dump_path_aggregation

    @staticmethod
    def prepare_api_input_aggregation(api_info, forward_or_backward=Const.FORWARD):
        """
        Args:
            api_info: ApiInfo
            forward_or_backward: str
        Returns:
            ApiInputAggregation
        """
        forward_inputs = api_info.get_compute_element_list(Const.FORWARD, Const.INPUT)
        kwargs = api_info.get_kwargs()
        if forward_or_backward == Const.FORWARD:
            gradient_inputs = None
        else:
            gradient_inputs = api_info.get_compute_element_list(Const.BACKWARD, Const.INPUT)
        return ApiInputAggregation(forward_inputs, kwargs, gradient_inputs)

    @staticmethod
    def is_api_checkable(api_name_str):
        '''
        Args:
            api_name_str: str, e.g. "MintFunctional.relu.0.forward", key in data field of api_info.json
        Returns:
            is_checkable: bool
        Description:
            tell whether this api is checkable based on the key in "data" dict in api_info.json
        '''
        api_name_str_list = api_name_str.split(Const.SEP)
        if len(api_name_str_list) < MsCompareConst.API_NAME_STR_LENGTH:
            return False
        api_type_str = api_name_str_list[0]
        real_api_str = Const.SEP.join(api_name_str_list[1:-2])
        api_list = load_yaml(yaml_path)
        supported_tensor_api_list = api_list.get(MsCompareConst.SUPPORTED_TENSOR_LIST_KEY)
        supported_fusion_api_list = MsCompareConst.SUPPORTED_FUSION_LIST
        if api_type_str in (MsCompareConst.MINT, MsCompareConst.MINT_FUNCTIONAL) \
                and global_context.get_framework() == Const.MS_FRAMEWORK:
            return True
        if api_type_str in MsCompareConst.MT_VALID_API_TYPES \
                and global_context.get_framework() == Const.MT_FRAMEWORK:
            return True
        if api_type_str == MsCompareConst.TENSOR_API and real_api_str in supported_tensor_api_list \
                and global_context.get_framework() == Const.MS_FRAMEWORK:
            return True
        if api_type_str == MsCompareConst.FUNCTIONAL_API and real_api_str in supported_fusion_api_list \
                and global_context.get_framework() == Const.MS_FRAMEWORK:
            return True
        return False

    def post_forward_hook(self, api_or_module_name, primitive_instance, args, kwargs, output):
        self.data_collector.update_api_or_module_name(api_or_module_name)
        module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=output)
        self.data_collector.forward_data_collect_only_tensor(
            api_or_module_name,
            primitive_instance,
            os.getpid(),
            module_input_output
        )

    def backward_hook(self, api_or_module_name, module, grad_input, grad_output):
        self.data_collector.update_api_or_module_name(api_or_module_name)

        module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_output, grad_output=grad_input)
        self.data_collector.backward_data_collect_only_tensor(
            api_or_module_name,
            module,
            os.getpid(),
            module_input_output
        )

    def run_and_compare_helper(self, api_info, api_name_str, api_input_aggregation, forward_or_backward):
        """
        Args:
            api_info: ApiInfo
            api_name_str: str
            api_input_aggregation: ApiInputAggregation
            forward_or_backward: str: Union["forward", "backward"]

        Return:
            output_list: List[tuple(str, str, BasicInfoAndStatus, dict{str: CompareResult})]

        Description:
            get mindspore api output, run torch api and get output.
            compare output.
            record compare result.
        """
        # get output
        if global_context.get_is_constructed():
            if forward_or_backward == Const.FORWARD:
                tested_outputs, inputs, kwargs, forward_result_tuple = api_runner(api_input_aggregation, api_name_str,
                                                                                  forward_or_backward,
                                                                                  global_context.get_framework())
            elif forward_or_backward == Const.BACKWARD:
                tested_outputs, gradient_inputs, backward_result_tuple = api_runner(api_input_aggregation, api_name_str,
                                                                                    forward_or_backward,
                                                                                    global_context.get_framework())
            else:
                tested_outputs = api_runner(api_input_aggregation, api_name_str,
                                            forward_or_backward, global_context.get_framework())
        else:
            tested_outputs = api_info.get_compute_element_list(forward_or_backward, Const.OUTPUT)

        bench_outputs = api_runner(api_input_aggregation, api_name_str, forward_or_backward, Const.PT_FRAMEWORK)

        tested_outputs = trim_output_compute_element_list(tested_outputs, forward_or_backward)
        bench_outputs = trim_output_compute_element_list(bench_outputs, forward_or_backward)
        if len(tested_outputs) != len(bench_outputs):
            logger.warning(f"ApiAccuracyChecker.run_and_compare_helper: api: {api_name_str}.{forward_or_backward}, "
                           "number of bench outputs and tested outputs is different, comparing result can be wrong. "
                           f"tested outputs: {len(tested_outputs)}, bench outputs: {len(bench_outputs)}")

        # compare output
        output_list = []
        for i, (bench_out, tested_out) in enumerate(zip(bench_outputs, tested_outputs)):
            api_name_with_slot = Const.SEP.join([api_name_str, forward_or_backward, Const.OUTPUT, str(i)])
            bench_dtype = bench_out.get_dtype()
            tested_dtype = tested_out.get_dtype()
            shape = bench_out.get_shape()

            compare_result_dict = dict()
            for compare_algorithm_name, compare_algorithm in compare_algorithms.items():
                compare_result = compare_algorithm(bench_out, tested_out)
                compare_result_dict[compare_algorithm_name] = compare_result

            if compare_result_dict.get(CompareConst.COSINE).pass_status == CompareConst.PASS and \
                    compare_result_dict.get(CompareConst.MAX_ABS_ERR).pass_status == CompareConst.PASS:
                status = CompareConst.PASS
                err_msg = ""

            else:
                status = CompareConst.ERROR
                err_msg = (compare_result_dict.get(CompareConst.COSINE).err_msg +
                           compare_result_dict.get(CompareConst.MAX_ABS_ERR).err_msg)
                if forward_or_backward == Const.FORWARD and self.save_error_data \
                        and global_context.get_is_constructed():
                    api_name_str_backward = f"{api_name_str}{Const.SEP}{Const.FORWARD}"
                    self.post_forward_hook(api_name_str_backward, None, inputs, kwargs, forward_result_tuple)

                if forward_or_backward == Const.BACKWARD and self.save_error_data \
                        and global_context.get_is_constructed():
                    api_name_str_backward = f"{api_name_str}{Const.SEP}{Const.BACKWARD}"
                    self.backward_hook(api_name_str_backward, None, gradient_inputs, backward_result_tuple)

            basic_info_status = \
                BasicInfoAndStatus(api_name_with_slot, bench_dtype, tested_dtype, shape, status, err_msg)
            output_list.append(tuple([api_name_str, forward_or_backward, basic_info_status, compare_result_dict]))
        return output_list

    def parse(self, api_info_path):

        api_info_dict = load_json(api_info_path)

        # init global context
        task = check_and_get_from_json_dict(api_info_dict, MsCompareConst.TASK_FIELD,
                                            "task field in api_info.json", accepted_type=str,
                                            accepted_value=(MsCompareConst.STATISTICS_TASK,
                                                            MsCompareConst.TENSOR_TASK))
        try:
            framework = check_and_get_from_json_dict(api_info_dict, MsCompareConst.FRAMEWORK,
                                                     "framework field in api_info.json", accepted_type=str,
                                                     accepted_value=(Const.MS_FRAMEWORK,
                                                                     Const.MT_FRAMEWORK))
        except Exception as e:
            framework = Const.MS_FRAMEWORK
            logger.warning(f"JSON parsing error in framework field: {e}")

        if framework == Const.MT_FRAMEWORK and not torch_mindtorch_importer.is_valid_pt_mt_env:
            raise Exception(f"Please check if you have a valid PyTorch and MindTorch environment")

        is_constructed = task == MsCompareConst.STATISTICS_TASK
        if not is_constructed:
            dump_data_dir = check_and_get_from_json_dict(api_info_dict, MsCompareConst.DUMP_DATA_DIR_FIELD,
                                                         "dump_data_dir field in api_info.json", accepted_type=str)
        else:
            dump_data_dir = ""
        global_context.init(is_constructed, dump_data_dir, framework)

        api_info_data = check_and_get_from_json_dict(api_info_dict, MsCompareConst.DATA_FIELD,
                                                     "data field in api_info.json", accepted_type=dict)
        for api_name, api_info in api_info_data.items():
            if not self.is_api_checkable(api_name):
                continue
            forbackward_str = api_name.split(Const.SEP)[-1]
            if forbackward_str not in (Const.FORWARD, Const.BACKWARD):
                logger.warning(f"api: {api_name} is not recognized as forward api or backward api, skip this.")
            api_name = Const.SEP.join(api_name.split(Const.SEP)[:-1])  # www.xxx.yyy.zzz --> www.xxx.yyy
            if api_name not in self.api_infos:
                self.api_infos[api_name] = ApiInfo(api_name)

            if forbackward_str == Const.FORWARD:
                self.api_infos[api_name].load_forward_info(api_info)
            else:
                self.api_infos[api_name].load_backward_info(api_info)

    def process_forward(self, api_name_str, api_info):
        """处理前向检查"""
        if not api_info.check_forward_info():
            logger.debug(f"api: {api_name_str} is lack of forward information, skip forward check.")
            process_result_packet = ProcessResultPacket(process_status=MsCompareConst.ProcessStatus.API_NOT_FOUND,
                                                        result=None,
                                                        err_msg=f"forward info of {api_name_str} is not found")
            return process_result_packet

        try:
            forward_inputs_aggregation = self.prepare_api_input_aggregation(api_info, Const.FORWARD)
        except Exception as e:
            logger.warning(f"Exception occurs when getting inputs for {api_name_str} forward api. "
                           f"Skipping forward check. Detailed exception information: {e}.")
            process_result_packet = ProcessResultPacket(process_status=MsCompareConst.ProcessStatus.EXCEPTION_SKIP,
                                                        result=None, err_msg=f"{e}")
            return process_result_packet

        try:
            forward_output_list = self.run_and_compare_helper(api_info, api_name_str, forward_inputs_aggregation,
                                                              Const.FORWARD)
        except Exception as e:
            logger.warning(f"Exception occurs when running and comparing {api_name_str} forward api. "
                           f"Detailed exception information: {e}.")
            process_result_packet = ProcessResultPacket(process_status=MsCompareConst.ProcessStatus.EXCEPTION_SKIP,
                                                        result=None, err_msg=f"{e}")
            return process_result_packet

        process_result_packet = ProcessResultPacket(process_status=MsCompareConst.ProcessStatus.SUCCESS,
                                                    result=forward_output_list, err_msg="")
        return process_result_packet

    def process_backward(self, api_name_str, api_info):
        """处理反向检查"""
        if not api_info.check_backward_info():
            logger.debug(f"api: {api_name_str} is lack of backward information, skipping backward check.")
            process_result_packet = ProcessResultPacket(process_status=MsCompareConst.ProcessStatus.API_NOT_FOUND,
                                                        result=None,
                                                        err_msg=f"backward info of {api_name_str} is not found")
            return process_result_packet

        try:
            backward_inputs_aggregation = self.prepare_api_input_aggregation(api_info, Const.BACKWARD)
        except Exception as e:
            logger.warning(f"Exception occurs when getting inputs for {api_name_str} backward api. "
                           f"Skipping backward check. Detailed exception information: {e}.")
            process_result_packet = ProcessResultPacket(process_status=MsCompareConst.ProcessStatus.EXCEPTION_SKIP,
                                                        result=None, err_msg=f"{e}")
            return process_result_packet

        try:
            backward_output_list = self.run_and_compare_helper(api_info, api_name_str, backward_inputs_aggregation,
                                                               Const.BACKWARD)
        except Exception as e:
            logger.warning(f"Exception occurs when running and comparing {api_name_str} backward api. "
                           f"Detailed exception information: {e}.")
            process_result_packet = ProcessResultPacket(process_status=MsCompareConst.ProcessStatus.EXCEPTION_SKIP,
                                                        result=None, err_msg=f"{e}")
            return process_result_packet

        process_result_packet = ProcessResultPacket(process_status=MsCompareConst.ProcessStatus.SUCCESS,
                                                    result=backward_output_list, err_msg="")
        return process_result_packet

    def run_and_compare(self):
        for api_name_str, api_info in tqdm(self.api_infos.items()):
            if not self.data_manager.is_unique_api(api_name_str):
                continue

            # 处理前向
            process_result_packet = self.process_forward(api_name_str, api_info)
            if process_result_packet.process_status is MsCompareConst.ProcessStatus.SUCCESS:
                self.data_manager.record(process_result_packet.result)
            elif process_result_packet.process_status == MsCompareConst.ProcessStatus.EXCEPTION_SKIP:
                self.data_manager.record_exception_skip(api_name_str, Const.FORWARD, process_result_packet.err_msg)

            # 处理反向
            process_result_packet = self.process_backward(api_name_str, api_info)
            if process_result_packet.process_status is MsCompareConst.ProcessStatus.SUCCESS:
                self.data_manager.record(process_result_packet.result)
            elif process_result_packet.process_status == MsCompareConst.ProcessStatus.EXCEPTION_SKIP:
                self.data_manager.record_exception_skip(api_name_str, Const.BACKWARD, process_result_packet.err_msg)

            self.data_manager.save_results(api_name_str)