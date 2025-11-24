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

from msprobe.core.common.const import Const
from msprobe.core.dump.data_dump.data_processor.base import BaseDataProcessor
from msprobe.core.common.file_utils import check_file_or_directory_path
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException


class DataProcessorFactory:
    _data_processor = {}
    _module_processor = {}

    @classmethod
    def register_processor(cls, framework, task, processor_class):
        key = (framework, task)
        cls._data_processor[key] = processor_class

    @classmethod
    def register_module_processor(cls, framework, processor_class):
        cls._module_processor[framework] = processor_class

    @classmethod
    def get_module_processor(cls, framework):
        processor_class = cls._module_processor.get(framework)
        if not processor_class:
            raise ValueError(f"ModuleProcessor not found for framework: {framework}")
        return processor_class

    @classmethod
    def create_processor(cls, config, data_writer):
        cls.register_processors(config.framework)
        task = Const.KERNEL_DUMP if config.level == "L2" else config.task
        key = (config.framework, task)
        bench_path = getattr(config, "bench_path", None)
        if config.task == Const.TENSOR and bench_path is not None and not isinstance(bench_path, str):
            logger.error_log_with_exp("bench_path is invalid, it should be a string",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        elif config.task == Const.TENSOR and bench_path is not None and isinstance(bench_path, str):
            check_file_or_directory_path(bench_path, True)
            processor_class = cls._data_processor.get(("pytorch", Const.DIFF_CHECK))
        else:
            processor_class = cls._data_processor.get(key)
        if not processor_class:
            raise ValueError(f"Processor not found for framework: {config.framework}, task: {config.task}")
        return processor_class(config, data_writer)

    @classmethod
    def register_processors(cls, framework):
        if framework == Const.PT_FRAMEWORK:
            from msprobe.core.dump.data_dump.data_processor.pytorch_processor import (
                StatisticsDataProcessor as PytorchStatisticsDataProcessor,
                TensorDataProcessor as PytorchTensorDataProcessor,
                DiffCheckDataProcessor as PytorchDiffCheckDataProcessor,
                KernelDumpDataProcessor as PytorchKernelDumpDataProcessor
            )
            from msprobe.pytorch.dump.module_dump.module_processor import ModuleProcessor
            cls.register_processor(Const.PT_FRAMEWORK, Const.STATISTICS, PytorchStatisticsDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.TENSOR, PytorchTensorDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.KERNEL_DUMP, PytorchKernelDumpDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.DIFF_CHECK, PytorchDiffCheckDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.STRUCTURE, BaseDataProcessor)
            cls.register_module_processor(Const.PT_FRAMEWORK, ModuleProcessor)
        elif framework == Const.MS_FRAMEWORK:
            from msprobe.core.dump.data_dump.data_processor.mindspore_processor import (
                StatisticsDataProcessor as MindsporeStatisticsDataProcessor,
                TensorDataProcessor as MindsporeTensorDataProcessor,
                KernelDumpDataProcessor as MindsporeKernelDumpDataProcessor
            )
            from msprobe.mindspore.dump.cell_processor import CellProcessor
            cls.register_processor(Const.MS_FRAMEWORK, Const.STATISTICS, MindsporeStatisticsDataProcessor)
            cls.register_processor(Const.MS_FRAMEWORK, Const.TENSOR, MindsporeTensorDataProcessor)
            cls.register_processor(Const.MS_FRAMEWORK, Const.KERNEL_DUMP, MindsporeKernelDumpDataProcessor)
            cls.register_processor(Const.MS_FRAMEWORK, Const.STRUCTURE, BaseDataProcessor)
            cls.register_module_processor(Const.MS_FRAMEWORK, CellProcessor)
