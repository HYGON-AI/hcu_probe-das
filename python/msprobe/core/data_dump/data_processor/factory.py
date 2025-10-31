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
from msprobe.core.data_dump.data_processor.base import BaseDataProcessor


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
            raise ValueError(f"ModuleProcesser not found for framework: {framework}")
        return processor_class

    @classmethod
    def create_processor(cls, config, data_writer):
        cls.register_processors(config.framework)
        task = Const.KERNEL_DUMP if config.level == "L2" else config.task
        key = (config.framework, task)
        processor_class = cls._data_processor.get(key)
        if not processor_class:
            raise ValueError(f"Processor not found for framework: {config.framework}, task: {config.task}")
        return processor_class(config, data_writer)

    @classmethod
    def register_processors(cls, framework):
        if framework == Const.PT_FRAMEWORK:
            from msprobe.core.data_dump.data_processor.pytorch_processor import (
                StatisticsDataProcessor as PytorchStatisticsDataProcessor,
                TensorDataProcessor as PytorchTensorDataProcessor,
                OverflowCheckDataProcessor as PytorchOverflowCheckDataProcessor,
                FreeBenchmarkDataProcessor as PytorchFreeBenchmarkDataProcessor,
                KernelDumpDataProcessor as PytorchKernelDumpDataProcessor
            )
            from msprobe.pytorch.dump.module_dump.module_processer import ModuleProcesser
            cls.register_processor(Const.PT_FRAMEWORK, Const.STATISTICS, PytorchStatisticsDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.TENSOR, PytorchTensorDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.OVERFLOW_CHECK, PytorchOverflowCheckDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.FREE_BENCHMARK, PytorchFreeBenchmarkDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.KERNEL_DUMP, PytorchKernelDumpDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.STRUCTURE, BaseDataProcessor)
            cls.register_module_processor(Const.PT_FRAMEWORK, ModuleProcesser)
        elif framework == Const.MS_FRAMEWORK:
            from msprobe.core.data_dump.data_processor.mindspore_processor import (
                StatisticsDataProcessor as MindsporeStatisticsDataProcessor,
                TensorDataProcessor as MindsporeTensorDataProcessor,
                OverflowCheckDataProcessor as MindsporeOverflowCheckDataProcessor,
                KernelDumpDataProcessor as MindsporeKernelDumpDataProcessor
            )
            from msprobe.mindspore.cell_processor import CellProcessor
            cls.register_processor(Const.MS_FRAMEWORK, Const.STATISTICS, MindsporeStatisticsDataProcessor)
            cls.register_processor(Const.MS_FRAMEWORK, Const.TENSOR, MindsporeTensorDataProcessor)
            cls.register_processor(Const.MS_FRAMEWORK, Const.OVERFLOW_CHECK, MindsporeOverflowCheckDataProcessor)
            cls.register_processor(Const.MS_FRAMEWORK, Const.KERNEL_DUMP, MindsporeKernelDumpDataProcessor)
            cls.register_processor(Const.MS_FRAMEWORK, Const.STRUCTURE, BaseDataProcessor)
            cls.register_module_processor(Const.MS_FRAMEWORK, CellProcessor)
