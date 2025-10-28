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

from python.msprobe.core.common.utils import CompareException
from python.msprobe.core.common.file_utils import create_directory
from python.msprobe.core.common.exceptions import FileCheckException
from python.msprobe.mindspore.common.log import logger
from python.msprobe.core.compare.utils import compare_distributed_inner
from python.msprobe.mindspore.compare.ms_graph_compare import GraphMSComparator


def ms_compare_distributed(npu_dump_dir, bench_dump_dir, output_path, **kwargs):
    compare_distributed_inner(npu_dump_dir, bench_dump_dir, output_path, **kwargs)


def ms_graph_compare(input_param, outputs):
    try:
        create_directory(outputs)
    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        return
    ms_comparator = GraphMSComparator(input_param, outputs)
    ms_comparator.compare_core()
