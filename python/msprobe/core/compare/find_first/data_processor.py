# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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


from python.msprobe.core.common.const import Const
from python.msprobe.core.common.log import logger


class DataProcessor:
    def __init__(self, data_frame):
        self.data_frame = data_frame
        if self.data_frame == Const.PT_FRAMEWORK:
            from python.msprobe.pytorch.compare.distributed_compare import compare_distributed
            self.process_func = compare_distributed
        elif self.data_frame == Const.MS_FRAMEWORK:
            from python.msprobe.mindspore.compare.distributed_compare import ms_compare_distributed
            self.process_func = ms_compare_distributed
        else:
            raise ValueError(f"Unsupported data_frame: {self.data_frame}")

    def process(self, npu_path, bench_path, output_path):
        logger.info("Start comparing data ......")
        return self.process_func(npu_path, bench_path, output_path, first_diff_analyze=True)
