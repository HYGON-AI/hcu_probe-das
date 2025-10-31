# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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


from msprobe.pytorch.compare.distributed_compare import compare_distributed


def pt_diff_analyze(npu_dump_dir, bench_dump_dir, output_path, first_diff_analyze):
    compare_distributed(npu_dump_dir, bench_dump_dir, output_path, first_diff_analyze=first_diff_analyze)
