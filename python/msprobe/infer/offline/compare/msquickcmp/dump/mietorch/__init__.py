# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from msprobe.infer.offline.compare.msquickcmp.common.args_check import safe_string, check_cann_path_legality


ascend_toolkit_home_path = os.getenv("ASCEND_TOOLKIT_HOME")
if not ascend_toolkit_home_path:
    raise EnvironmentError("Please first source CANN environment by running set_env.sh.")
ascend_toolkit_home_path = safe_string(ascend_toolkit_home_path)
ascend_toolkit_home_path = check_cann_path_legality(ascend_toolkit_home_path)  # check cann path

path_components = ["tools", "ait_backend", "mindie_torch_dump", "libmindiedump.so"]
mindie_rt_dump_so_path = os.path.join(ascend_toolkit_home_path, *path_components)

cur_dir = os.path.dirname(os.path.abspath(__file__))
mindie_rt_dump_config = os.path.join(cur_dir, "acl.json")
ld_preload = os.getenv("LD_PRELOAD")
if ld_preload:
    os.environ["LD_PRELOAD"] = f'{mindie_rt_dump_so_path}:{ld_preload}'
else:
    os.environ["LD_PRELOAD"] = mindie_rt_dump_so_path

os.environ["MINDIE_RT_DUMP_CONFIG_PATH"] = mindie_rt_dump_config
