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

from msprobe.core.common.file_utils import check_file_type, check_file_or_directory_path
from msprobe.core.common.const import FileCheckConst, Const
from msprobe.core.common.utils import CompareException
from msprobe.core.common.log import logger
from msprobe.core.compare.utils import get_paired_dirs
from msprobe.core.compare.utils import get_compare_framework
from msprobe.core.compare.utils import compare_distributed_inner
from msprobe.core.compare.mode_dispatcher import dispatch_compare_mode


def compare_cli(args, depth=1):
    """
    Main comparison CLI entry point
    Uses mode-based dispatch system
    """
    return dispatch_compare_mode(args, depth)

