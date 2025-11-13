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


from msprobe.core.common.utils import CompareException
from msprobe.core.common.log import logger
from msprobe.core.compare.auto_compare import compare_auto_mode
from msprobe.core.compare.torchair_acc_cmp import compare_torchair_mode


MODE_DISPATCHER = {
    'auto': compare_auto_mode,
    'torchair': compare_torchair_mode,
}


def compare_cli(args):
    """
    Dispatch comparison based on mode parameter
    """
    mode = getattr(args, 'mode', 'auto')

    # Get the appropriate function based on mode
    compare_func = MODE_DISPATCHER.get(mode)
    if compare_func is None:
        logger.error(f"Invalid mode '{mode}'. Available modes: {list(MODE_DISPATCHER.keys())}")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)
    
    # Execute the comparison function
    return compare_func(args)
