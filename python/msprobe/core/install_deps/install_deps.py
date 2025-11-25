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

from msprobe.core.common.log import logger
from msprobe.infer.offline.compare.msquickcmp.main import install_offline_deps_cli


def _install_deps_parser(parser):
    parser.add_argument("-m", "--mode", dest="mode", type=str,
                        help="Install deps mode: 'offline' for offline_dump om and compare offline_model",
                        required=True)
    parser.add_argument("--no_check", dest="no_check", action="store_true",
                        help="<optional> Whether to skip checking the target website's certificate information "
                             "when installing dependency packages poses a certain security risk. "
                             "This poses a certain security risk, "
                             "and users should use it with caution and bear the consequences themselves.",
                        required=False)


MODE_DISPATCHER = {
    'offline': install_offline_deps_cli
}


def install_deps_cli(args):
    """
    Dispatch install deps based on mode parameter
    """
    mode = getattr(args, "mode", None)
    if not mode:
        logger.error("No mode specified, please check.")
    # Get the appropriate function based on mode
    install_deps_func = MODE_DISPATCHER.get(mode)
    if install_deps_func is None:
        logger.error(f"Invalid install deps mode: '{mode}'. Available modes: {list(MODE_DISPATCHER.keys())}")
        raise RuntimeError
    # Execute the install_deps function
    return install_deps_func(args)
