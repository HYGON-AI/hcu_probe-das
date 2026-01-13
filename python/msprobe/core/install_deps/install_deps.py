# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

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
