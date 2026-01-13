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

import os

from msprobe.core.common.file_utils import create_directory, check_path_readability, check_path_writability
from msprobe.core.common.const import Const
from msprobe.core.common.log import logger



def try_import_torchair():
    try:
        import torch
        import torch_npu
        import torchair
    except ModuleNotFoundError as e:
        logger.error("torch or torch_npu with torchair not found. Try installing the latest torch_npu.")
        raise e


def set_ge_dump_config(
        dump_path='',
        dump_mode='all',
        fusion_switch_file=None,
        dump_token=None,
        dump_layer=None,
        compiler_config=None
):
    try_import_torchair()
    from torchair.configs.compiler_config import CompilerConfig

    dump_path = os.path.realpath(os.path.join(dump_path, Const.TORCHAIR_GE_DATA_DIRECTORY))
    create_directory(dump_path)
    check_path_readability(dump_path)
    check_path_writability(dump_path)
    if os.listdir(dump_path):
        logger.warning('The specified directory for GE-Dump data is not empty, which may result in data mixing.')

    if compiler_config is not None and not isinstance(compiler_config, CompilerConfig):
        raise TypeError(f'compiler_config must be a CompilerConfig, but got {type(compiler_config)}')
    config = compiler_config if compiler_config else CompilerConfig()
    # Generate GE mapping graph
    config.debug.graph_dump.type = "txt"
    if hasattr(config.debug.graph_dump, "_path"):  # interface changed since 8.0.RC1.b080
        setattr(config.debug.graph_dump, "_path", dump_path)
    else:
        config.debug.graph_dump.path = dump_path

    if fusion_switch_file is not None:
        if not os.path.exists(fusion_switch_file):
            raise FileNotFoundError(f'fusion_switch_file: {fusion_switch_file} not found')
        config.fusion_config.fusion_switch_file = fusion_switch_file

    # Enable GE dump
    config.dump_config.enable_dump = True
    config.dump_config.dump_mode = dump_mode
    config.dump_config.dump_path = dump_path

    if dump_token is not None:
        new_token = [str(x) for x in dump_token]
        config.dump_config.dump_step = "|".join(new_token)
    if dump_layer is not None:
        dump_layer = " ".join(dump_layer)
        config.dump_config.dump_layer = dump_layer

    return config


def set_fx_dump_config(dump_path='', op_list=None, compiler_config=None):
    try_import_torchair()
    from torchair.configs.compiler_config import CompilerConfig

    dump_path = os.path.realpath(os.path.join(dump_path, Const.TORCHAIR_FX_DATA_DIRECTORY))
    create_directory(dump_path)
    check_path_readability(dump_path)
    check_path_writability(dump_path)
    if os.listdir(dump_path):
        logger.warning('The specified directory for FX-Dump data is not empty, which may result in data mixing.')

    if compiler_config is not None and not isinstance(compiler_config, CompilerConfig):
        raise TypeError(f'compiler_config must be a CompilerConfig, but got {type(compiler_config)}')
    config = compiler_config if compiler_config else CompilerConfig()
    if op_list:
        if not isinstance(op_list, list) or not all(isinstance(item, str) for item in op_list):
            raise TypeError("op_list must be a list of strings")
        from msprobe.infer.utils.util import filter_cmd
        op_list = filter_cmd(op_list)
        config.debug.data_dump.filter = (
            lambda x: x
            if any(keyword.lower() in x.name.lower() for keyword in op_list)
            else None
        )
        

    # Enable FX dump
    config.debug.data_dump.type = "npy"
    if hasattr(config.debug.data_dump, "path"):
        config.debug.data_dump.path = dump_path

    return config
