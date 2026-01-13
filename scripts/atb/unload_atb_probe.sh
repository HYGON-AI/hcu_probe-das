#!/bin/bash
#
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

if [ -n "$LD_PRELOAD" ]; then
    var=$(echo ":$LD_PRELOAD" | sed 's|:[^:]*libatb_probe_abi0\.so|:|g')
    var=$(echo ":$var" | sed 's|:[^:]*libatb_probe_abi1\.so|:|g')
    var=$(echo $var | sed 's|:[:]*:|:|g')
    var=$(echo $var | sed 's|^:[:]*||')
    export LD_PRELOAD=${var}
fi
unset ATB_OUTPUT_DIR
unset ATB_DUMP_CONFIG
