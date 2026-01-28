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

function get_cxx_abi_option()
{
    cxx_abi=""
    if [[ "$ATB_HOME_PATH" =~ cxx_abi_0$ ]]; then
        cxx_abi=0
    else
        if [[ "$ATB_HOME_PATH" =~ cxx_abi_1$ ]]; then
            cxx_abi=1
        else
            echo "[WARNING] The running environment of ATB model is not ready. ATB probe would not work"
        fi
    fi
}

function get_output_option()
{
    output_path="$(pwd)"
    if [ -n "$1" ]; then
        if [[ "$1" =~ ^--output=.* ]]; then
            var="$1"
            output_path="${var#*=}"
        fi
    fi
    if [ -n "$2" ]; then
        if [[ "$2" =~ ^--output=.* ]]; then
            var="$2"
            output_path="${var#*=}"
        fi
    fi
    left="${output_path%%/*}"
    right="${output_path#*/}"
    if [ -n "$left" ]; then
        if [ "$left" == "$output_path" ]; then
            output_path=$(realpath $output_path)
        else
            output_path="$(realpath $left)"/${right}
        fi
    fi
}

function get_config_option()
{
    config_path=$(dirname $(realpath "${BASH_SOURCE[0]}"))/config.json
    if [ -n "$1" ]; then
        if [[ "$1" =~ ^--config=.* ]]; then
            var="$1"
            config_path="${var#*=}"
        fi
    fi
    if [ -n "$2" ]; then
        if [[ "$2" =~ ^--config=.* ]]; then
            var="$2"
            config_path="${var#*=}"
        fi
    fi
    left="${config_path%%/*}"
    right="${config_path#*/}"
    if [ -n "$left" ]; then
        if [ "$left" == "$config_path" ]; then
            config_path=$(realpath $config_path)
        else
            config_path="$(realpath $left)"/${right}
        fi
    fi
}

if [ $# -gt 2 ]; then
    echo "[WARNING] The number of parameters should not be greater than 2, but got $#. ATB probe would not work"
else
    if [ -z "$ASCEND_HOME_PATH" ]; then
        echo "[WARNING] ASCEND_HOME_PATH enviroment variable is not set. ATB probe would not work"
    else
        get_cxx_abi_option
        if [ -n "$cxx_abi" ]; then
            get_output_option "$@"
            get_config_option "$@"
            lib_path=$(realpath $(dirname $(realpath "${BASH_SOURCE[0]}"))/../../lib)
            export LD_PRELOAD=${lib_path}/libatb_probe_abi${cxx_abi}.so:$LD_PRELOAD
            export ATB_OUTPUT_DIR=$output_path
            export ATB_DUMP_CONFIG=$config_path
        fi
    fi
fi
