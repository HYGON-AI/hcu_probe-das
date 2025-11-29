#!/bin/bash
#
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
            export LD_PRELOAD=${ASCEND_HOME_PATH}/tools/ait_backend/dump/libatb_probe_abi${cxx_abi}.so:$LD_PRELOAD
            export ATB_OUTPUT_DIR=$output_path
            export ATB_DUMP_CONFIG=$config_path
        fi
    fi
fi
