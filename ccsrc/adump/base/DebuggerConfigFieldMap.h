/*
 * Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>
#include <map>

#include "DebuggerConfig.h"

namespace MindStudioDebugger {

constexpr const char* FRAMEWORK = "framework";
constexpr const char* FRAMEWORK_PYTORCH = "PyTorch";
constexpr const char* FRAMEWORK_MINDSPORE = "MindSpore";

constexpr const char* TASK_STATISTICS = "statistics";
constexpr const char* TASK_DUMP_TENSOR = "tensor";
constexpr const char* TASK_OVERFLOW_CHECK = "overflow_check";
constexpr const char* TASK_FREE_BENCHMARK = "free_benchmark";
constexpr const char* TASK_RUN_UT = "run_ut";
constexpr const char* TASK_GRAD_PROBE = "grad_probe";

constexpr const char* LEVEL0 = "L0";
constexpr const char* LEVEL1 = "L1";
constexpr const char* LEVEL2 = "L2";
constexpr const char* LEVEL_MIX = "mix";

constexpr const char* DIRECTION_FORWARD = "forward";
constexpr const char* DIRECTION_BACKWARD = "backward";
constexpr const char* DIRECTION_BOTH = "both";
constexpr const char* INOUT_INPUT = "input";
constexpr const char* INOUT_OUTPUT = "output";
constexpr const char* INOUT_BOTH = "both";
constexpr const char* DATA_MODE_ALL = "all";

constexpr const char* FREE_BENCHMARK_HANDLER_CHECK = "check";
constexpr const char* FREE_BENCHMARK_HANDLER_FIX = "fix";

constexpr const char* DUMP_FILE_FORMAT_BIN = "bin";
constexpr const char* DUMP_FILE_FORMAT_NPY = "npy";

constexpr const char* OP_CHECK_LEVEL_AICORE = "aicore";
constexpr const char* OP_CHECK_LEVEL_ATOMIC = "atomic";
constexpr const char* OP_CHECK_LEVEL_ALL = "all";

constexpr const char* TASK = "task";
constexpr const char* TASKS = "tasks";
constexpr const char* OUTPUT_PATH = "dump_path";
constexpr const char* RANK = "rank";
constexpr const char* STEP = "step";
constexpr const char* LEVEL = "level";
constexpr const char* SEED = "seed";
constexpr const char* IS_DETERMINISTIC = "is_deterministic";
constexpr const char* ENABLE_DATALOADER = "enable_dataloader";
constexpr const char* ACL_CONFIG = "acl_config";

constexpr const char* SCOPE = "scope";
constexpr const char* LIST = "list";

constexpr const char* DATA_MODE = "data_mode";
constexpr const char* SUMMARY_MODE = "summary_mode";
constexpr const char* FILE_FORMAT = "file_format";
constexpr const char* OVERFLOW_NUMS = "overflow_nums";
constexpr const char* CHECK_MODE = "check_mode";
constexpr const char* BACKWARD_INPUT = "backward_input";

constexpr const char* STATISTICS = "statistics";
constexpr const char* MD5 = "md5";
constexpr const char* MAX = "max";
constexpr const char* MIN = "min";
constexpr const char* MEAN = "mean";
constexpr const char* L2_NORM = "l2norm";
constexpr const char* NAN_COUNT = "nan count";
constexpr const char* NEGATIVE_INF_COUNT = "negative inf count";
constexpr const char* POSITIVE_INF_COUNT = "positive inf count";

const std::map<int32_t, std::string> FRAMEWORK_ENUM_2_NAME = {
    {static_cast<int32_t>(DebuggerFramework::FRAMEWORK_PYTORCH), FRAMEWORK_PYTORCH},
    {static_cast<int32_t>(DebuggerFramework::FRAMEWORK_MINDSPORE), FRAMEWORK_MINDSPORE},
};

const std::map<int32_t, std::string> TASK_TYPE_ENUM_2_NAME = {
    {static_cast<int32_t>(DebuggerTaskType::TASK_DUMP_TENSOR), TASK_DUMP_TENSOR},
    {static_cast<int32_t>(DebuggerTaskType::TASK_DUMP_STATISTICS), TASK_STATISTICS},
    {static_cast<int32_t>(DebuggerTaskType::TASK_OVERFLOW_CHECK), TASK_OVERFLOW_CHECK},
    {static_cast<int32_t>(DebuggerTaskType::TASK_FREE_BENCHMARK), TASK_FREE_BENCHMARK},
    {static_cast<int32_t>(DebuggerTaskType::TASK_RUN_UT), TASK_RUN_UT},
    {static_cast<int32_t>(DebuggerTaskType::TASK_GRAD_PROBE), TASK_GRAD_PROBE},
};

const std::map<int32_t, std::string> DEBUGGER_LEVEL_ENUM_2_NAME = {
    {static_cast<int32_t>(DebuggerLevel::L0), LEVEL0},
    {static_cast<int32_t>(DebuggerLevel::L1), LEVEL0},
    {static_cast<int32_t>(DebuggerLevel::L2), LEVEL2},
    {static_cast<int32_t>(DebuggerLevel::MIX), LEVEL_MIX},
};

const std::map<int32_t, std::string> DATA_DIRECTION_ENUM_2_NAME = {
    {static_cast<int32_t>(DebuggerDataDirection::DIRECTION_FORWARD), DIRECTION_FORWARD},
    {static_cast<int32_t>(DebuggerDataDirection::DIRECTION_BACKWARD), DIRECTION_BACKWARD},
    {static_cast<int32_t>(DebuggerDataDirection::DIRECTION_BOTH), DIRECTION_BOTH},
};

const std::map<int32_t, std::string> DATA_INOUT_ENUM_2_NAME = {
    {static_cast<int32_t>(DebuggerDataInOut::INOUT_INPUT), INOUT_INPUT},
    {static_cast<int32_t>(DebuggerDataInOut::INOUT_OUTPUT), INOUT_OUTPUT},
    {static_cast<int32_t>(DebuggerDataInOut::INOUT_BOTH), INOUT_BOTH},
};

const std::map<int32_t, std::string> DUMP_FILE_FORMAT_ENUM_2_NAME = {
    {static_cast<int32_t>(DebuggerDumpFileFormat::FILE_FORMAT_BIN), DUMP_FILE_FORMAT_BIN},
    {static_cast<int32_t>(DebuggerDumpFileFormat::FILE_FORMAT_NPY), DUMP_FILE_FORMAT_NPY},
};

const std::map<int32_t, std::string> OP_CHECK_LEVEL_ENUM_2_NAME = {
    {static_cast<int32_t>(DebuggerOpCheckLevel::CHECK_LEVEL_AICORE), OP_CHECK_LEVEL_AICORE},
    {static_cast<int32_t>(DebuggerOpCheckLevel::CHECK_LEVEL_ATOMIC), OP_CHECK_LEVEL_ATOMIC},
    {static_cast<int32_t>(DebuggerOpCheckLevel::CHECK_LEVEL_ALL), OP_CHECK_LEVEL_ALL},
};

const std::map<int32_t, std::string> SUMMARY_OPTION_ENUM_2_NAME = {
    {static_cast<int32_t>(DebuggerSummaryOption::MAX), MAX},
    {static_cast<int32_t>(DebuggerSummaryOption::MIN), MIN},
    {static_cast<int32_t>(DebuggerSummaryOption::MEAN), MEAN},
    {static_cast<int32_t>(DebuggerSummaryOption::NAN_CNT), NAN_COUNT},
    {static_cast<int32_t>(DebuggerSummaryOption::NEG_INF_CNT), NEGATIVE_INF_COUNT},
    {static_cast<int32_t>(DebuggerSummaryOption::POS_INF_CNT), POSITIVE_INF_COUNT},
    {static_cast<int32_t>(DebuggerSummaryOption::L2NORM), L2_NORM},

    {static_cast<int32_t>(DebuggerSummaryOption::MD5), MD5},
};

inline int32_t GetEnumIdFromName(const std::map<int32_t, std::string>& enum2name, const std::string& name)
{
    for (auto iter = enum2name.begin(); iter != enum2name.end(); iter++) {
        if (iter->second == name) {
            return iter->first;
        }
    }
    return DEBUGGER_INVALID_ENUM;
}

inline std::string GetNameFromEnumId(const std::map<int32_t, std::string>& enum2name, int32_t id)
{
    auto iter = enum2name.find(id);
    if (iter == enum2name.end()) {
        return "UNKNOWN";
    }
    return iter->second;
}

}