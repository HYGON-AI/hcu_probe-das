/*
 * -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#pragma once

#include <nlohmann/json.hpp>

#include "include/ErrorCode.h"
#include "DataUtils.h"

namespace MindStudioDebugger {

constexpr const char* JSON_SUFFIX = "json";
constexpr const char* NPY_SUFFIX = "npy";
constexpr const char* CSV_SUFFIX = "csv";

namespace FileOperation {

DebuggerErrno DumpJson(const std::string &path, const nlohmann::json& content);
bool IsDtypeSupportByNpy(DataUtils::DataType dt);
DebuggerErrno DumpNpy(const std::string &path, const uint8_t* data, size_t len, DataUtils::DataType dt,
                      const DataUtils::TensorShape& shape);

}
}