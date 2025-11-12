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
#include <vector>

#include "include/ErrorCode.h"
#include "third_party/proto/AclDumpMsg.pb.h"
#include "utils/DataUtils.h"

namespace MindStudioDebugger {

using AclShape = DataUtils::TensorShape;
using AclDtype = DataUtils::DataType;
using AclFormat = DataUtils::TensorFormat;

constexpr uint8_t DIM_1 = 1;
constexpr uint8_t DIM_2 = 2;
constexpr uint8_t DIM_3 = 3;
constexpr uint8_t DIM_4 = 4;
constexpr uint8_t DIM_5 = 5;
constexpr uint8_t DIM_6 = 6;

struct AclTensorInfo {
    std::string dumpPath;
    const uint8_t* aclData;
    AclDtype dtype;
    AclDtype oriDtype;
    AclFormat deviceFmt;
    AclFormat hostFmt;
    AclShape deviceShape;
    AclShape hostShape;
    size_t dataSize;
    int32_t subFormat;
    std::string inout;
    uint32_t slot;
    bool dumpOriginData;
    std::vector<uint8_t> transBuf;

    std::string ToString() const
    {
        return "AclTensor(path=" + dumpPath + ",dtype=" + DataUtils::GetDTypeString(dtype) + ",inout=" + inout + ")";
    }
};

inline std::string operator+(const std::string& s, const AclTensorInfo& tensor)
{
    return s + tensor.ToString();
}

inline std::string operator+(const AclTensorInfo& tensor, const std::string& s)
{
    return tensor.ToString() + s;
}

namespace AclTensor {
size_t SizeOfTensor(const AclTensorInfo& tensor, bool host = true);
template <typename T>
AclTensorInfo ParseAttrsFromDumpData(const std::string &dumpPath, const uint8_t* data, const T& tensor,
                                     const std::string& io, uint32_t slot);
DebuggerErrno TransFormatD2H(AclTensorInfo& tensor);
DebuggerErrno TransDtype(AclTensorInfo& tensor, AclDtype to);
bool IsDtypeSupportTrans(AclDtype dtype);

}
}
