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
