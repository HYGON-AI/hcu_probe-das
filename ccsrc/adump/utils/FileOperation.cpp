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

#include <unordered_map>
#include <sstream>
#include <climits>

#include "FileUtils.h"
#include "DataUtils.h"
#include "FileOperation.h"

namespace MindStudioDebugger {
namespace FileOperation {

using namespace  MindStudioDebugger;
using DataType = DataUtils::DataType;
using NpyVersion = std::pair<char, char>;

struct NpyDtypeDescr {
    char byteorder;
    char type;
    size_t length;

    std::string Str() const
    {
        std::ostringstream buffer;
        buffer << "\'" << byteorder << type << length << "\'";
        return buffer.str();
    }
};

// npy file header start information
constexpr char NPY_MAGIC_PREFIX[] = "\x93NUMPY";
constexpr size_t NPY_MAGIC_LEN = sizeof(NPY_MAGIC_PREFIX) - 1;
constexpr size_t NPY_ARRAY_ALIGN = 64;
static const std::unordered_map<DataType, NpyDtypeDescr> npyTypeDescMap = {
    {DataType::DT_BOOL, NpyDtypeDescr{'|', 'b', 1}},      {DataType::DT_INT8, NpyDtypeDescr{'|', 'i', 1}},
    {DataType::DT_INT16, NpyDtypeDescr{'<', 'i', 2}},     {DataType::DT_INT32, NpyDtypeDescr{'<', 'i', 4}},
    {DataType::DT_INT64, NpyDtypeDescr{'<', 'i', 8}},     {DataType::DT_UINT8, NpyDtypeDescr{'|', 'u', 1}},
    {DataType::DT_UINT16, NpyDtypeDescr{'<', 'u', 2}},    {DataType::DT_UINT32, NpyDtypeDescr{'<', 'u', 4}},
    {DataType::DT_UINT64, NpyDtypeDescr{'<', 'u', 8}},    {DataType::DT_FLOAT16, NpyDtypeDescr{'<', 'f', 2}},
    {DataType::DT_FLOAT, NpyDtypeDescr{'<', 'f', 4}},     {DataType::DT_DOUBLE, NpyDtypeDescr{'<', 'f', 8}},
    {DataType::DT_COMPLEX128, NpyDtypeDescr{'<', 'c', 16}}, {DataType::DT_COMPLEX64, NpyDtypeDescr{'<', 'c', 8}},
};

DebuggerErrno DumpJson(const std::string &path, const nlohmann::json& content)
{
    DebuggerErrno ret;
    std::ofstream ofs;

    ret = FileUtils::OpenFile(path, ofs);
    if (ret != DebuggerErrno::OK) {
        return ret;
    }

    try {
        ofs << content.dump();
    } catch (std::exception &e) {
        ret = DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE;
    }

    if (ofs.fail()) {
        ret = DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE;
    }

    ofs.close();
    return ret;
}

inline static std::string NpyTransShapeToStr(const DataUtils::TensorShape &shape)
{
    std::ostringstream buffer;
    buffer << "(";
    for (const auto i : shape) {
        buffer << std::to_string(i) << ",";
    }
    buffer << ")";
    return buffer.str();
}

inline static std::vector<char> NpyLen2Bytes(size_t length, size_t lengthLen)
{
    std::vector<char> buff;
    lengthLen = std::min(lengthLen, static_cast<size_t>(sizeof(length)));
    for (size_t i = 0; i < lengthLen; i++) {
        buff.emplace_back(length & 0xff);
        length >>= CHAR_BIT;
    }
    return buff;
}

static std::string GenerateNpyHeader(const DataUtils::TensorShape &shape,
    DataUtils::DataType dt, bool fortranOrder = false)
{
    auto typeDesc = npyTypeDescMap.find(dt);
    if (typeDesc == npyTypeDescMap.end()) {
        return std::string();
    }

    std::ostringstream buffer;
    std::string fortranOrderStr = fortranOrder ? "True" : "False" ;

    buffer << "{";
    buffer << "'descr': " << typeDesc->second.Str() << ", ";
    buffer << "'fortran_order': " << fortranOrderStr << ", ";
    buffer << "'shape': " << NpyTransShapeToStr(shape) << ", ";
    buffer << "}";

    std::string headerStr = buffer.str();
    NpyVersion version{1, 0};
    const size_t headerLen = headerStr.length();
    constexpr const size_t versionLen = 2;
    constexpr const size_t maxLen = 65535;
    constexpr const size_t lengthLenV1 = 2;
    constexpr const size_t lengthLenV2 = 4;
    size_t lengthLen = lengthLenV1;

    size_t totalLen = NPY_MAGIC_LEN + versionLen + lengthLen + headerLen + 1;
    if (totalLen > maxLen) {
        version = {2, 0};
        lengthLen = lengthLenV2;
        totalLen = NPY_MAGIC_LEN + versionLen + lengthLen + headerLen + 1;
    }

    const size_t padLen = NPY_ARRAY_ALIGN - totalLen % NPY_ARRAY_ALIGN;
    const size_t paddingHeaderLen = headerLen + padLen + 1;
    const std::string padding(padLen, ' ');
    std::vector<char> lengthBytes = NpyLen2Bytes(paddingHeaderLen, lengthLen);
    std::ostringstream out;
    out.write(NPY_MAGIC_PREFIX, DataUtils::SizeToS64(NPY_MAGIC_LEN));
    out.put(version.first);
    out.put(version.second);
    out.write(lengthBytes.data(), DataUtils::SizeToS64(lengthBytes.size()));
    out << headerStr << padding << "\n";
    return out.str();
}

bool IsDtypeSupportByNpy(DataUtils::DataType dt)
{
    return npyTypeDescMap.find(dt) != npyTypeDescMap.end();
}

DebuggerErrno DumpNpy(const std::string &path, const uint8_t* data, size_t len, DataUtils::DataType dt,
                      const DataUtils::TensorShape& shape)
{
    DebuggerErrno ret;
    std::string header = GenerateNpyHeader(shape, dt);
    if (header.empty()) {
        return DebuggerErrno::ERROR_INVALID_FORMAT;
    }

    std::ofstream fd;
    ret = FileUtils::OpenFile(path, fd, std::ios::out | std::ios::binary);
    if (ret != DebuggerErrno::OK) {
        return ret;
    }

    fd << header;
    fd.write(reinterpret_cast<const char*>(data), len);
    if (fd.fail()) {
        ret = DebuggerErrno::ERROR_OPERATION_FAILED;
    }
    fd.close();

    return ret;
}

}
}