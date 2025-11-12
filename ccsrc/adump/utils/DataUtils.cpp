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

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "DataUtils.h"

namespace MindStudioDebugger {
namespace  DataUtils {

int64_t SizeToS64(size_t v)
{
    if (v > static_cast<size_t>(INT64_MAX)) {
        throw std::runtime_error("Value " + std::to_string(v) + "exceeds the maximum value of int64.");
    }
    return static_cast<int64_t>(v);
}

std::string U64ToHexString(uint64_t v)
{
    std::stringstream ss;
    ss << "0x" << std::hex << std::uppercase << v;
    return std::move(ss.str());
}

BFloat16::BFloat16(float f32)
{
    if (std::isnan(f32)) {
        value_ = BFloat16::NAN_VALUE;
    } else {
        constexpr uint8_t offsetSize = 16;
        union {
            uint32_t u32Value;
            float f32Value;
        };
        f32Value = f32;
        uint32_t rounding_bias = ((u32Value >> offsetSize) & 1) + UINT32_C(0x7FFF);
        value_ = static_cast<uint16_t>((u32Value + rounding_bias) >> offsetSize);
    }
}

BFloat16::operator float() const
{
    /* 为了兼容性，不要用c++20的bit_cast */
    constexpr uint8_t offsetSize = 16;
    union {
        float f32;
        uint32_t ui32;
    };
    ui32 = static_cast<uint32_t>(value_);
    ui32 <<= offsetSize; // 将ui32左移16位
    return f32;
}

constexpr std::pair<DataType, size_t> TYPE_SIZE_ARRAY[] = {
    {DataType::DT_BOOL, 1},
    {DataType::DT_INT8, 1},
    {DataType::DT_UINT8, 1},
    {DataType::DT_INT16, 2},
    {DataType::DT_UINT16, 2},
    {DataType::DT_FLOAT16, 2},
    {DataType::DT_BF16, 2},
    {DataType::DT_INT32, 4},
    {DataType::DT_UINT32, 4},
    {DataType::DT_FLOAT, 4},
    {DataType::DT_INT64, 8},
    {DataType::DT_UINT64, 8},
    {DataType::DT_DOUBLE, 8},
    {DataType::DT_COMPLEX64, 8},
    {DataType::DT_COMPLEX128, 16},
};

size_t SizeOfDType(DataType type)
{
    for (const auto& pair : TYPE_SIZE_ARRAY) {
        if (pair.first == type) {
            return pair.second;
        }
    }
    return 0;
}

constexpr auto OP_DTYPE_UNKNOWN = "UNKNOWN";
const std::pair<DataType, std::string_view> DTYPE_TO_STRING_ARRAY[] = {
    {DataType::DT_UNDEFINED, "UNDEFINED"},
    {DataType::DT_FLOAT, "FLOAT"},
    {DataType::DT_FLOAT16, "FLOAT16"},
    {DataType::DT_INT8, "INT8"},
    {DataType::DT_UINT8, "UINT8"},
    {DataType::DT_INT16, "INT16"},
    {DataType::DT_UINT16, "UINT16"},
    {DataType::DT_INT32, "INT32"},
    {DataType::DT_INT64, "INT64"},
    {DataType::DT_UINT32, "UINT32"},
    {DataType::DT_UINT64, "UINT64"},
    {DataType::DT_BOOL, "BOOL"},
    {DataType::DT_DOUBLE, "DOUBLE"},
    {DataType::DT_STRING, "STRING"},
    {DataType::DT_DUAL_SUB_INT8, "DUAL_SUB_INT8"},
    {DataType::DT_DUAL_SUB_UINT8, "DUAL_SUB_UINT8"},
    {DataType::DT_COMPLEX64, "COMPLEX64"},
    {DataType::DT_COMPLEX128, "COMPLEX128"},
    {DataType::DT_QINT8, "QINT8"},
    {DataType::DT_QINT16, "QINT16"},
    {DataType::DT_QINT32, "QINT32"},
    {DataType::DT_QUINT8, "QUINT8"},
    {DataType::DT_QUINT16, "QUINT16"},
    {DataType::DT_RESOURCE, "RESOURCE"},
    {DataType::DT_STRING_REF, "STRING_REF"},
    {DataType::DT_DUAL, "DUAL"},
    {DataType::DT_VARIANT, "VARIANT"},
    {DataType::DT_BF16, "BF16"},
    {DataType::DT_INT4, "INT4"},
    {DataType::DT_UINT1, "UINT1"},
    {DataType::DT_INT2, "INT2"},
    {DataType::DT_UINT2, "UINT2"},
};

std::string GetDTypeString(DataType dtype)
{
    for (const auto& pair : DTYPE_TO_STRING_ARRAY) {
        if (pair.first == dtype) {
            return std::string(pair.second);
        }
    }
    return OP_DTYPE_UNKNOWN;
}

constexpr auto OP_FORMAT_UNKNOWN = "UNKNOWN";
const std::pair<TensorFormat, std::string_view> FORMAT_TO_STRING_ARRAY[] = {
    {TensorFormat::FORMAT_NCHW, "NCHW"},
    {TensorFormat::FORMAT_NHWC, "NHWC"},
    {TensorFormat::FORMAT_ND, "ND"},
    {TensorFormat::FORMAT_NC1HWC0, "NC1HWC0"},
    {TensorFormat::FORMAT_FRACTAL_Z, "FRACTAL_Z"},
    {TensorFormat::FORMAT_NC1C0HWPAD, "NC1C0HWPAD"},
    {TensorFormat::FORMAT_NHWC1C0, "NHWC1C0"},
    {TensorFormat::FORMAT_FSR_NCHW, "FSR_NCHW"},
    {TensorFormat::FORMAT_FRACTAL_DECONV, "FRACTAL_DECONV"},
    {TensorFormat::FORMAT_C1HWNC0, "C1HWNC0"},
    {TensorFormat::FORMAT_FRACTAL_DECONV_TRANSPOSE, "FRACTAL_DECONV_TRANSPOSE"},
    {TensorFormat::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS, "FRACTAL_DECONV_SP_STRIDE_TRANS"},
    {TensorFormat::FORMAT_NC1HWC0_C04, "NC1HWC0_C04"},
    {TensorFormat::FORMAT_FRACTAL_Z_C04, "FRACTAL_Z_C04"},
    {TensorFormat::FORMAT_CHWN, "CHWN"},
    {TensorFormat::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS, "FRACTAL_DECONV_SP_STRIDE8_TRANS"},
    {TensorFormat::FORMAT_HWCN, "HWCN"},
    {TensorFormat::FORMAT_NC1KHKWHWC0, "NC1KHKWHWC0"},
    {TensorFormat::FORMAT_BN_WEIGHT, "BN_WEIGHT"},
    {TensorFormat::FORMAT_FILTER_HWCK, "FILTER_HWCK"},
    {TensorFormat::FORMAT_HASHTABLE_LOOKUP_LOOKUPS, "HASHTABLE_LOOKUP_LOOKUPS"},
    {TensorFormat::FORMAT_HASHTABLE_LOOKUP_KEYS, "HASHTABLE_LOOKUP_KEYS"},
    {TensorFormat::FORMAT_HASHTABLE_LOOKUP_VALUE, "HASHTABLE_LOOKUP_VALUE"},
    {TensorFormat::FORMAT_HASHTABLE_LOOKUP_OUTPUT, "HASHTABLE_LOOKUP_OUTPUT"},
    {TensorFormat::FORMAT_HASHTABLE_LOOKUP_HITS, "HASHTABLE_LOOKUP_HITS"},
    {TensorFormat::FORMAT_C1HWNCOC0, "C1HWNCoC0"},
    {TensorFormat::FORMAT_MD, "MD"},
    {TensorFormat::FORMAT_NDHWC, "NDHWC"},
    {TensorFormat::FORMAT_FRACTAL_ZZ, "FRACTAL_ZZ"},
    {TensorFormat::FORMAT_FRACTAL_NZ, "FRACTAL_NZ"},
    {TensorFormat::FORMAT_NCDHW, "NCDHW"},
    {TensorFormat::FORMAT_DHWCN, "DHWCN"},
    {TensorFormat::FORMAT_NDC1HWC0, "NDC1HWC0"},
    {TensorFormat::FORMAT_FRACTAL_Z_3D, "FRACTAL_Z_3D"},
    {TensorFormat::FORMAT_CN, "CN"},
    {TensorFormat::FORMAT_NC, "NC"},
    {TensorFormat::FORMAT_DHWNC, "DHWNC"},
    {TensorFormat::FORMAT_FRACTAL_Z_3D_TRANSPOSE, "FRACTAL_Z_3D_TRANSPOSE"},
    {TensorFormat::FORMAT_FRACTAL_ZN_LSTM, "FRACTAL_ZN_LSTM"},
    {TensorFormat::FORMAT_FRACTAL_Z_G, "FRACTAL_Z_G"},
    {TensorFormat::FORMAT_RESERVED, "RESERVED"},
    {TensorFormat::FORMAT_ALL, "ALL"},
    {TensorFormat::FORMAT_NULL, "NULL"},
    {TensorFormat::FORMAT_ND_RNN_BIAS, "ND_RNN_BIAS"},
    {TensorFormat::FORMAT_FRACTAL_ZN_RNN, "FRACTAL_ZN_RNN"},
    {TensorFormat::FORMAT_YUV, "YUV"},
    {TensorFormat::FORMAT_YUV_A, "YUV_A"},
    {TensorFormat::FORMAT_NCL, "NCL"},
    {TensorFormat::FORMAT_FRACTAL_Z_WINO, "FRACTAL_Z_WINO"},
    {TensorFormat::FORMAT_C1HWC0, "C1HWC0"},
};

std::string GetFormatString(TensorFormat fmt)
{
    for (const auto& pair : FORMAT_TO_STRING_ARRAY) {
        if (pair.first == fmt) {
            return std::string(pair.second);
        }
    }
    return OP_FORMAT_UNKNOWN;
}

std::string GetShapeString(const TensorShape& shape)
{
    std::ostringstream buffer;
    buffer << "(";
    for (size_t i = 0; i < shape.size(); i++) {
        buffer << (i > 0 ? "," : "") << shape[i];
    }
    buffer << ")";
    return buffer.str();
}

}
}