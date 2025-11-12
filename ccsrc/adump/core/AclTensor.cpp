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

#include <unordered_set>
#include <unordered_map>
#include <map>
#include <set>
#include <stdexcept>
#include <cstring>
#include <algorithm>

#include "utils/DataUtils.h"
#include "utils/MathUtils.h"
#include "base/ErrorInfosManager.h"
#include "AclTensor.h"

namespace MindStudioDebugger {
namespace AclDumpMsg = toolkit::dumpdata;
namespace AclTensor {

using namespace MathUtils;

constexpr int64_t CUBE_SIZE = 16;
constexpr int64_t CUBE_16 = CUBE_SIZE;
constexpr int64_t CUBE_32 = 32;
constexpr int64_t CUBE_64 = 64;
constexpr int64_t CUBE_SIZE_C04 = 4;

constexpr size_t HW_H = 1;
constexpr size_t HW_W = 2;
constexpr size_t FNZ_W1 = 4;
constexpr size_t FNZ_H1 = 3;
constexpr size_t FNZ_H0 = 2;
constexpr size_t FNZ_W0 = 1;
constexpr size_t FZ_N0 = 1;
constexpr size_t FZ_NI = 2;
constexpr size_t FZ_C0 = 3;

using TensorTransFunc = DebuggerErrno (*)(AclTensorInfo &);

static DebuggerErrno FRAC_Z_TO_NCHW(AclTensorInfo& tensor);
static DebuggerErrno FRAC_NZ_TO_NCHW(AclTensorInfo& tensor);
static DebuggerErrno NC1HWC0_TO_NCHW(AclTensorInfo& tensor);
static DebuggerErrno NDC1HWC0_TO_NCDHW(AclTensorInfo& tensor);
static DebuggerErrno C1HWNCoC0_TO_NCHW(AclTensorInfo& tensor);
static DebuggerErrno NC1HWC0_C04_TO_NCHW(AclTensorInfo& tensor);
static DebuggerErrno FRAC_Z3D_TO_NCDHW(AclTensorInfo& tensor);

const static std::unordered_set<AclDtype> kSupportedDtypes = {
    AclDtype::DT_UNDEFINED,
    AclDtype::DT_FLOAT,
    AclDtype::DT_FLOAT16,
    AclDtype::DT_INT8,
    AclDtype::DT_UINT8,
    AclDtype::DT_INT16,
    AclDtype::DT_UINT16,
    AclDtype::DT_INT32,
    AclDtype::DT_INT64,
    AclDtype::DT_UINT32,
    AclDtype::DT_UINT64,
    AclDtype::DT_BOOL,
    AclDtype::DT_DOUBLE,
    AclDtype::DT_BF16,
    AclDtype::DT_COMPLEX64,
    AclDtype::DT_COMPLEX128,
};

const static std::unordered_set<AclFormat> kSupportedFormat = {
    AclFormat::FORMAT_NCHW,
    AclFormat::FORMAT_NHWC,
    AclFormat::FORMAT_ND,
    AclFormat::FORMAT_NC1HWC0,
    AclFormat::FORMAT_FRACTAL_Z,
    AclFormat::FORMAT_NC1HWC0_C04,
    AclFormat::FORMAT_FRACTAL_Z_C04,
    AclFormat::FORMAT_NC1KHKWHWC0,
    AclFormat::FORMAT_HWCN,
    AclFormat::FORMAT_NDHWC,
    AclFormat::FORMAT_NCDHW,
    AclFormat::FORMAT_DHWCN,
    AclFormat::FORMAT_DHWNC,
    AclFormat::FORMAT_NDC1HWC0,
    AclFormat::FORMAT_FRACTAL_Z_3D,
    AclFormat::FORMAT_C1HWNCOC0,
    AclFormat::FORMAT_FRACTAL_NZ,
    AclFormat::FORMAT_FRACTAL_ZN_LSTM,
    AclFormat::FORMAT_NCL,
};

const static std::map<std::pair<AclFormat, AclFormat>, TensorTransFunc> formatTransFuncMap = {
    {{AclFormat::FORMAT_HWCN, AclFormat::FORMAT_NCHW}, nullptr},
    {{AclFormat::FORMAT_NHWC, AclFormat::FORMAT_NCHW}, nullptr},
    {{AclFormat::FORMAT_FRACTAL_Z, AclFormat::FORMAT_NCHW}, FRAC_Z_TO_NCHW},
    {{AclFormat::FORMAT_FRACTAL_NZ, AclFormat::FORMAT_NCHW}, FRAC_NZ_TO_NCHW},
    {{AclFormat::FORMAT_NC1HWC0, AclFormat::FORMAT_NCHW}, NC1HWC0_TO_NCHW},
    {{AclFormat::FORMAT_NDC1HWC0, AclFormat::FORMAT_NCHW}, NDC1HWC0_TO_NCDHW},
    {{AclFormat::FORMAT_C1HWNCOC0, AclFormat::FORMAT_NCHW}, C1HWNCoC0_TO_NCHW},
    {{AclFormat::FORMAT_NC1HWC0_C04, AclFormat::FORMAT_NCHW}, NC1HWC0_C04_TO_NCHW},
    {{AclFormat::FORMAT_FRACTAL_Z_3D, AclFormat::FORMAT_NCHW}, FRAC_Z3D_TO_NCDHW},
};

const static std::unordered_map<AclDumpMsg::OutputDataType, AclDtype> dtypeTransMap = {
    {AclDumpMsg::OutputDataType::DT_UNDEFINED, AclDtype::DT_UNDEFINED},
    {AclDumpMsg::OutputDataType::DT_FLOAT, AclDtype::DT_FLOAT},
    {AclDumpMsg::OutputDataType::DT_FLOAT16, AclDtype::DT_FLOAT16},
    {AclDumpMsg::OutputDataType::DT_INT8, AclDtype::DT_INT8},
    {AclDumpMsg::OutputDataType::DT_UINT8, AclDtype::DT_UINT8},
    {AclDumpMsg::OutputDataType::DT_INT16, AclDtype::DT_INT16},
    {AclDumpMsg::OutputDataType::DT_UINT16, AclDtype::DT_UINT16},
    {AclDumpMsg::OutputDataType::DT_INT32, AclDtype::DT_INT32},
    {AclDumpMsg::OutputDataType::DT_INT64, AclDtype::DT_INT64},
    {AclDumpMsg::OutputDataType::DT_UINT32, AclDtype::DT_UINT32},
    {AclDumpMsg::OutputDataType::DT_UINT64, AclDtype::DT_UINT64},
    {AclDumpMsg::OutputDataType::DT_BOOL, AclDtype::DT_BOOL},
    {AclDumpMsg::OutputDataType::DT_DOUBLE, AclDtype::DT_DOUBLE},
    {AclDumpMsg::OutputDataType::DT_STRING, AclDtype::DT_STRING},
    {AclDumpMsg::OutputDataType::DT_DUAL_SUB_INT8, AclDtype::DT_DUAL_SUB_INT8},
    {AclDumpMsg::OutputDataType::DT_DUAL_SUB_UINT8, AclDtype::DT_DUAL_SUB_UINT8},
    {AclDumpMsg::OutputDataType::DT_COMPLEX64, AclDtype::DT_COMPLEX64},
    {AclDumpMsg::OutputDataType::DT_COMPLEX128, AclDtype::DT_COMPLEX128},
    {AclDumpMsg::OutputDataType::DT_QINT8, AclDtype::DT_QINT8},
    {AclDumpMsg::OutputDataType::DT_QINT16, AclDtype::DT_QINT16},
    {AclDumpMsg::OutputDataType::DT_QINT32, AclDtype::DT_QINT32},
    {AclDumpMsg::OutputDataType::DT_QUINT8, AclDtype::DT_QUINT8},
    {AclDumpMsg::OutputDataType::DT_QUINT16, AclDtype::DT_QUINT16},
    {AclDumpMsg::OutputDataType::DT_RESOURCE, AclDtype::DT_RESOURCE},
    {AclDumpMsg::OutputDataType::DT_STRING_REF, AclDtype::DT_STRING_REF},
    {AclDumpMsg::OutputDataType::DT_DUAL, AclDtype::DT_DUAL},
    {AclDumpMsg::OutputDataType::DT_VARIANT, AclDtype::DT_VARIANT},
    {AclDumpMsg::OutputDataType::DT_BF16, AclDtype::DT_BF16},
    {AclDumpMsg::OutputDataType::DT_INT4, AclDtype::DT_INT4},
    {AclDumpMsg::OutputDataType::DT_UINT1, AclDtype::DT_UINT1},
    {AclDumpMsg::OutputDataType::DT_INT2, AclDtype::DT_INT2},
    {AclDumpMsg::OutputDataType::DT_UINT2, AclDtype::DT_UINT2},
};

const static std::unordered_map<AclDumpMsg::OutputFormat, AclFormat> formatTransMap = {
    {AclDumpMsg::OutputFormat::FORMAT_NCHW, AclFormat::FORMAT_NCHW},
    {AclDumpMsg::OutputFormat::FORMAT_NHWC, AclFormat::FORMAT_NHWC},
    {AclDumpMsg::OutputFormat::FORMAT_ND, AclFormat::FORMAT_ND},
    {AclDumpMsg::OutputFormat::FORMAT_NC1HWC0, AclFormat::FORMAT_NC1HWC0},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_Z, AclFormat::FORMAT_FRACTAL_Z},
    {AclDumpMsg::OutputFormat::FORMAT_NC1C0HWPAD, AclFormat::FORMAT_NC1C0HWPAD},
    {AclDumpMsg::OutputFormat::FORMAT_NHWC1C0, AclFormat::FORMAT_NHWC1C0},
    {AclDumpMsg::OutputFormat::FORMAT_FSR_NCHW, AclFormat::FORMAT_FSR_NCHW},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_DECONV, AclFormat::FORMAT_FRACTAL_DECONV},
    {AclDumpMsg::OutputFormat::FORMAT_C1HWNC0, AclFormat::FORMAT_C1HWNC0},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_DECONV_TRANSPOSE, AclFormat::FORMAT_FRACTAL_DECONV_TRANSPOSE},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS, AclFormat::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS},
    {AclDumpMsg::OutputFormat::FORMAT_NC1HWC0_C04, AclFormat::FORMAT_NC1HWC0_C04},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_Z_C04, AclFormat::FORMAT_FRACTAL_Z_C04},
    {AclDumpMsg::OutputFormat::FORMAT_CHWN, AclFormat::FORMAT_CHWN},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS,
        AclFormat::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS},
    {AclDumpMsg::OutputFormat::FORMAT_HWCN, AclFormat::FORMAT_HWCN},
    {AclDumpMsg::OutputFormat::FORMAT_NC1KHKWHWC0, AclFormat::FORMAT_NC1KHKWHWC0},
    {AclDumpMsg::OutputFormat::FORMAT_BN_WEIGHT, AclFormat::FORMAT_BN_WEIGHT},
    {AclDumpMsg::OutputFormat::FORMAT_FILTER_HWCK, AclFormat::FORMAT_FILTER_HWCK},
    {AclDumpMsg::OutputFormat::FORMAT_HASHTABLE_LOOKUP_LOOKUPS, AclFormat::FORMAT_HASHTABLE_LOOKUP_LOOKUPS},
    {AclDumpMsg::OutputFormat::FORMAT_HASHTABLE_LOOKUP_KEYS, AclFormat::FORMAT_HASHTABLE_LOOKUP_KEYS},
    {AclDumpMsg::OutputFormat::FORMAT_HASHTABLE_LOOKUP_VALUE, AclFormat::FORMAT_HASHTABLE_LOOKUP_VALUE},
    {AclDumpMsg::OutputFormat::FORMAT_HASHTABLE_LOOKUP_OUTPUT, AclFormat::FORMAT_HASHTABLE_LOOKUP_OUTPUT},
    {AclDumpMsg::OutputFormat::FORMAT_HASHTABLE_LOOKUP_HITS, AclFormat::FORMAT_HASHTABLE_LOOKUP_HITS},
    {AclDumpMsg::OutputFormat::FORMAT_C1HWNCoC0, AclFormat::FORMAT_C1HWNCOC0},
    {AclDumpMsg::OutputFormat::FORMAT_MD, AclFormat::FORMAT_MD},
    {AclDumpMsg::OutputFormat::FORMAT_NDHWC, AclFormat::FORMAT_NDHWC},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_ZZ, AclFormat::FORMAT_FRACTAL_ZZ},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_NZ, AclFormat::FORMAT_FRACTAL_NZ},
    {AclDumpMsg::OutputFormat::FORMAT_NCDHW, AclFormat::FORMAT_NCDHW},
    {AclDumpMsg::OutputFormat::FORMAT_DHWCN, AclFormat::FORMAT_DHWCN},
    {AclDumpMsg::OutputFormat::FORMAT_NDC1HWC0, AclFormat::FORMAT_NDC1HWC0},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_Z_3D, AclFormat::FORMAT_FRACTAL_Z_3D},
    {AclDumpMsg::OutputFormat::FORMAT_CN, AclFormat::FORMAT_CN},
    {AclDumpMsg::OutputFormat::FORMAT_NC, AclFormat::FORMAT_NC},
    {AclDumpMsg::OutputFormat::FORMAT_DHWNC, AclFormat::FORMAT_DHWNC},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_Z_3D_TRANSPOSE, AclFormat::FORMAT_FRACTAL_Z_3D_TRANSPOSE},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_ZN_LSTM, AclFormat::FORMAT_FRACTAL_ZN_LSTM},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_Z_G, AclFormat::FORMAT_FRACTAL_Z_G},
    {AclDumpMsg::OutputFormat::FORMAT_RESERVED, AclFormat::FORMAT_RESERVED},
    {AclDumpMsg::OutputFormat::FORMAT_ALL, AclFormat::FORMAT_ALL},
    {AclDumpMsg::OutputFormat::FORMAT_NULL, AclFormat::FORMAT_NULL},
    {AclDumpMsg::OutputFormat::FORMAT_ND_RNN_BIAS, AclFormat::FORMAT_ND_RNN_BIAS},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_ZN_RNN, AclFormat::FORMAT_FRACTAL_ZN_RNN},
    {AclDumpMsg::OutputFormat::FORMAT_YUV, AclFormat::FORMAT_YUV},
    {AclDumpMsg::OutputFormat::FORMAT_YUV_A, AclFormat::FORMAT_YUV_A},
    {AclDumpMsg::OutputFormat::FORMAT_NCL, AclFormat::FORMAT_NCL},
    {AclDumpMsg::OutputFormat::FORMAT_FRACTAL_Z_WINO, AclFormat::FORMAT_FRACTAL_Z_WINO},
    {AclDumpMsg::OutputFormat::FORMAT_C1HWC0, AclFormat::FORMAT_C1HWC0},
};

enum Axis4D : int { AXIS_N = 0, AXIS_C, AXIS_H, AXIS_W, NCHW_DIMS };
enum Axis5D : int {
    N_NCDHW,
    C_NCDHW,
    D_NCDHW,
    H_NCDHW,
    W_NCDHW,
    NCDHW,
    N_NDC1HWC0,
    D_NDC1HWC0,
    C1_NDC1HWC0,
    H_NDC1HWC0,
    W_NDC1HWC0,
    C0_NDC1HWC0
};

static inline AclDtype transAclDtype2MS(AclDumpMsg::OutputDataType dt)
{
    auto it = dtypeTransMap.find(dt);
    if (it != dtypeTransMap.end()) {
        return it->second;
    }
    return AclDtype::DT_MAX;
}

static inline AclFormat transAclFormat2MS(AclDumpMsg::OutputFormat fmt)
{
    auto it = formatTransMap.find(fmt);
    if (it != formatTransMap.end()) {
        return it->second;
    }
    return AclFormat::FORMAT_MAX;
}

static size_t EleNumOfTensor(const AclTensorInfo& tensor, bool host = true)
{
    size_t num = 1;
    const AclShape& shape = host ? tensor.hostShape : tensor.deviceShape;
    for (auto dim : shape) {
        if (dim <= 0) {
            /* For dynamic shape which has negative dimensions, data size should be zero. */
            return 0;
        }

        if (SIZE_MAX / static_cast<unsigned long>(dim) < static_cast<unsigned long>(num)) {
            throw std::out_of_range(tensor + ": Count of element over size_t.");
        }
        num *= static_cast<size_t>(dim);
    }
    return num;
}

static inline size_t SizeOfAclDType(const AclTensorInfo& tensor)
{
    return DataUtils::SizeOfDType(tensor.dtype);
}

static inline size_t SizeOfAclDType(const AclDtype& dtype)
{
    return DataUtils::SizeOfDType(dtype);
}

size_t SizeOfTensor(const AclTensorInfo& tensor, bool host)
{
    size_t num = EleNumOfTensor(tensor, host);
    size_t eleSize = SizeOfAclDType(tensor);
    if (num != 0 && SIZE_MAX / num < eleSize) {
        throw std::runtime_error(tensor + ": Size over size_t.");
    }
    return num * eleSize;
}

static inline int64_t GetCubeSizeByType(const AclDtype& dtype)
{
    if (dtype == AclDtype::DT_UINT8 || dtype == AclDtype::DT_INT8) {
        return CUBE_32;
    }

    if (dtype == AclDtype::DT_INT4) {
        return CUBE_64;
    }

    return CUBE_16;
}

static inline void AssertDim(const AclShape& shape, size_t dim)
{
    if (shape.size() != dim) {
        throw std::runtime_error("Dimension of tensor is expected to be " + std::to_string(dim)  +
                                 ", but actually " + std::to_string(shape.size()) +".");
    }
}

static inline void AssertConsis(const AclTensorInfo& tensor)
{
    size_t tensorSize = EleNumOfTensor(tensor, false) * SizeOfAclDType(tensor);
    // Processing dtype whose size < 1
    // The ele num of quantization type(qint4*2) in MindSpore must be even.
    size_t int4_size_factor = 2;
    if (tensor.dtype == AclDtype::DT_INT4) {
        tensorSize = EleNumOfTensor(tensor, false) / int4_size_factor;
    }
    if (tensorSize != tensor.dataSize) {
        throw std::runtime_error(tensor + ": The internal data of Tensor is inconsistent.");
    }
}

template <typename T>
AclTensorInfo ParseAttrsFromDumpData(const std::string& dumpPath, const uint8_t* data, const T& tensor,
                                     const std::string& io, uint32_t slot)
{
    AclDumpMsg::OutputDataType oriDtype = tensor.data_type();
    AclDtype dtype = transAclDtype2MS(oriDtype);
    bool dumpOriginData = false;
    size_t dataSize = static_cast<size_t>(tensor.size());
    if (dtype == AclDtype::DT_MAX || kSupportedDtypes.find(dtype) == kSupportedDtypes.end()) {
        dumpOriginData = true;
    }

    AclDumpMsg::OutputFormat oriDeviceFmt = tensor.format();
    AclFormat dFmt = transAclFormat2MS(oriDeviceFmt);
    if (dFmt == AclFormat::FORMAT_MAX || kSupportedFormat.find(dFmt) == kSupportedFormat.end()) {
        dumpOriginData = true;
    }

    AclShape dShape;
    std::transform(tensor.shape().dim().begin(), tensor.shape().dim().end(), std::back_inserter(dShape),
                   DataUtils::SizeToS64);
    AclShape hShape;
    for (auto d : tensor.original_shape().dim()) {
        if (d > INT64_MAX) {
            LOG_WARNING(DebuggerErrno::ERROR_VALUE_OVERFLOW,
                        "The value(" + std::to_string(d) + ") exceeds the max value of int64_t, " +
                        "this maybe caused by the unfixed shape operaters.");
            hShape.clear();
            break;
        }
        hShape.push_back(DataUtils::SizeToS64(d));
    }

    // convert format to host format. It can be either NCHW or ND (non 4-dimemsions).
    AclFormat hFmt;
    if (hShape.size() == DIM_4) {
        hFmt = AclFormat::FORMAT_NCHW;
    } else if (hShape.empty()) {
        hFmt = dFmt;
        hShape = dShape;
        LOG_WARNING(DebuggerErrno::NONE,
                    "Tensor(" +  dumpPath + "): The host shape is empty, use device shape as host shape.");
    } else {
        hFmt = AclFormat::FORMAT_ND;
    }

    int32_t subFormat = tensor.sub_format();
    return AclTensorInfo{dumpPath, data, dtype, dtype, dFmt, hFmt,
        dShape, hShape, dataSize, subFormat, io, slot, dumpOriginData};
}

template AclTensorInfo ParseAttrsFromDumpData<AclDumpMsg::OpOutput>(
    const std::string& dumpPath, const uint8_t* data, const AclDumpMsg::OpOutput& tensor, const std::string& io,
    uint32_t slot);
template AclTensorInfo ParseAttrsFromDumpData<AclDumpMsg::OpInput>(
    const std::string& dumpPath, const uint8_t* data, const AclDumpMsg::OpInput& tensor, const std::string& io,
    uint32_t slot);

static inline void AllocTensorTransBuf(AclTensorInfo& tensor)
{
    tensor.transBuf.resize(SizeOfTensor(tensor));
}

static DebuggerErrno FRAC_Z_TO_NCHW_WITH_GROUPS(AclTensorInfo& tensor)
{
    AssertDim(tensor.hostShape, DIM_4);
    AssertConsis(tensor);
    AllocTensorTransBuf(tensor);

    auto nDim = tensor.hostShape[AXIS_N];
    auto cDim = tensor.hostShape[AXIS_C];
    auto hDim = tensor.hostShape[AXIS_H];
    auto wDim = tensor.hostShape[AXIS_W];
    auto groups = tensor.subFormat;
    auto cinOri = cDim;
    if (groups == 0) {
        LOG_WARNING(DebuggerErrno::ERROR_INVALID_VALUE, "groups must not equal to 0.");
        return DebuggerErrno::ERROR_INVALID_VALUE;
    }
    auto coutOri = nDim / groups;

    if (cinOri == 0 || coutOri == 0) {
        LOG_WARNING(DebuggerErrno::ERROR_INVALID_VALUE, tensor + ": cin/cout ori must not equal to 0.");
        return DebuggerErrno::ERROR_INVALID_VALUE;
    }

    auto cubeK = GetCubeSizeByType(tensor.dtype);
    auto eMult = std::min(Lcm(Lcm(cinOri, cubeK) / cinOri, Lcm(coutOri, CUBE_SIZE) / cinOri),
                          static_cast<int64_t>(groups));
    if (eMult == 0) {
        LOG_WARNING(DebuggerErrno::ERROR_INVALID_VALUE,
                    tensor + ": The value of e_mult should be greater than 0.");
        return DebuggerErrno::ERROR_INVALID_VALUE;
    }

    auto cinOpt = AlignCeil(eMult * cinOri, cubeK);
    auto coutOpt = AlignCeil(eMult * coutOri, CUBE_SIZE);
    auto c1Dim = cinOpt / cubeK;
    const uint8_t* src = tensor.aclData;
    auto dst = tensor.transBuf.begin();
    int64_t dtypeSize = static_cast<int64_t>(SizeOfAclDType(tensor));
    int64_t dstSize = static_cast<int64_t>(tensor.transBuf.size());

    for (int64_t g = 0; g < groups; ++g) {
        for (int64_t c = 0; c < cDim; ++c) {
            for (int64_t h = 0; h < hDim; ++h) {
                for (int64_t w = 0; w < wDim; ++w) {
                    for (int64_t n = 0; n < coutOri; ++n) {
                        int64_t eVal = g % eMult;
                        int64_t dstCi = eVal * cinOri + c;
                        int64_t dstCo = eVal * coutOri + n;
                        int64_t srcCo = g * coutOri + n;
                        int64_t temporary = dstCi % cubeK;
                        int64_t devIdx = (g / eMult) * c1Dim * hDim * wDim * coutOpt * cubeK +
                                        (dstCi / cubeK) * hDim * wDim * coutOpt * cubeK + h * wDim * coutOpt * cubeK +
                                        w * coutOpt * cubeK + dstCo * cubeK + temporary;
                        int64_t hstIdx = srcCo * cDim * hDim * wDim + c * hDim * wDim + h * wDim + w;
                        int64_t devOffset = devIdx * dtypeSize;
                        int64_t hstOffset = hstIdx * dtypeSize;
                        if (hstOffset  + dtypeSize > dstSize) {
                            return DebuggerErrno::ERROR_INVALID_VALUE;
                        }
                        std::copy(src + devOffset, src + devOffset + dtypeSize,
                                  dst + hstOffset);
                    }
                }
            }
        }
    }
    return DebuggerErrno::OK;
}

static DebuggerErrno FRAC_Z_TO_NCHW(AclTensorInfo& tensor)
{
    if (tensor.subFormat > 1) {
        return FRAC_Z_TO_NCHW_WITH_GROUPS(tensor);
    }

    AssertDim(tensor.hostShape, DIM_4);
    AssertConsis(tensor);
    AllocTensorTransBuf(tensor);

    auto n0 = tensor.deviceShape.at(FZ_N0);
    auto ni = tensor.deviceShape.at(FZ_NI);
    auto c0 = tensor.deviceShape.at(FZ_C0);
    auto n = tensor.hostShape[AXIS_N];
    auto c = tensor.hostShape[AXIS_C];
    auto h = tensor.hostShape[AXIS_H];
    auto w = tensor.hostShape[AXIS_W];
    auto nc = ni * n0;
    auto ncc0 = nc * c0;
    auto wncc0 = w * ncc0;
    auto hwncc0 = h * wncc0;
    auto hw = h * w;
    auto chw = c * hw;

    if (c0 == 0) {
        return DebuggerErrno::ERROR_INVALID_VALUE;
    }

    const uint8_t* src = tensor.aclData;
    auto dst = tensor.transBuf.begin();
    int64_t dtypeSize = static_cast<int64_t>(SizeOfAclDType(tensor));
    int64_t dstSize = static_cast<int64_t>(tensor.transBuf.size());
    for (int64_t nIdx = 0; nIdx < n; nIdx++) {
        int64_t nHeadAddr = nIdx * chw;
        for (int64_t cIdx = 0; cIdx < c; cIdx++) {
            int64_t cHeadAddr = nHeadAddr + cIdx * hw;
            for (int64_t hIdx = 0; hIdx < h; hIdx++) {
                int64_t hHeadAddr = cHeadAddr + hIdx * w;
                for (int64_t wIdx = 0; wIdx < w; wIdx++) {
                    auto dstIdx = hHeadAddr + wIdx;
                    auto c1Idx = cIdx / c0;
                    auto c0Idx = cIdx % c0;
                    auto ncIdx = nIdx;
                    auto srcIdx = c1Idx * hwncc0 + hIdx * wncc0 + wIdx * ncc0 + ncIdx * c0 + c0Idx;
                    auto dstOffset = dstIdx * dtypeSize;
                    auto srcOffset = srcIdx * dtypeSize;
                    if (dstOffset  + dtypeSize > dstSize) {
                        return DebuggerErrno::ERROR_INVALID_VALUE;
                    }
                    std::copy(src + srcOffset, src + srcOffset + dtypeSize,
                              dst + dstOffset);
                }
            }
        }
    }
    return DebuggerErrno::OK;
}

static void TransShapeToHwNz(const AclShape &hostShape, AclShape& hwShape)
{
    if (hostShape.size() == DIM_1) {
        hwShape.push_back(1);
        hwShape.push_back(1);
        hwShape.push_back(hostShape[0]);
        return;
    }
    auto size = hostShape.size();
    int64_t times = 1;
    for (size_t i = 0; i != size - DIM_2; i++) {
        times *= hostShape[i];
    }
    hwShape.push_back(times);
    hwShape.push_back(hostShape[size - DIM_2]);
    hwShape.push_back(hostShape[size - DIM_1]);
}

static DebuggerErrno FRAC_NZ_TO_NCHW(AclTensorInfo& tensor)
{
    AssertConsis(tensor);
    AllocTensorTransBuf(tensor);

    AclShape hwShape;
    TransShapeToHwNz(tensor.hostShape, hwShape);
    auto times = hwShape.at(0);
    auto h = hwShape.at(HW_H);
    auto w = hwShape.at(HW_W);
    auto hw = h * w;

    auto shapeSize = tensor.deviceShape.size();
    if (shapeSize < DIM_4) {
        LOG_WARNING(DebuggerErrno::ERROR_INVALID_VALUE, tensor + ": Invalid shape size.");
        return DebuggerErrno::ERROR_INVALID_VALUE;
    }

    auto w1 = tensor.deviceShape[shapeSize - FNZ_W1];
    auto h1 = tensor.deviceShape[shapeSize - FNZ_H1];
    auto h0 = tensor.deviceShape[shapeSize - FNZ_H0];
    auto w0 = tensor.deviceShape[shapeSize - FNZ_W0];
    auto h1h0w0 = h1 * h0 * w0;
    auto w1h1h0w0 = w1 * h1h0w0;
    if (w0 == 0) {
        LOG_WARNING(DebuggerErrno::ERROR_INVALID_VALUE, tensor + ": Invalid shape size.");
        return DebuggerErrno::ERROR_INVALID_VALUE;
    }
    auto numW1 = w / w0;

    const uint8_t* src = tensor.aclData;
    auto dst = tensor.transBuf.begin();
    int64_t dtypeSize = static_cast<int64_t>(SizeOfAclDType(tensor));
    int64_t dstSize = static_cast<int64_t>(tensor.transBuf.size());

    for (int64_t timesIdx = 0; timesIdx < times; timesIdx++) {
        auto timesHead = timesIdx * w1h1h0w0;
        auto srcTimesHead = timesIdx * hw;
        for (int64_t h1h0Idx = 0; h1h0Idx < h; h1h0Idx++) {
            auto h1h0Head = timesHead + h1h0Idx * w0;
            auto srcHHead = srcTimesHead + h1h0Idx * w;
            for (int64_t w1Idx = 0; w1Idx < numW1; w1Idx++) {
                for (int64_t i = 0; i < w0; ++i) {
                    int64_t srcIdx = h1h0Head + w1Idx * h1h0w0 + i;
                    int64_t dstIdx = srcHHead + w1Idx * w0 + i;
                    int64_t dstOffset = dstIdx * dtypeSize;
                    int64_t srcOffset = srcIdx * dtypeSize;
                    if (dstOffset  + dtypeSize > dstSize) {
                        return DebuggerErrno::ERROR_INVALID_VALUE;
                    }
                    std::copy(src + srcOffset, src + srcOffset + dtypeSize,
                              dst + dstOffset);
                }
            }
            auto w1Head = numW1 * w0;
            for (int64_t w0Idx = 0; w1Head + w0Idx < w; w0Idx++) {
                auto srcWIdx = w1Head + w0Idx;
                int64_t srcIdx = h1h0Head + numW1 * h1h0w0 + w0Idx;
                int64_t dstIdx = srcHHead + srcWIdx;
                int64_t dstOffset = dstIdx * dtypeSize;
                int64_t srcOffset = srcIdx * dtypeSize;
                if (dstOffset  + dtypeSize > dstSize) {
                    return DebuggerErrno::ERROR_INVALID_VALUE;
                }
                std::copy(src + srcOffset, src + srcOffset + dtypeSize, dst + dstOffset);
            }
        }
    }
    return DebuggerErrno::OK;
}

static DebuggerErrno NC1HWC0_TO_NCHW(AclTensorInfo& tensor)
{
    AssertDim(tensor.hostShape, DIM_4);
    AssertConsis(tensor);
    AllocTensorTransBuf(tensor);

    auto n = tensor.hostShape[AXIS_N];
    auto c = tensor.hostShape[AXIS_C];
    auto h = tensor.hostShape[AXIS_H];
    auto w = tensor.hostShape[AXIS_W];
    auto c1 = tensor.deviceShape[DIM_1];
    auto c0 = tensor.deviceShape[DIM_4];
    if (c0 == 0) {
        LOG_WARNING(DebuggerErrno::ERROR_INVALID_VALUE, tensor + ": Invalid shape size.");
        return DebuggerErrno::ERROR_INVALID_VALUE;
    }

    auto hw = h * w;
    auto chw = c * hw;
    auto wc0 = w * c0;
    auto hwc0 = h * wc0;
    auto c1hwc0 = c1 * hwc0;

    const uint8_t* src = tensor.aclData;
    auto dst = tensor.transBuf.begin();
    int64_t dtypeSize = static_cast<int64_t>(SizeOfAclDType(tensor));
    int64_t dstSize = static_cast<int64_t>(tensor.transBuf.size());
    for (int64_t nIndex = 0; nIndex < n; nIndex++) {
        int64_t nHeadAddr = nIndex * chw;
        for (int64_t cIndex = 0; cIndex < c; cIndex++) {
            int64_t cHeadAddr = nHeadAddr + cIndex * hw;
            for (int64_t hIndex = 0; hIndex < h; hIndex++) {
                int64_t hHeadAddr = cHeadAddr + hIndex * w;
                for (int64_t wIndex = 0; wIndex < w; wIndex++) {
                    int64_t dstIdx = hHeadAddr + wIndex;
                    int64_t c1Index = cIndex / c0;
                    int64_t c0Index = cIndex % c0;
                    int64_t srcIdx = nIndex * c1hwc0 + c1Index * hwc0 + hIndex * wc0 + wIndex * c0 + c0Index;
                    int64_t dstOffset = dstIdx * dtypeSize;
                    int64_t srcOffset = srcIdx * dtypeSize;
                    if (dstOffset  + dtypeSize > dstSize) {
                        return DebuggerErrno::ERROR_INVALID_VALUE;
                    }
                    std::copy(src + srcOffset, src + srcOffset + dtypeSize,
                              dst + dstOffset);
                }
            }
        }
    }
    return DebuggerErrno::OK;
}

static DebuggerErrno NDC1HWC0_TO_NCDHW(AclTensorInfo& tensor)
{
    AssertDim(tensor.hostShape, DIM_5);
    AssertConsis(tensor);
    AllocTensorTransBuf(tensor);

    auto n = tensor.hostShape[N_NCDHW];
    auto c = tensor.hostShape[C_NCDHW];
    auto d = tensor.hostShape[D_NCDHW];
    auto h = tensor.hostShape[H_NCDHW];
    auto w = tensor.hostShape[W_NCDHW];
    auto c1 = tensor.deviceShape[C1_NDC1HWC0];
    auto c0 = tensor.deviceShape[C0_NDC1HWC0];
    if (c0 == 0) {
        LOG_WARNING(DebuggerErrno::ERROR_INVALID_VALUE, tensor + ": Invalid shape size.");
        return DebuggerErrno::ERROR_INVALID_VALUE;
    }

    const int64_t cdhw = c * d * h * w;
    const int64_t dhw = d * h * w;
    const int64_t hw = h * w;
    const int64_t dc1hwc0 = d * c1 * h * w * c0;
    const int64_t c1hwc0 = c1 * h * w * c0;
    const int64_t hwc0 = h * w * c0;
    const int64_t wc0 = w * c0;

    const uint8_t* src = tensor.aclData;
    auto dst = tensor.transBuf.begin();
    int64_t dtypeSize = static_cast<int64_t>(SizeOfAclDType(tensor));
    int64_t dstSize = static_cast<int64_t>(tensor.transBuf.size());
    for (int64_t nIndex = 0; nIndex < n; nIndex++) {
        int64_t nHead = nIndex * cdhw;
        for (int64_t cIndex = 0; cIndex < c; cIndex++) {
            int64_t cHead = nHead + cIndex * dhw;
            for (int64_t dIndex = 0; dIndex < d; dIndex++) {
                int64_t dHead = cHead + dIndex * hw;
                for (int64_t hIndex = 0; hIndex < h; hIndex++) {
                    int64_t hHead = dHead + hIndex * w;
                    for (int64_t wIndex = 0; wIndex < w; wIndex++) {
                        int64_t dstIdx = hHead + wIndex;
                        int64_t c1Index = cIndex / c0;
                        int64_t c0Index = cIndex % c0;
                        auto srcIdx = nIndex * dc1hwc0 + dIndex * c1hwc0 + c1Index * hwc0 + hIndex * wc0 +
                                      wIndex * c0 + c0Index;
                        int64_t dstOffset = dstIdx * dtypeSize;
                        int64_t srcOffset = srcIdx * dtypeSize;
                        if (dstOffset  + dtypeSize > dstSize) {
                            return DebuggerErrno::ERROR_INVALID_VALUE;
                        }
                        std::copy(src + srcOffset, src + srcOffset + dtypeSize,
                                  dst + dstOffset);
                    }
                }
            }
        }
    }
    return DebuggerErrno::OK;
}

static DebuggerErrno C1HWNCoC0_TO_NCHW(AclTensorInfo& tensor)
{
    AssertDim(tensor.hostShape, DIM_4);
    AssertConsis(tensor);
    AllocTensorTransBuf(tensor);

    auto n = tensor.hostShape[AXIS_N];
    auto c = tensor.hostShape[AXIS_C];
    auto h = tensor.hostShape[AXIS_H];
    auto w = tensor.hostShape[AXIS_W];
    const int coIdx = 4;
    const int c0Idx = 5;
    auto co = tensor.deviceShape[coIdx];
    auto c0 = tensor.deviceShape[c0Idx];
    auto cubeK = GetCubeSizeByType(tensor.dtype);

    const uint8_t* src = tensor.aclData;
    auto dst = tensor.transBuf.begin();
    int64_t dtypeSize = static_cast<int64_t>(SizeOfAclDType(tensor));
    int64_t dstSize = static_cast<int64_t>(tensor.transBuf.size());
    for (int64_t nIndex = 0; nIndex < n; nIndex++) {
        for (int64_t cIndex = 0; cIndex < c; cIndex++) {
            for (int64_t hIndex = 0; hIndex < h; hIndex++) {
                for (int64_t wIndex = 0; wIndex < w; wIndex++) {
                    int64_t dstIdx = nIndex * c * h * w + cIndex * h * w + hIndex * w + wIndex;
                    int64_t c1Index = cIndex / cubeK;
                    int64_t c0Index = cIndex % cubeK;
                    int64_t coIndex = c0Index;
                    int64_t srcIdx = c1Index * h * w * n * co * c0 + hIndex * w * n * co * c0 + wIndex * n * co * c0 +
                            nIndex * co * c0 + coIndex * c0 + c0Index;
                    int64_t dstOffset = dstIdx * dtypeSize;
                    int64_t srcOffset = srcIdx * dtypeSize;
                    if (dstOffset  + dtypeSize > dstSize) {
                        return DebuggerErrno::ERROR_INVALID_VALUE;
                    }
                    std::copy(src + srcOffset, src + srcOffset + dtypeSize,
                              dst + dstOffset);
                }
            }
        }
    }
    return DebuggerErrno::OK;
}

static DebuggerErrno NC1HWC0_C04_TO_NCHW(AclTensorInfo& tensor)
{
    return NC1HWC0_TO_NCHW(tensor);
}

static DebuggerErrno FRAC_Z3D_TO_NCDHW(AclTensorInfo& tensor)
{
    AssertDim(tensor.hostShape, DIM_5);
    AssertConsis(tensor);
    AllocTensorTransBuf(tensor);

    auto n = tensor.hostShape[N_NCDHW];
    auto c = tensor.hostShape[C_NCDHW];
    auto d = tensor.hostShape[D_NCDHW];
    auto h = tensor.hostShape[H_NCDHW];
    auto w = tensor.hostShape[W_NCDHW];
    constexpr int FZ3D_C0 = 3;
    auto c0 = tensor.deviceShape[FZ3D_C0];
    if (c0 == 0) {
        LOG_WARNING(DebuggerErrno::ERROR_INVALID_VALUE, tensor + ": Invalid shape size.");
        return DebuggerErrno::ERROR_INVALID_VALUE;
    }
    auto cube_k = GetCubeSizeByType(tensor.dtype);
    auto c1 = DivCeil(c, cube_k);
    constexpr int64_t kNiSize = 16;
    auto n1n0 = AlignCeil(n, kNiSize);
    auto n1n0c0 = n1n0 * c0;
    auto wn1n0c0 = w * n1n0c0;
    auto hwn1n0c0 = h * wn1n0c0;
    auto c1hwn1n0c0 = c1 * hwn1n0c0;
    auto hw = h * w;
    auto dhw = d * hw;
    auto cdhw = c * dhw;

    const uint8_t* src = tensor.aclData;
    auto dst = tensor.transBuf.begin();
    int64_t dtypeSize = static_cast<int64_t>(SizeOfAclDType(tensor));
    int64_t dstSize = static_cast<int64_t>(tensor.transBuf.size());
    for (int64_t nIdx = 0; nIdx < n; nIdx++) {
        int64_t nHead = nIdx * cdhw;
        for (int64_t cIdx = 0; cIdx < c; cIdx++) {
            int64_t cHead = nHead + cIdx * dhw;
            for (int64_t dIdx = 0; dIdx < d; dIdx++) {
                int64_t dHead = cHead + dIdx * hw;
                for (int64_t hIdx = 0; hIdx < h; hIdx++) {
                    int64_t hHead = dHead + hIdx * w;
                    for (int64_t wI = 0; wI < w; wI++) {
                        int64_t dstIdx = hHead + wI;
                        int64_t c1I = cIdx / c0;
                        int64_t c0I = cIdx % c0;
                        int64_t ncIdx = nIdx;
                        int64_t srcIdx = dIdx * c1hwn1n0c0 + c1I * c1hwn1n0c0 + hIdx * wn1n0c0 + wI * n1n0c0 +
                                           ncIdx * c0 + c0I;
                        int64_t dstOffset = dstIdx * dtypeSize;
                        int64_t srcOffset = srcIdx * dtypeSize;
                        if (dstOffset  + dtypeSize > dstSize) {
                            return DebuggerErrno::ERROR_INVALID_VALUE;
                        }
                        std::copy(src + srcOffset, src + srcOffset + dtypeSize,
                                  dst + dstOffset);
                    }
                }
            }
        }
    }
    return DebuggerErrno::OK;
}

DebuggerErrno TransFormatD2H(AclTensorInfo& tensor)
{
    AclFormat from = tensor.deviceFmt;
    AclFormat to = tensor.hostFmt;
    auto it = formatTransFuncMap.find(std::make_pair(from, to));
    if (it == formatTransFuncMap.end()) {
        return DebuggerErrno::ERROR_UNKNOWN_TRANS;
    }

    try {
        return it->second(tensor);
    } catch (const std::exception& e) {
        LOG_ERROR(DebuggerErrno::ERROR_OPERATION_FAILED, tensor + ": Failed to conver dtype from " +
                  std::to_string(from) + " to " + std::to_string(to) + "(" + e.what() + ").");
        return DebuggerErrno::ERROR_OPERATION_FAILED;
    }
}

static DebuggerErrno TransBf16ToFp32(const uint8_t* input, size_t num, uint8_t* output, size_t bufferSize)
{
    if (bufferSize < num * sizeof(float)) {
        LOG_ERROR(DebuggerErrno::ERROR_BUFFER_OVERFLOW, "Insufficient space for converting data from bf16 to fp32.");
        return DebuggerErrno::ERROR_BUFFER_OVERFLOW;
    }
    const DataUtils::BFloat16* in = reinterpret_cast<const DataUtils::BFloat16*>(input);
    float* out = reinterpret_cast<float*>(output);

    for (size_t i = 0; i < num; i++) {
        out[i] = static_cast<float>(in[i]);
    }
    return DebuggerErrno::OK;
}

static DebuggerErrno TransInt4ToInt8(const uint8_t* input,
                                     size_t elemNums,
                                     uint8_t* output,
                                     size_t bufferSize)
{
    // 输出缓冲区要能容纳 elemNums 个 int8_t
    if (bufferSize < elemNums * sizeof(int8_t)) {
        LOG_ERROR(DebuggerErrno::ERROR_BUFFER_OVERFLOW,
                  "Insufficient space for converting data from int4 to int8.");
        return DebuggerErrno::ERROR_BUFFER_OVERFLOW;
    }

    const uint8_t* srcData = input;       // 原始数据按字节读取
    int8_t*       dstData = reinterpret_cast<int8_t*>(output);
    size_t        inputLength = elemNums / 2;

    const int8_t  maxValue     =  7;
    const int8_t  minValue     = -8;
    const uint8_t signBitMask  = 0x08;
    const int     signBitShift = 3;

    for (size_t i = 0; i < inputLength; ++i) {
        uint8_t byte = srcData[i];

        // —— 低 4 位 ——
        uint8_t u = byte & 0x0F;  // 在无符号变量上做 AND
        uint8_t sign = (u & signBitMask) >> signBitShift;
        if (sign) {
            u |= 0xF0;             // 在无符号变量上做 OR
        }
        // 转回有符号并检查范围
        int8_t t = static_cast<int8_t>(u);
        if (t < minValue || t > maxValue) {
            LOG_ERROR(DebuggerErrno::ERROR_INVALID_VALUE,
                      "Invalid int4 value (low nibble).");
        }
        *dstData++ = t;

        // —— 高 4 位 ——
        u = (byte >> 4) & 0x0F;   // 无符号右移后截低 4 位
        sign = (u & signBitMask) >> signBitShift;
        if (sign) {
            u |= 0xF0;
        }
        t = static_cast<int8_t>(u);
        if (t < minValue || t > maxValue) {
            LOG_ERROR(DebuggerErrno::ERROR_INVALID_VALUE,
                      "Invalid int4 value (high nibble).");
        }
        *dstData++ = t;
    }

    return DebuggerErrno::OK;
}

DebuggerErrno TransDtype(AclTensorInfo& tensor, AclDtype to)
{
    if (tensor.dtype == to) {
        return DebuggerErrno::OK;
    }

    tensor.oriDtype = tensor.dtype;
    std::vector<uint8_t> buffer;
    try {
        AssertConsis(tensor);
    } catch (const std::runtime_error& e) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_OPERATION, e.what());
        return DebuggerErrno::ERROR_INVALID_OPERATION;
    }
    size_t bufferSize = EleNumOfTensor(tensor) * SizeOfAclDType(to);
    buffer.resize(bufferSize);
    const uint8_t* input = tensor.transBuf.empty() ? tensor.aclData : tensor.transBuf.data();
    uint8_t* output = buffer.data();
    DebuggerErrno ret;

    if (tensor.dtype == AclDtype::DT_BF16 && to == AclDtype::DT_FLOAT) {
        ret = TransBf16ToFp32(input, EleNumOfTensor(tensor), output, bufferSize);
    } else if (tensor.dtype == AclDtype::DT_INT4 && to == AclDtype::DT_INT8) {
        ret = TransInt4ToInt8(input, EleNumOfTensor(tensor), output, bufferSize);
    } else {
        LOG_ERROR(DebuggerErrno::ERROR_UNKNOWN_TRANS, tensor + ": Trans " + DataUtils::GetDTypeString(tensor.dtype)
                  + " to " + DataUtils::GetDTypeString(to) + " is not supported.");
        return DebuggerErrno::ERROR_UNKNOWN_TRANS;
    }

    if (ret != DebuggerErrno::OK) {
        return ret;
    }

    tensor.transBuf = std::move(buffer);
    tensor.dtype = to;
    return DebuggerErrno::OK;
}

}
}