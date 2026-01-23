/* -------------------------------------------------------------------------
 *  This file is part of the MindStudio project.
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
 * ------------------------------------------------------------------------- */


#include "atb_probe.h"

#include <syscall.h>
#include <cctype>
#include <cstdlib>
#include <dlfcn.h>
#include <cstdio>
#include <unordered_map>
// for calculating dumped-tensor statistics
#include <vector>
#include <thread>
#include <algorithm>
#include <functional>
#include <memory>
#include <complex>
#include <utility>
#include <chrono>
#include "Statistics.h"
// endfor
#include <unistd.h>
#include <sys/statvfs.h>
#include <sys/stat.h>

#include "bin_file.h"
#include "nlohmann/json.hpp"
#include "ait_logger.h"
#include "utils.h"
#include "umask_wrapper.h"
#include "DumpThreadPool.h"
#include "const.h"
#include "safety_guard.h"
#include "file.h"

using ordered_json = nlohmann::ordered_json;
using MsConst::SAFETY_RET;

// C++11标准不支持make_unique函数，此处为make_unique的简单实现
#if __cplusplus < 201402L
namespace std {
    // 非数组类型
    template<typename T, typename... Args>
    typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type make_unique(Args&&... args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    // 数组类型(T[])
    template <typename T>
    using EnableIfDynamicArray = typename std::enable_if<
        std::is_array<T>::value && std::extent<T>::value == 0,
        std::unique_ptr<T>
    >::type;

    template<typename T>
    EnableIfDynamicArray<T> make_unique(size_t size)
    {
        if (size <= 0) {
            throw std::invalid_argument("make_unique: array size must be greater than 0");
        }
        using U = typename std::remove_extent<T>::type;
        return std::unique_ptr<T>(new U[size]);
    }

    // 禁用定长数组(如T[5])
    template<typename T, typename... Args>
    typename std::enable_if<std::extent<T>::value != 0, std::unique_ptr<T>>::type make_unique(Args&&...) = delete;
}
#endif

namespace {
    unsigned long long g_minDiskSpaceFreeSize = 2147483648; // 2G
    static const int POOLNUM = 12;
    constexpr size_t FREE_SIZE_MULTIPLE_OF_DATA_SIZE = 2; // free size至少两倍data size大小
    constexpr uint32_t MAX_RECUR_DEPTH = 1000;
    constexpr uint16_t MAX_PATH_LENGTH = 2048;
    constexpr const char* LIBASCENDCL_SO = "libascendcl.so";
    constexpr const char* ASCEND_TOOLKIT_HOME = "ASCEND_TOOLKIT_HOME";

    template <typename T>
    inline bool Likely(T&& condition) noexcept
    {
        return __builtin_expect(static_cast<bool>(condition), 1);
    }

    template <typename T>
    inline bool Unlikely(T&& condition) noexcept
    {
        return __builtin_expect(static_cast<bool>(condition), 0);
    }

    static int SafetyStoi(const char* env, int defaultValue)
    {
        int ans = defaultValue;
        try {
            ans = std::stoi(env);
        } catch (const std::invalid_argument& e) {
            AIT_LOG_ERROR("The argument passed in can not be converted to int.");
            return ans;
        } catch (const std::out_of_range& e) {
            AIT_LOG_ERROR("The value of argument is out of range.");
            return ans;
        }
        return ans;
    }

    struct LayerGraphMap {
        std::map<std::string, std::string> layerGraphMap_;

        void SaveLayerGraph(const std::string &opName, const std::string &graph)
        {
            layerGraphMap_[opName] = graph;
        };

        std::string GetLayerGraph(const std::string &opName)
        {
            auto it = layerGraphMap_.find(opName);
            return (it == layerGraphMap_.end()) ? "" : it->second;
        };
    };

    std::unique_ptr<ThreadPool::DumpThreadPool>& GetGlobalPool(size_t numThreads = POOLNUM)
    {
        static std::unique_ptr<ThreadPool::DumpThreadPool> instance =
            std::make_unique<ThreadPool::DumpThreadPool>(numThreads);
        return instance;
    }

    template<typename T>
    bool ParseJsonBaseObj2Var(const nlohmann::json& content, const std::string& field, T& output)
    {
        nlohmann::json::const_iterator iter = content.find(field);
        if (iter == content.end()) {return false;}
        try {
            output = iter->get<T>();
            return true;
        } catch (const nlohmann::detail::type_error& e) {
            return false;
        }
    }
}

static int GetFreeSpace(std::string path, unsigned long long &freeSpace)
{
    struct statvfs diskInfo;

    if (statvfs(path.c_str(), &diskInfo) == -1) {
        AIT_LOG_ERROR("statvfs() error:" + Utils::GetLastErrorStr());
        return 1;
    }
    freeSpace = diskInfo.f_bavail * diskInfo.f_bsize;
    return 0;
}

static int32_t GetCurrentDeviceId()
{
    int32_t deviceId = -1;
    const char* ascendToolkitHome = std::getenv(ASCEND_TOOLKIT_HOME);
    if (ascendToolkitHome == nullptr) { return deviceId; }
    std::string ascendclPath = std::string(ascendToolkitHome) + "/lib64/" + LIBASCENDCL_SO;
    struct stat fileStat;
    if (stat(ascendclPath.c_str(), &fileStat) != 0) { return deviceId; }
    if (getuid() != 0 && fileStat.st_uid != getuid()) { return deviceId; }
    mode_t permissions = fileStat.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
    if ((permissions & MsConst::READ_FILE_NOT_PERMITTED) > 0) { return deviceId; }
    void* handle = dlopen(ascendclPath.c_str(), RTLD_LAZY);
    if (handle == nullptr) { return deviceId; }
    int (*getDeviceFunc)(int32_t*) = reinterpret_cast<int (*)(int32_t*)>(dlsym(handle, "aclrtGetDevice"));
    if (getDeviceFunc == nullptr) {
        dlclose(handle);
        return deviceId;
    }
    int aclRet = getDeviceFunc(&deviceId);
    if (aclRet != 0) {
        deviceId = -1;
    }
    dlclose(handle);
    return deviceId;
}

static int32_t GetCurrentProcessId()
{
    int32_t pid = getpid();
    if (pid == -1) {
        AIT_LOG_ERROR("get pid failed");
    }
    return pid;
}

static bool IsPrefix(const std::string &str, const std::string &prefix)
{
    return str.compare(0, prefix.length(), prefix) == 0;
}

static const std::string& GetOutDir()
{
    static std::string outDir = "";
    if (outDir == "") {
        const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
        outDir = (outputDir != nullptr ? outputDir : "./");
        outDir = GetRealPath(outDir);
        if (outDir.length() > MAX_PATH_LENGTH) {
            AIT_LOG_ERROR("The path length of ATB_OUTPUT_DIR must not be greater than " +
                          std::to_string(MAX_PATH_LENGTH) + " characters, but got " + std::to_string(outDir.length()));
            outDir = "";
            return outDir;
        }
        while (!outDir.empty() && outDir.back() == '/') {
            outDir.pop_back();
        }
        outDir = outDir + "/atb_dump_data/";
        bool ret = Utils::CheckDirectory(outDir);
        if (!ret) {
            AIT_LOG_ERROR("Create directory failed: " + outDir);
            outDir = "";
        }
    }
    return outDir;
}

static bool IsInTensorBinPath(const std::string &filePath)
{
    size_t sepPos = filePath.rfind('/');
    std::string fileName = filePath;
    if (sepPos != std::string::npos) {
        fileName.erase(0, sepPos + 1U);
    }
    bool flag = (fileName.find("intensor") != std::string::npos) || (fileName.find("inTensor") != std::string::npos);
    AIT_LOG_DEBUG("IsInTensorBinPath: " + std::to_string(flag));
    return flag;
}

static bool IsOutTensorBinPath(const std::string &filePath)
{
    size_t sepPos = filePath.rfind('/');
    std::string fileName = filePath;
    if (sepPos != std::string::npos) {
        fileName.erase(0, sepPos + 1U);
    }
    bool flag = (fileName.find("outtensor") != std::string::npos) || (fileName.find("outTensor") != std::string::npos);
    AIT_LOG_DEBUG("IsOutTensorBinPath: " + std::to_string(flag));
    return flag;
}

static void DfsToModifyGraphTensors(ordered_json &curNodeToSave,
    const std::vector<std::string> &fatherNodeTensorNameList, const ordered_json &curNodeInput, uint32_t curDepth = 1)
{
    if (curDepth >= MAX_RECUR_DEPTH) {
        AIT_LOG_ERROR("The depth of the current graph structure is too deep, and the traversal node fails");
        throw std::runtime_error("The maximum recursive depth exceeds " + std::to_string(MAX_RECUR_DEPTH) + ".");
    }
    std::string opName = curNodeInput["opName"].get<std::string>();
    curNodeToSave["opName"] = curNodeInput["opName"];
    curNodeToSave["opType"] = curNodeInput["opType"];
    curNodeToSave["param"] = curNodeInput["param"];

    std::vector<std::string> curNodeTensorNameList;

    // 子节点的inTensors\outTensors为父节点的fatherNodeTensorNameList子集, 根据inTensorIds、outTensorIds获取
    for (auto item : curNodeInput["inTensorIds"]) {
        uint32_t inputIndex = item.get<uint32_t>();
        if (inputIndex >= fatherNodeTensorNameList.size()) {
            AIT_LOG_ERROR("inputIndex out of fatherNodeTensorNameList: " + opName);
            return;
        }
        curNodeToSave["inTensors"].emplace_back(fatherNodeTensorNameList[inputIndex]);
        curNodeTensorNameList.emplace_back(fatherNodeTensorNameList[inputIndex]);
    }

    for (auto item : curNodeInput["outTensorIds"]) {
        uint32_t outputIndex = item.get<uint32_t>();
        if (outputIndex >= fatherNodeTensorNameList.size()) {
            AIT_LOG_ERROR("outputIndex out of fatherNodeTensorNameList: " + opName);
            return;
        }
        curNodeToSave["outTensors"].emplace_back(fatherNodeTensorNameList[outputIndex]);
        curNodeTensorNameList.emplace_back(fatherNodeTensorNameList[outputIndex]);
    }

    // 子节点的internalTensors根据自己的opName + id, 组成tensor name
    uint32_t internalTensorNum = (curNodeInput.find("internalTensorNum") == curNodeInput.end()) ?
                                  0 : curNodeInput["internalTensorNum"].get<uint32_t>();
    for (size_t i = 0; i < internalTensorNum; i++) {
        std::string tensorName = opName + "_internal_" + std::to_string(i);
        curNodeToSave["internalTensors"].emplace_back(tensorName);
        curNodeTensorNameList.emplace_back(tensorName);
    }

    // 递归调用获取子节点信息
    if (curNodeInput.find("nodes") != curNodeInput.end()) {
        for (auto childNodeInput : curNodeInput["nodes"]) {
            ordered_json childNodeToSave;
            DfsToModifyGraphTensors(childNodeToSave, curNodeTensorNameList, childNodeInput, curDepth + 1);
            curNodeToSave["nodes"].emplace_back(childNodeToSave);
        }
    }
    return;
}

static LayerGraphMap g_layerGraphMap;
static unsigned long long g_aitOperationBaseId(0);
static void MergeLayerTopoInfo(ordered_json &layerJson)
{
    // 获取atb仓打桩保存的layer的拓扑信息
    layerJson["opType"] = layerJson["opName"];
    std::string opName = layerJson["opName"].get<std::string>()+ "_" + std::to_string(g_aitOperationBaseId++);
    layerJson["opName"] = opName;
    std::string atbLayerGraph = g_layerGraphMap.GetLayerGraph(opName);
    if (atbLayerGraph == "") {
        return;
    }
    ordered_json atbLayerJson;
    // inTensor和outTensor从model里获取，internalTensor自己申请id, 组成layerTensorNameList
    try {
        atbLayerJson = ordered_json::parse(atbLayerGraph);
    } catch (const ordered_json::parse_error& ex) {
        AIT_LOG_ERROR("json parse error! opName:" + opName);
        AIT_LOG_ERROR("message: " + std::string(ex.what()) + '\n' + "exception id: " + std::to_string(ex.id) + '\n' +
               "byte position of error: " + std::to_string(ex.byte));
        return;
    }
    std::vector<std::string> layerTensorNameList;
    for (auto item : layerJson["inTensors"]) {
        layerTensorNameList.emplace_back(item);
    }

    for (auto item : layerJson["outTensors"]) {
        layerTensorNameList.emplace_back(item);
    }

    uint32_t internalTensorNum = (atbLayerJson.find("internalTensorNum") == atbLayerJson.end()) ?
                                  0 : atbLayerJson["internalTensorNum"].get<uint32_t>();
    for (size_t i = 0; i < internalTensorNum; i++) {
        std::string tensorName = opName + "_internal_" + std::to_string(i);
        layerJson["internalTensors"].emplace_back(tensorName);
        layerTensorNameList.emplace_back(tensorName);
    }

    // 递归调用获取每个layer的子节点信息
    if (atbLayerJson.find("nodes") != atbLayerJson.end()) {
        for (auto childNodeInput : atbLayerJson["nodes"]) {
            ordered_json childNodeToSave;
            DfsToModifyGraphTensors(childNodeToSave, layerTensorNameList, childNodeInput);
            layerJson["nodes"].emplace_back(childNodeToSave);
        }
    }
    return;
}

namespace Mki {

    size_t GetTensorElementSize(const TensorDType dtype)
    {
        auto iter = MAP_OF_DTYPE_SIZE.find(dtype);
        if (iter == MAP_OF_DTYPE_SIZE.end()) {
            AIT_LOG_ERROR("Get Tensor ElementSize:dtype not found!");
            return 0;
        }
        return iter->second;
    }

    TensorDType ConvertToTensorDType(int dType)
    {
        TensorDType tensorDType = TensorDType::TENSOR_DTYPE_UNDEFINED;
        switch (dType) {
            case static_cast<int>(TensorDType::TENSOR_DTYPE_UNDEFINED):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT16):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_INT8):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_INT32):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_UINT8):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_INT16):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_UINT16):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_UINT32):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_INT64):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_UINT64):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_DOUBLE):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_BOOL):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_STRING):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_COMPLEX64):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_COMPLEX128):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_BF16):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_INT4):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_UINT1):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_COMPLEX32):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_HIFLOAT8):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT8_E5M2):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT8_E4M3FN):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT8_E8M0):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT6_E3M2):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT6_E2M3):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT4_E2M1):
            case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT4_E1M2):
                tensorDType = static_cast<TensorDType>(dType);
                break;
            default:
                break;
        }
        return tensorDType;
    }

    TensorFormat ConvertToTensorFormat(int format)
    {
        TensorFormat tensorFormat = TensorFormat::TENSOR_FORMAT_UNDEFINED;
        switch (format) {
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_UNDEFINED):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_NCHW):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_NHWC):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_ND):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_NC1HWC0):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_FRACTAL_Z):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_NC1HWC0_C04):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_HWCN):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_NDHWC):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_FRACTAL_NZ):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_NCDHW):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_NDC1HWC0):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_FRACTAL_Z_3D):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_NC):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_NCL):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_FRACTAL_NZ_C0_16):
            case static_cast<int>(TensorFormat::TENSOR_FORMAT_FRACTAL_NZ_C0_32):
                tensorFormat = static_cast<TensorFormat>(format);
                break;
            default:
                break;
        }
        return tensorFormat;
    }

    const std::string &GetDTypeStr(const TensorDType &dType)
    {
        auto it = MAP_DTYPE_TO_STRING.find(dType);
        if (it != MAP_DTYPE_TO_STRING.end()) {
            return it->second;
        }
        return UNDEFINED_STR;
    }

    const std::string &GetFormatStr(const TensorFormat &format)
    {
        auto it = MAP_FORMAT_TO_STRING.find(format);
        if (it != MAP_FORMAT_TO_STRING.end()) {
            return it->second;
        }
        return UNDEFINED_STR;
    }

    float ConvertToFloat32(uint16_t value, size_t exponentBits, size_t mantissaBits)
    {
        // Determine the bias of the semi-precision type
        int32_t exponentBias = (1 << (exponentBits - 1)) - 1;

        // Obtain the mask
        uint16_t exponentMask = ((1 << exponentBits) - 1) << mantissaBits;
        uint16_t mantissaMask = (1 << mantissaBits) - 1;
        uint16_t signMask = 1 << (exponentBits + mantissaBits);

        // Extract symbol bits
        int sign = (value & signMask) ? -1 : 1;

        // Extract index and mantissa
        int32_t rawExponent = (value & exponentMask) >> mantissaBits;
        uint32_t mantissa = value & mantissaMask;

        // Handle special values
        if (rawExponent == (1 << exponentBits) - 1) { // All 1s represent NaN or infinity
            if (mantissa != 0) {
                return std::numeric_limits<float>::quiet_NaN(); // NaN
            } else {
                return sign * std::numeric_limits<float>::infinity(); // Infinity
            }
        } else if (rawExponent == 0) { // Exponents with all zeros indicate non-normalized numbers or zeros
            if (mantissa == 0) {
                return sign * 0.0f; // Zero
            } else {
                // Unnormalized number
                float result = sign * std::ldexp(static_cast<float>(mantissa),
                1 - static_cast<int>(exponentBias) - static_cast<int>(mantissaBits));
                return result;
            }
        }

        // Normalized number
        float normalizedMantissa = 1.0f + static_cast<float>(mantissa) / (1 << mantissaBits);
        float result = sign * std::ldexp(normalizedMantissa, rawExponent - exponentBias);
        return result;
    }
}

namespace atb {

constexpr size_t CONFIG_UPDATE_PERIOD = 16384;
constexpr size_t LAYER_PATH_DEPTH = 3;
constexpr int NO_FILTER_FOR_DATA = 0;
constexpr int FILTER_OPERATION_DATA = 1;
constexpr int FILTER_KERNEL_DATA = 2;
const std::string TENSOR_TASK = "tensor";
const std::string STATS_TASK = "statistics";
const std::string ALL_TASK = "all";
const std::string INTENSOR_PREFIX = "intensor";
const std::string OUTTENSOR_PREFIX = "outtensor";
const std::string STATS_FILE_NAME = "statistic.csv";
const std::vector<std::string> VALID_TASKS = {TENSOR_TASK, STATS_TASK, ALL_TASK};

struct BinFileInfo {
    std::string format;
    std::string dtype;
    std::string dims;
    std::string filePath;
    const std::string opId;
    const std::string opName;
    const void *hostData;
    const uint64_t dataSize;
};

static size_t g_configUpdateTimes;

static bool g_isAtbRunning;
static bool g_isDebugMode;
static bool g_dumpEnable;
static bool g_saveChild;

static std::string g_configPath;
static std::string g_task;
static std::string g_executionCountRange;
static std::string g_operationId;
static std::string g_operationName;
static std::string g_device;

static int g_filterLevel;

static std::vector<std::string> g_splitOperationIds;
static std::vector<std::string> g_splitOperationNames;
static std::vector<std::string> g_splitDeviceIds;

static std::unordered_map<std::string, std::string> g_realTensors;

static std::set<std::string> g_DumpedLayerSet;

static bool OrderOperations(
    const ordered_json& operation, const std::string& idPrefix, std::vector<std::string>& orderedOperations,
    std::unordered_map<std::string, std::vector<std::string>>& opAndTensors, int& depth)
{
    depth += 1;
    if (depth > MAX_RECUR_DEPTH) { return false; }
    if (operation.find("opName") == operation.end()) { return false; }

    try {
        std::string opName = operation.find("opName")->get<std::string>();
        size_t found = opName.find(idPrefix);
        if (found == std::string::npos) { return false; }

        std::string opId = opName.substr(found + 1);
        orderedOperations.emplace_back(opId);
        opAndTensors[opId + "_" + INTENSOR_PREFIX] = operation.find("inTensors")->get<std::vector<std::string>>();
        opAndTensors[opId + "_" + OUTTENSOR_PREFIX] = operation.find("outTensors")->get<std::vector<std::string>>();

        if (operation.find("nodes") == operation.end()) { return true; }

        auto nodes = operation.find("nodes")->get<std::vector<ordered_json>>();
        for (const auto &node : nodes) {
            if (!OrderOperations(node, idPrefix, orderedOperations, opAndTensors, depth)) {
                return false;
            }
        }
    } catch (const nlohmann::detail::type_error& e) {
        return false;
    }

    return true;
}

static void PreDealWhenPassingOpTensors(const std::string &operationId, std::stack<std::string> &orderedOperationId,
                                        std::vector<std::vector<std::string>> &orderedInputAndOutput)
{
    int loopCount = 0;
    while (!orderedOperationId.empty()) {
        if (++loopCount > MAX_RECUR_DEPTH || operationId.find(orderedOperationId.top()) == 0) {
            return;
        }

        orderedInputAndOutput.emplace_back(std::vector<std::string>{orderedOperationId.top(), OUTTENSOR_PREFIX});
        orderedOperationId.pop();
    }
}

static bool NeedSave(const std::string &tensorName, const std::vector<std::string>& lastTensors,
                     const std::unordered_map<std::string, std::string> &tensorsAliases)
{
    if (tensorsAliases.find(tensorName) == tensorsAliases.end() ||
        std::find(lastTensors.begin(), lastTensors.end(), tensorName) != lastTensors.end()) {
        return true;
    }

    return false;
}

static void FilterUnnecessaryData(const ordered_json& layerStructure)
{
    std::string layerName = layerStructure.find("opName")->get<std::string>();
    size_t found = layerName.rfind('_');
    if (found == std::string ::npos) { return; }
    std::string idPrefix = layerName.substr(found);

    std::vector<std::string> orderedOperations;
    std::unordered_map<std::string, std::vector<std::string>> opAndTensors;
    std::stack<std::string> orderedOperationId;
    std::vector<std::vector<std::string>> orderedInputAndOutput;
    int depth = 0;
    auto ret = OrderOperations(layerStructure, idPrefix, orderedOperations, opAndTensors, depth);
    if (!ret || orderedOperations.empty()) { return; }

    std::string lastId;
    for (const auto &operationId : orderedOperations) {
        if (!lastId.empty() && operationId.find(lastId) != 0) {
            PreDealWhenPassingOpTensors(operationId, orderedOperationId, orderedInputAndOutput);
        }
        orderedInputAndOutput.emplace_back(std::vector<std::string>{operationId, INTENSOR_PREFIX});
        orderedOperationId.emplace(operationId);
        lastId = operationId;
    }

    while (!orderedOperationId.empty()) {
        orderedInputAndOutput.emplace_back(
            std::vector<std::string>{orderedOperationId.top(), OUTTENSOR_PREFIX});
        orderedOperationId.pop();
    }

    std::unordered_map<std::string, std::string> tensorsAliases;
    for (size_t i = 0; i < orderedInputAndOutput.size(); i++) {
        const std::string ioType = orderedInputAndOutput.at(i).at(1);
        std::string opWithIoType = orderedInputAndOutput.at(i).at(0) + "_" + ioType;
        const std::vector<std::string>& tensors = opAndTensors[opWithIoType];
        std::vector<std::string> lastTensors;
        if (i != 0 && ioType == OUTTENSOR_PREFIX && orderedInputAndOutput.at(i - 1).at(1) == INTENSOR_PREFIX) {
            lastTensors = opAndTensors[orderedInputAndOutput.at(i - 1).at(0) + "_" + INTENSOR_PREFIX];
        }
        for (size_t j = 0; j < tensors.size(); j++) {
            if (NeedSave(tensors[j], lastTensors, tensorsAliases)) {
                g_realTensors[opWithIoType + std::to_string(j) + ".bin"] = "bin";
                tensorsAliases[tensors.at(j)] = opWithIoType + std::to_string(j) + ".bin";
            } else {
                g_realTensors[opWithIoType + std::to_string(j) + ".bin"] = tensorsAliases.at(tensors.at(j));
            }
        }
    }
}

static std::vector<std::string> SplitDataPath(const std::string &dataPath)
{
    std::vector<std::string> splitPath = SplitString(dataPath.c_str(), '/');
    if (splitPath.size() < LAYER_PATH_DEPTH) {
        AIT_LOG_WARNING("Expected a valid data path whose depth must not be less than " +
                        std::to_string(LAYER_PATH_DEPTH) + " but got " + std::to_string(splitPath.size()));
        splitPath.clear();
    }

    return splitPath;
}

static std::string GetOpIdFromDataPath(const std::vector<std::string> &dataPathVec, size_t endOffset)
{
    std::string opId;
    if (dataPathVec.size() <= endOffset) { return opId; }
    for (size_t i = 2; i < dataPathVec.size() - endOffset; ++i) {
        auto found = dataPathVec[i].find('_');
        if (found == std::string::npos) {
            return "";
        }
        std::string singleId = dataPathVec[i].substr(0, found);
        if (i != 2U) {
            opId += "_" + singleId;
        } else {
            opId += singleId;
        }
    }
    return opId;
}

static std::string GetFullOpNameFromDataPath(const std::vector<std::string> &dataPathVec, size_t endOffset)
{
    std::string fullOpName;
    if (dataPathVec.size() <= endOffset) { return fullOpName; }
    for (size_t i = 2; i < dataPathVec.size() - endOffset; ++i) {
        auto found = dataPathVec[i].find('_');
        if (found == std::string::npos) {
            return "";
        }
        std::string singleOpName = dataPathVec[i].substr(found + 1);
        if (i != 2U) {
            fullOpName += "/" + singleOpName;
        } else {
            fullOpName += singleOpName;
        }
    }
    return fullOpName;
}
} // end of namespace atb
