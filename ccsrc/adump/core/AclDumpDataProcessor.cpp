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

#include <cstring>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <chrono>
#include <sys/file.h>

#include "include/Macro.h"
#include "utils/FileUtils.h"
#include "utils/FileOperation.h"
#include "utils/DataUtils.h"
#include "utils/MathUtils.h"
#include "core/AclTensor.h"
#include "base/ErrorInfosManager.h"
#include "third_party/proto/AclDumpMsg.pb.h"
#include "AclDumpDataProcessor.h"

namespace MindStudioDebugger {

namespace AclDumpMsg = toolkit::dumpdata;

constexpr size_t DHA_ATOMIC_ADD_INFO_SIZE = 128;
constexpr size_t L2_ATOMIC_ADD_INFO_SIZE = 128;
constexpr size_t AICORE_INFO_SIZE = 256;
constexpr size_t DHA_ATOMIC_ADD_STATUS_SIZE = 256;
constexpr size_t L2_ATOMIC_ADD_STATUS_SIZE = 256;
constexpr size_t UINT64_SIZE = sizeof(uint64_t);
constexpr const char* DEBUG_FILE_SIGN = "Opdebug.Node_OpDebug.";

constexpr const char* STATS_HEADER_INOUT = "Input/Output";
constexpr const char* STATS_HEADER_ID = "Index";
constexpr const char* STATS_HEADER_DATA_SIZE = "Data Size";
constexpr const char* STATS_HEADER_DATA_TYPE = "Data Type";
constexpr const char* STATS_HEADER_FORMAT = "Format";
constexpr const char* STATS_HEADER_SHAPE = "Shape";
constexpr const char* STATS_HEADER_MAX = "Max Value";
constexpr const char* STATS_HEADER_MIN = "Min Value";
constexpr const char* STATS_HEADER_AVG = "Avg Value";
constexpr const char* STATS_HEADER_L2NORM = "l2norm";
constexpr const char* STATS_CSV_HEADER_L2NORM = "L2Norm Value";
constexpr const char* STATS_HEADER_MD5 = "MD5 Value";
constexpr const char* STATS_HEADER_NAN = "Nan Count";
constexpr const char* STATS_CSV_HEADER_NAN = "NaN Count";
constexpr const char* STATS_HEADER_NEG_INF = "Negative Inf Count";
constexpr const char* STATS_HEADER_POS_INF = "Positive Inf Count";
constexpr const char* RANK_ID = "RANK_ID";
constexpr const char* DIGITAL_NUMBERS = "0123456789";

static const std::map<DebuggerSummaryOption, std::pair<std::string, std::string>> SUMMARY_OPTION_HEADER_STR_MAP = {
    {DebuggerSummaryOption::MAX, {STATS_HEADER_MAX, STATS_HEADER_MAX}},
    {DebuggerSummaryOption::MIN, {STATS_HEADER_MIN, STATS_HEADER_MIN}},
    {DebuggerSummaryOption::MEAN, {STATS_HEADER_AVG, STATS_HEADER_AVG}},
    {DebuggerSummaryOption::L2NORM, {STATS_HEADER_L2NORM, STATS_CSV_HEADER_L2NORM}},
    {DebuggerSummaryOption::NAN_CNT, {STATS_HEADER_NAN, STATS_CSV_HEADER_NAN}},
    {DebuggerSummaryOption::NEG_INF_CNT, {STATS_HEADER_NEG_INF, STATS_HEADER_NEG_INF}},
    {DebuggerSummaryOption::POS_INF_CNT, {STATS_HEADER_POS_INF, STATS_HEADER_POS_INF}},
    {DebuggerSummaryOption::MD5, {STATS_HEADER_MD5, STATS_HEADER_MD5}},
};

const static std::map<AclDtype, AclDtype> kDtypeTransMap = {
    {AclDtype::DT_BF16, AclDtype::DT_FLOAT},
    {AclDtype::DT_INT4, AclDtype::DT_INT8},
};

class AclTensorStats {
public:
    AclTensorStats() = default;
    explicit AclTensorStats(const AclTensorInfo& tensor, const std::map<DebuggerSummaryOption, std::string>& summary);
    ~AclTensorStats() = default;

    std::string GetCsvHeader() const;
    std::string GetCsvValue() const;
    std::string GetPath() const {return path;}
    bool Empty() const {return stats.empty();};

    static AclTensorStats CalTensorSummary(const AclTensorInfo& tensor, const std::vector<DebuggerSummaryOption>& opt);
    static AclTensorStats ParseTensorSummary(const std::string& dumpPath, const std::string& input);

private:
    std::string path;
    std::string opType;
    std::string opName;
    std::string taskID;
    std::string streamID;
    std::string timestamp;
    std::string inout;
    std::string slot;
    std::string dataSize;
    std::string dataType;
    std::string format;
    std::string shape;
    std::map<DebuggerSummaryOption, std::string> stats;

    void ParseInfoFromDumpPath(const std::string& dumpPath);
    std::string& operator[](DebuggerSummaryOption opt) { return stats[opt]; }

    static constexpr const size_t BUFFER_LEN = 1024;
};

void AclTensorStats::ParseInfoFromDumpPath(const std::string& dumpPath)
{
    std::string filename;
    if (FileUtils::GetFileSuffix(dumpPath) == "csv") {
        filename = FileUtils::GetFileBaseName(dumpPath);
    } else {
        filename = FileUtils::GetFileName(dumpPath);
    }

    path = FileUtils::GetParentDir(dumpPath);
    std::vector<std::string> tokens = FileUtils::SplitPath(filename, '.');

    /* dump文件名格式：{optype}.{opname}.{taskid}.{streamid}.{timestamp} */
    if (tokens.size() < 5) {
        LOG_WARNING(DebuggerErrno::ERROR_INVALID_FORMAT, "Skip dumping invalid op " + filename);
        stats.clear();
        return;
    }

    opType = std::move(tokens[0]);
    opName = std::move(tokens[1]);
    taskID = std::move(tokens[2]);
    streamID = std::move(tokens[3]);
    timestamp = std::move(tokens[4]);
}

AclTensorStats::AclTensorStats(const AclTensorInfo& tensor, const std::map<DebuggerSummaryOption, std::string>& summary)
    : stats{summary}
{
    ParseInfoFromDumpPath(tensor.dumpPath);
    /* stats为空说明是header行，不需要落盘 */
    if (stats.empty()) {
        return;
    }
    inout = tensor.inout;
    slot = std::to_string(tensor.slot);
    dataSize = std::to_string(tensor.dataSize);
    dataType = DataUtils::GetDTypeString(tensor.dtype);
    format = DataUtils::GetFormatString(tensor.hostFmt);
    shape = DataUtils::GetShapeString(tensor.hostShape);
}

AclTensorStats AclTensorStats::CalTensorSummary(const AclTensorInfo& tensor,
    const std::vector<DebuggerSummaryOption>& opt)
{
    DEBUG_FUNC_TRACE();
    std::map<DebuggerSummaryOption, std::string> summary;
    if (ELE_IN_VECTOR(opt, DebuggerSummaryOption::MD5)) {
        const uint8_t* data = tensor.transBuf.empty() ?  tensor.aclData : tensor.transBuf.data();
        summary[DebuggerSummaryOption::MD5] = MathUtils::CalculateMD5(data, tensor.dataSize);
    }

    return AclTensorStats(tensor, summary);
}

static std::map<uint32_t, DebuggerSummaryOption> ParseTensorSummaryHeaderOrder(const std::vector<std::string>& segs)
{
    std::map<uint32_t, DebuggerSummaryOption> ret;
    for (size_t pos = 0; pos < segs.size(); ++pos) {
        const std::string& opt = segs[pos];
        for (auto it = SUMMARY_OPTION_HEADER_STR_MAP.begin(); it != SUMMARY_OPTION_HEADER_STR_MAP.end(); ++it) {
            if (opt == it->second.first) {
                ret[pos] = it->first;
                break;
            }
        }
    }
    return ret;
}

AclTensorStats AclTensorStats::ParseTensorSummary(const std::string& dumpPath, const std::string& input)
{
    constexpr const size_t optPosBase = 7;
    static std::map<uint32_t, DebuggerSummaryOption> order;
    static uint32_t headerLen = 0;

    std::vector<std::string> segs = FileUtils::SplitPath(input, ',');
    /* device计算统计量场景，各个kernel的统计项的顺序是相同的，只要计算一次即可 */
    if (order.empty()) {
        if (segs.size() <= optPosBase || segs[0] != STATS_HEADER_INOUT) {
            LOG_WARNING(DebuggerErrno::ERROR_INVALID_FORMAT, "Summary data miss header, some data may lose.");
            return AclTensorStats();
        }
        headerLen = segs.size();
        order = ParseTensorSummaryHeaderOrder(segs);

        return AclTensorStats();
    }

    if (segs.size() < headerLen) {
        LOG_WARNING(DebuggerErrno::ERROR_INVALID_FORMAT, "Summary data miss some fields, some data may lose.");
        return AclTensorStats();
    }

    /* 不重复解析header行 */
    if (segs[0] == STATS_HEADER_INOUT) {
        return AclTensorStats();
    }

    /* device侧计算统计量格式：Input/Output,Index,Data Size,Data Type,Format,Shape,Count,...(统计量) */
    AclTensorStats stat = AclTensorStats();
    stat.ParseInfoFromDumpPath(dumpPath);
    stat.inout = segs[0];
    stat.slot = segs[1];
    stat.dataSize = segs[2];
    stat.dataType = segs[3];
    stat.format = segs[4];
    stat.shape = segs[5];
    for (auto it = order.begin(); it != order.end(); ++it) {
        stat[it->second] = segs[it->first];
    }
    return stat;
}

std::string AclTensorStats::GetCsvHeader() const
{
    if (stats.empty()) {
        return std::string();
    }
    std::string ret;
    ret.reserve(BUFFER_LEN);
    ret.append("Op Type,Op Name,Task ID,Stream ID,Timestamp,Input/Output,Slot,Data Size,Data Type,Format,Shape");
    for (auto it = stats.begin(); it != stats.end(); it++) {
        ret.append(",");
        ret.append(SUMMARY_OPTION_HEADER_STR_MAP.at(it->first).second);
    }
    ret.append("\n");

    return ret;
}

std::string AclTensorStats::GetCsvValue() const
{
    if (stats.empty()) {
        return std::string();
    }

    std::string ret;
    ret.reserve(BUFFER_LEN);
    ret.append(opType).append(",").append(opName).append(",").append(taskID).append(",").append(streamID).append(",") \
       .append(timestamp).append(",").append(inout).append(",").append(slot).append(",") .append(dataSize) \
       .append(",").append(dataType).append(",").append(format).append(",").append(shape);
    /* map会根据键值自动排序，此处可以保障头和值的顺序，直接追加写即可 */
    for (auto it = stats.begin(); it != stats.end(); it++) {
        ret.append(",");
        ret.append(it->second);
    }
    ret.append("\n");

    return ret;
}

AclDumpDataProcessor::~AclDumpDataProcessor()
{
    while (!buffer.empty()) {
        delete buffer.front();
        buffer.pop();
    }
}

std::string AclDumpDataProcessor::ToString() const
{
    return "AclDumpDataProcessor(path=" + dumpPath + ",completed=" + std::to_string(completed) + ",len=" +
           std::to_string(totalLen) + ")";
}

DebuggerErrno AclDumpDataProcessor::PushData(const AclDumpChunk *chunk)
{
    DEBUG_FUNC_TRACE();
    if (completed) {
        LOG_WARNING(DebuggerErrno::ERROR_INVALID_OPERATION,
                    ToString() + " receive data when completed. Some errors may occur.");
        return DebuggerErrno::ERROR_INVALID_OPERATION;
    }

    /* 防止最后一包处理出错导致processor残留，此处先设置完成标记位 */
    if (chunk->isLastChunk) {
        completed = true;
    }

    size_t len = chunk->bufLen;
    if (len == 0) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_VALUE, ToString() + ": invalid value(cached size " +
                  std::to_string(totalLen) + ", receiving size " + std::to_string(len) + ").");
        errorOccurred = true;
        return DebuggerErrno::ERROR_INVALID_VALUE;
    }

    /* 防止正负翻转 */
    if (SIZE_MAX - len < totalLen || totalLen + len > MAX_DATA_LEN) {
        LOG_ERROR(DebuggerErrno::ERROR_BUFFER_OVERFLOW, ToString() + ": buffer overflow(cached size " +
                  std::to_string(totalLen) + ", receiving size " + std::to_string(len) + ").");
        errorOccurred = true;
        return DebuggerErrno::ERROR_BUFFER_OVERFLOW;
    }

    std::vector<uint8_t> *p = new (std::nothrow) std::vector<uint8_t>(len);
    if (p == nullptr) {
        LOG_ERROR(DebuggerErrno::ERROR_NO_MEMORY, "Acl dump data processor(" +  dumpPath + "): Alloc failed(" +
                  std::to_string(len) + " bytes).");
        errorOccurred = true;
        return DebuggerErrno::ERROR_NO_MEMORY;
    }

    /* vector p根据chunk->dataBuf的长度，即len，申请创建，所以无需校验空间大小 */
    try {
        std::copy(chunk->dataBuf, chunk->dataBuf + len, p->begin());
    } catch (const std::exception& e) {
        LOG_ERROR(DebuggerErrno::ERROR_SYSCALL_FAILED, ToString() + ": Failed to copy data;");
        delete p;
        errorOccurred = true;
        return DebuggerErrno::ERROR_SYSCALL_FAILED;
    }

    buffer.push(p);
    totalLen += len;
    if (!chunk->isLastChunk) { return DebuggerErrno::OK; }

    completed = true;
    DebuggerErrno ret = ConcatenateData();
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Acl dump data processor(" +  dumpPath + "): Failed to concatenate data.");
        errorOccurred = true;
        return ret;
    }
    LOG_DEBUG(ToString() + " is completed.");

    return DebuggerErrno::OK;
}

DebuggerErrno AclDumpDataProcessor::ConcatenateData()
{
    DEBUG_FUNC_TRACE();
    if (!completed) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_OPERATION, "Data processor(" +  dumpPath + "): Data is incomplete.");
        return DebuggerErrno::ERROR_INVALID_OPERATION;
    }

    if (buffer.empty()) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_VALUE, "Data processor(" +  dumpPath + "): No data.");
        return DebuggerErrno::ERROR_INVALID_VALUE;
    }

    /* 为了减少数据重复拷贝，此处只整合一次，不再剥数据头，用偏移来取数据段 */
    if (buffer.size() > 1) {
        std::vector<uint8_t> *p = new (std::nothrow) std::vector<uint8_t>(totalLen);
        if (p == nullptr) {
            LOG_ERROR(DebuggerErrno::ERROR_NO_MEMORY, "Alloc failed(" + std::to_string(totalLen) + ").");
            return DebuggerErrno::ERROR_NO_MEMORY;
        }

        size_t offset = 0;
        while (!buffer.empty()) {
            /* vector p根据buffer里所有vector的总长度，即totalLen，申请创建，所以无需校验空间大小 */
            try {
                std::copy(buffer.front()->begin(), buffer.front()->end(), p->begin() + offset);
            } catch (const std::exception& e) {
                delete p;
                LOG_ERROR(DebuggerErrno::ERROR_SYSCALL_FAILED, "Data processor(" +  dumpPath + "): Failed to copy.");
                return DebuggerErrno::ERROR_SYSCALL_FAILED;
            }
            offset += buffer.front()->size();
            delete buffer.front();
            buffer.pop();
        }
        buffer.push(p);
    }

    if (FileUtils::GetFileSuffix(dumpPath) == CSV_SUFFIX) {
        dataSegOffset = 0;
        dataSegLen = totalLen;
        return DebuggerErrno::OK;
    }

    headerSegOffset = sizeof(uint64_t);
    if (totalLen < headerSegOffset) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT, "Acl dump data processor(" +  dumpPath +
                  "): Invalid data length " + std::to_string(totalLen) + ".");
        return DebuggerErrno::ERROR_INVALID_FORMAT;
    }

    headerSegLen = *(reinterpret_cast<const uint64_t *>(buffer.front()->data()));
    if (totalLen < headerSegOffset + headerSegLen) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT, "Acl dump data processor(" +  dumpPath +
                  "): Invalid header len " + std::to_string(headerSegLen) + "/" + std::to_string(totalLen) + ".");
        return DebuggerErrno::ERROR_INVALID_FORMAT;
    }

    dataSegOffset = headerSegOffset + headerSegLen;
    dataSegLen = totalLen - dataSegOffset;
    return DebuggerErrno::OK;
}

static nlohmann::json ParseOverflowInfo(const uint8_t* data)
{
    DEBUG_FUNC_TRACE();
    uint32_t index = 0;
    nlohmann::json overflowInfo;
    uint64_t modelId = DataUtils::UnpackUint64ValueLe(data);
    index += UINT64_SIZE;
    uint64_t streamId = DataUtils::UnpackUint64ValueLe(data + index);
    index += UINT64_SIZE;
    uint64_t taskId = DataUtils::UnpackUint64ValueLe(data + index);
    index += UINT64_SIZE;
    uint64_t taskType = DataUtils::UnpackUint64ValueLe(data + index);
    index += UINT64_SIZE;
    uint64_t pcStart = DataUtils::UnpackUint64ValueLe(data + index);
    index += UINT64_SIZE;
    uint64_t paraBase = DataUtils::UnpackUint64ValueLe(data + index);

    overflowInfo["model_id"] = modelId;
    overflowInfo["stream_id"] = streamId;
    overflowInfo["task_id"] = taskId;
    overflowInfo["task_type"] = taskType;
    overflowInfo["pc_start"] = DataUtils::U64ToHexString(pcStart);
    overflowInfo["para_base"] = DataUtils::U64ToHexString(paraBase);
    return overflowInfo;
}

static DebuggerErrno DumpOpDebugDataToDisk(const std::string& dumpPath, AclDumpMsg::DumpData& dumpData,
                                           const uint8_t* data, size_t dataLen)
{
    DEBUG_FUNC_TRACE();
    std::string outPath = dumpPath + ".output.";
    uint32_t num = static_cast<uint32_t>(dumpData.output().size());
    for (uint32_t slot = 0; slot < num; slot++) {
        uint32_t offset = 0;
        // parse DHA Atomic Add info
        nlohmann::json dhaAtomicAddInfo = ParseOverflowInfo(data + offset);
        offset += DHA_ATOMIC_ADD_INFO_SIZE;
        // parse L2 Atomic Add info
        nlohmann::json l2AtomicAddInfo = ParseOverflowInfo(data + offset);
        offset += L2_ATOMIC_ADD_INFO_SIZE;
        // parse AICore info
        nlohmann::json aiCoreInfo = ParseOverflowInfo(data + offset);
        offset += AICORE_INFO_SIZE;
        // parse DHA Atomic Add status
        dhaAtomicAddInfo["status"] = DataUtils::UnpackUint64ValueLe(data + offset);
        offset += DHA_ATOMIC_ADD_STATUS_SIZE;
        // parse L2 Atomic Add status
        l2AtomicAddInfo["status"] = DataUtils::UnpackUint64ValueLe(data + offset);
        offset += L2_ATOMIC_ADD_STATUS_SIZE;
        // parse AICore status
        uint64_t kernelCode = DataUtils::UnpackUint64ValueLe(data + offset);
        offset += UINT64_SIZE;
        uint64_t blockIdx = DataUtils::UnpackUint64ValueLe(data + offset);
        offset += UINT64_SIZE;
        uint64_t status = DataUtils::UnpackUint64ValueLe(data + offset);
        aiCoreInfo["kernel_code"] = DataUtils::U64ToHexString(kernelCode);
        aiCoreInfo["block_idx"] = blockIdx;
        aiCoreInfo["status"] = status;

        nlohmann::json opdebugData;
        opdebugData["DHA Atomic Add"] = dhaAtomicAddInfo;
        opdebugData["L2 Atomic Add"] = l2AtomicAddInfo;
        opdebugData["AI Core"] = aiCoreInfo;

        // save json to file
        std::string filePath = outPath + std::to_string(slot) + "." + JSON_SUFFIX;
        DebuggerErrno ret = FileOperation::DumpJson(filePath, opdebugData);
        if (ret != DebuggerErrno::OK) {
            LOG_ERROR(ret, "Failed to dump data to " + filePath + ".");
            return ret;
        }
    }
    return DebuggerErrno::OK;
}

static DebuggerErrno ConvertFormatDeviceToHost(AclTensorInfo& tensor)
{
    DEBUG_FUNC_TRACE();
    if (tensor.deviceFmt == tensor.hostFmt || AclTensor::SizeOfTensor(tensor) == 0) {
        LOG_DEBUG(tensor + ": No need to convert format.");
        return DebuggerErrno::OK;
    }

    DebuggerErrno ret = AclTensor::TransFormatD2H(tensor);
    if (ret == DebuggerErrno::ERROR_UNKNOWN_TRANS) {
        LOG_INFO("Do not support convert format from " +
                    std::to_string(tensor.deviceFmt) + " to " + std::to_string(tensor.hostFmt) + ".");
        tensor.hostFmt = tensor.deviceFmt;
        return DebuggerErrno::OK;
    }

    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, tensor + ": Failed to convert format.");
        return ret;
    }

    LOG_DEBUG(tensor + ": Convert format successfully.");
    return DebuggerErrno::OK;
}

static std::string MappingFilePath(const std::string& originPath)
{
    /* adump一次最多传10个tensor数据，输入输出数超过10的算子会分包，但是时序上是连续的，此处缓存上一次的映射 */
    static std::string lastOriName;
    static std::string lastMappingPath;

    if (lastOriName == originPath && !lastMappingPath.empty()) {
        return lastMappingPath;
    }

    std::string dir = FileUtils::GetParentDir(originPath);
    std::string suffix = FileUtils::GetFileSuffix(originPath);
    std::string mappingName;
    uint32_t retry = 10;
    constexpr uint32_t randFileNameLen = 32;
    do {
        mappingName = MathUtils::RandomString(randFileNameLen, '0', '9');
        if (!suffix.empty()) {
            mappingName.append(".").append(suffix);
        }
        if (!FileUtils::IsPathExist(dir + "/" + mappingName)) {
            break;
        }
    } while (--retry);

    if (retry == 0) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to map path " + originPath + ".");
        return std::string();
    }

    DebuggerErrno ret = FileUtils::CreateDir(dir);
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to create directory " + dir + ".");
        return std::string();
    }
    std::ofstream ofs;
    constexpr const char* mapFileName = "mapping.csv";

    ret = FileUtils::OpenFile(dir + "/" + mapFileName, ofs, std::ofstream::app);
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to open mapping file " + dir + "/" + mapFileName + ".");
        return std::string();
    }

    ofs << mappingName << "," << FileUtils::GetFileName(originPath) << "\n";
    if (ofs.fail()) {
        LOG_ERROR(DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE, "Failed to write file " + dir + "/" + mapFileName + ".");
        ofs.close();
        return std::string();
    }
    ofs.close();
    lastOriName = originPath;
    lastMappingPath = dir + "/" + mappingName;
    return lastMappingPath;
}

static DebuggerErrno StandardizedDumpPath(std::string& originPath)
{
    std::string filename = FileUtils::GetFileName(originPath);
    if (filename.length() <= FileUtils::FILE_NAME_MAX) {
        return DebuggerErrno::OK;
    }

    std::string mappingPath = MappingFilePath(originPath);
    if (mappingPath.empty()) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to open mapping file " + originPath + ".");
        return DebuggerErrno::ERROR;
    }

    originPath = std::move(mappingPath);
    return DebuggerErrno::OK;
}

static std::string GenDataPath(const std::string& path)
{
    LOG_DEBUG("Original acl data path is " + path);
    std::string outputPath = DebuggerConfig::GetInstance().GetOutputPath();
    std::string dataPath;
    if (path.compare(0, outputPath.length(), outputPath) != 0) {
        return path;
    }
    dataPath = path.substr(outputPath.length());
    const std::vector<std::string> items = FileUtils::SplitPath(dataPath);
    constexpr const size_t expectSegLen = 9;
    constexpr const size_t rankIdPos = 0;
    constexpr const size_t timeStampPos = 1;
    constexpr const size_t stepIdPos = 2;
    constexpr const size_t dataNamePos = 8;

    if (items.size() >= expectSegLen) {
        dataPath = outputPath;
        if (dataPath.at(dataPath.length() - 1) != '/') {
            dataPath.append("/");
        }
        /*
        * ACL 接口返回数据的路径格式如下
        * {dump_path}/rank_{rank_id}/{time stamp}/step_{step_id}/{time}
        /{device_id}/{model_name}/{model_id}/{iteration_id}/{data name}
        * items[0] 表示 rank_{rank_id}
        * items[1] 表示 {time stamp}
        * items[2] 表示 step_{step_id}
        * items[8] 表示 {data name}
        */
        dataPath.append(items[rankIdPos] + "/");
        dataPath.append(items[timeStampPos] + "/");
        dataPath.append(items[stepIdPos] + "/");
        dataPath.append(items[dataNamePos]);
        return dataPath;
    }
    return path;
}

inline std::string GetTensorInfoSuffix(AclTensorInfo& tensor)
{
    return "." + tensor.inout + "." + std::to_string(tensor.slot) +
           "." + DataUtils::GetFormatString(tensor.hostFmt) + "." + DataUtils::GetDTypeString(tensor.oriDtype);
}

static DebuggerErrno DumpOneAclTensorFmtBin(AclTensorInfo& tensor)
{
    DebuggerErrno ret;
    std::string dumpPathSlot = tensor.dumpPath + GetTensorInfoSuffix(tensor);
    if (StandardizedDumpPath(dumpPathSlot) != DebuggerErrno::OK) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to standardize path " + dumpPathSlot + ".");
        return DebuggerErrno::ERROR;
    }

    std::ofstream ofs;
    ret = FileUtils::OpenFile(dumpPathSlot, ofs, std::ios::out | std::ios::binary);
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Failed to open file " + dumpPathSlot + ".");
        return ret;
    }

    ofs.write(reinterpret_cast<const char*>(tensor.aclData), tensor.dataSize);
    if (ofs.fail()) {
        LOG_ERROR(DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE, "Failed to write file " + dumpPathSlot + ".");
        ret = DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE;
    }
    ofs.close();
    return ret;
}

static DebuggerErrno DumpOneAclTensorFmtNpy(AclTensorInfo& tensor)
{
    DEBUG_FUNC_TRACE();
    DebuggerErrno ret;
    if (tensor.dataSize == 0) {
        LOG_INFO(tensor + ": Data size is 0. No need to dump.");
        return DebuggerErrno::OK;
    }

    auto it = kDtypeTransMap.find(tensor.dtype);
    if (it != kDtypeTransMap.end()) {
        AclDtype dstDtype = it->second;
        ret = AclTensor::TransDtype(tensor, dstDtype);
        if (ret != DebuggerErrno::OK) {
            LOG_ERROR(ret, tensor + ": Failed to transform dtype from " +
                        DataUtils::GetDTypeString(it->first) + " to " +
                        DataUtils::GetDTypeString(it->second)+ ".");
            return ret;
        }
    }

    // dump_path: dump_dir/op_type.op_name.task_id.stream_id.timestamp
    std::string dumpPathSlot = tensor.dumpPath + GetTensorInfoSuffix(tensor) +  "." + NPY_SUFFIX;
    if (StandardizedDumpPath(dumpPathSlot) != DebuggerErrno::OK) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to standardize path " + dumpPathSlot + ".");
        return DebuggerErrno::ERROR;
    }

    if (tensor.transBuf.empty()) {
        ret = FileOperation::DumpNpy(dumpPathSlot, tensor.aclData, tensor.dataSize, tensor.dtype, tensor.hostShape);
    } else {
        ret = FileOperation::DumpNpy(dumpPathSlot, tensor.transBuf.data(), tensor.transBuf.size(), tensor.dtype,
                                     tensor.hostShape);
    }

    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, tensor + ": Failed to dump as npy.");
        return ret;
    }

    LOG_DEBUG(tensor + ": dump successfully.");

    return ret;
}

static DebuggerErrno WriteOneTensorStatToDisk(const AclTensorStats& stat)
{
    DEBUG_FUNC_TRACE();
    if (stat.Empty()) {
        return DebuggerErrno::OK;
    }

    std::string dumpfile = stat.GetPath() + "/statistic.csv";
    /* 此处防止多进程间竞争，使用文件锁，故使用C风格接口 */
    uint32_t retry = 100;
    uint32_t interval = 10;
    if (FileUtils::CheckFileBeforeCreateOrWrite(dumpfile, true) != DebuggerErrno::OK) {
        LOG_ERROR(DebuggerErrno::ERROR_FILE_ALREADY_EXISTS, "File " + dumpfile + " exists and has invalid format.");
        return DebuggerErrno::ERROR_FILE_ALREADY_EXISTS;
    }

    int fd = open(dumpfile.c_str(), O_WRONLY | O_CREAT | O_APPEND, NORMAL_FILE_MODE_DEFAULT);
    if (fd < 0) {
        LOG_ERROR(DebuggerErrno::ERROR_FAILED_TO_OPEN_FILE, "Failed to open file " + dumpfile);
        return DebuggerErrno::ERROR_FAILED_TO_OPEN_FILE;
    }

    uint32_t i;
    for (i = 0; i < retry; ++i) {
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    }

    if (i == retry) {
        LOG_ERROR(DebuggerErrno::ERROR_SYSCALL_FAILED, "Failed to occupy file " + dumpfile);
        close(fd);
        return DebuggerErrno::ERROR_SYSCALL_FAILED;
    }

    /* 防止等待文件锁的期间又有别的进程写入内容，重新查找文件尾 */
    off_t offset = lseek(fd, 0, SEEK_END);
    if (offset == 0) {
        std::string header = stat.GetCsvHeader();
        if (write(fd, header.c_str(), header.length()) < static_cast<ssize_t>(header.length())) {
            LOG_ERROR(DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE, "Failed to write file " + dumpfile);
            flock(fd, LOCK_UN);
            close(fd);
            return DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE;
        }
    }

    std::string value = stat.GetCsvValue();
    DebuggerErrno ret = DebuggerErrno::OK;
    if (write(fd, value.c_str(), value.length()) < static_cast<ssize_t>(value.length())) {
        LOG_ERROR(DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE, "Failed to write file " + dumpfile);
        ret = DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE;
    }

    flock(fd, LOCK_UN);
    close(fd);
    return ret;
}

static DebuggerErrno DumpOneAclTensor(AclTensorInfo& tensor, std::vector<DebuggerSummaryOption>& opt)
{
    DEBUG_FUNC_TRACE();
    if (tensor.dumpOriginData || !FileOperation::IsDtypeSupportByNpy(tensor.dtype)) {
        if (kDtypeTransMap.find(tensor.dtype) == kDtypeTransMap.end()) {
            return DumpOneAclTensorFmtBin(tensor);
        }
    }

    DebuggerErrno ret = ConvertFormatDeviceToHost(tensor);
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, tensor + ": Failed to convert format to host.");
        return ret;
    }

    if (!opt.empty()) {
        AclTensorStats stat = AclTensorStats::CalTensorSummary(tensor, opt);
        return WriteOneTensorStatToDisk(stat);
    }

    return DumpOneAclTensorFmtNpy(tensor);
}

static void DumpAclTensor(std::vector<AclTensorInfo>::iterator begin, std::vector<AclTensorInfo>::iterator end,
                          std::vector<DebuggerSummaryOption> opt)
{
    DEBUG_FUNC_TRACE();
    DebuggerErrno ret = DebuggerErrno::OK;
    for (auto it = begin; it != end; it++) {
        ret = DumpOneAclTensor(*it, opt);
        if (ret != DebuggerErrno::OK) {
            LOG_WARNING(ret, *it + ": Failed to dump to disk.");
            break;
        }
    }
    return;
}

static DebuggerErrno DumpTensorDataToDisk(const std::string& dumpPath, AclDumpMsg::DumpData& dumpData,
                                          const uint8_t* data, size_t dataLen, std::vector<DebuggerSummaryOption>& opt)
{
    DEBUG_FUNC_TRACE();
    std::vector<AclTensorInfo> aclTensorInfos;
    uint64_t offset = 0;
    uint32_t slot = 0;
    for (auto& tensor : dumpData.input()) {
        aclTensorInfos.push_back(AclTensor::ParseAttrsFromDumpData(dumpPath, data + offset, tensor, "input", slot));
        offset += tensor.size();
        slot++;
    }

    slot = 0;
    for (auto& tensor : dumpData.output()) {
        aclTensorInfos.push_back(AclTensor::ParseAttrsFromDumpData(dumpPath, data + offset, tensor, "output", slot));
        offset += tensor.size();
        slot++;
    }

    if (aclTensorInfos.empty()) {
        return DebuggerErrno::OK;
    }

    if (offset > dataLen) {
        LOG_ERROR(DebuggerErrno::ERROR_VALUE_OVERFLOW, dumpPath + ": offset overflow " + std::to_string(offset) + "/" +
                  std::to_string(dataLen) + ".");
        return DebuggerErrno::ERROR_VALUE_OVERFLOW;
    }

    /* 根据tensor的数据量，1MB以下串行，1MB以上多线程并发，最大并发量为 最大线程数/4 */
    constexpr int kMaxTensorSize = 1024 * 1024;
    if (offset < kMaxTensorSize) {
        DumpAclTensor(aclTensorInfos.begin(), aclTensorInfos.end(), opt);
    } else {
        size_t concurrent = std::max<size_t>(1, std::thread::hardware_concurrency() / 4);
        concurrent = std::min(concurrent, aclTensorInfos.size());
        size_t total = aclTensorInfos.size();
        size_t batch = MathUtils::DivCeil(total, concurrent);
        size_t cur = 0;
        std::vector<std::thread> threads;
        std::vector<AclTensorInfo>::iterator begin = aclTensorInfos.begin();

        threads.reserve(concurrent);
        while (cur < total) {
            threads.emplace_back(std::thread(&DumpAclTensor, begin + cur, begin + std::min(total, cur + batch), opt));
            cur += batch;
        }

        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    DebuggerErrLevel err = ErrorInfosManager::GetTopErrLevelInDuration();
    return err >= DebuggerErrLevel::LEVEL_ERROR ? DebuggerErrno::ERROR : DebuggerErrno::OK;
}

static DebuggerErrno DumpStatsDataToDisk(const std::string& dumpPath, const uint8_t* data, size_t dataLen)
{
    DEBUG_FUNC_TRACE();
    constexpr const size_t maxDataSize = 10 * 1024 * 1024;

    if (dataLen > maxDataSize) {
        LOG_ERROR(DebuggerErrno::ERROR_FILE_TOO_LARGE, "File " + dumpPath + " is too large to be dumped.");
        return DebuggerErrno::ERROR_FILE_TOO_LARGE;
    }

    std::string content(reinterpret_cast<const char*>(data), dataLen);
    std::vector<std::string> lines = FileUtils::SplitPath(content, '\n');
    DebuggerErrno ret;
    for (const auto& line : lines) {
        if (line.empty() || line[0] == '\0') {
            continue;
        }
        AclTensorStats stat = AclTensorStats::ParseTensorSummary(dumpPath, line);
        ret = WriteOneTensorStatToDisk(stat);
        if (ret != DebuggerErrno::OK) {
            return ret;
        }
    }

    return DebuggerErrno::OK;
}

DebuggerErrno AclDumpDataProcessor::DumpToDisk()
{
    DEBUG_FUNC_TRACE();
    if (!completed) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_OPERATION, ToString() + ": Data is incomplete.");
        return DebuggerErrno::ERROR_INVALID_OPERATION;
    }

    uint8_t* msg = buffer.front()->data();
    AclDumpMsg::DumpData dumpData;
    if (headerSegLen > 0) {
        if (!dumpData.ParseFromArray(msg + headerSegOffset, headerSegLen)) {
            LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT, ToString() + ": Failed to parse header.");
            return DebuggerErrno::ERROR_INVALID_FORMAT;
        }
    }

    const std::string dataPath = GenDataPath(dumpPath);
    DebuggerErrno ret;
    if (FileUtils::GetFileName(dumpPath).find(DEBUG_FILE_SIGN) == 0 &&
        DebuggerConfig::GetInstance().GetOverflowCheckCfg() != nullptr) {
        ret = DumpOpDebugDataToDisk(dataPath, dumpData, msg + dataSegOffset, dataSegLen);
    } else if (DebuggerConfig::GetInstance().GetStatisticsCfg() != nullptr &&
               hostAnalysisOpts.empty()) {
        ret = DumpStatsDataToDisk(dataPath, msg + dataSegOffset, dataSegLen);
    } else {
        ret = DumpTensorDataToDisk(dataPath, dumpData, msg + dataSegOffset, dataSegLen, hostAnalysisOpts);
    }

    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(DebuggerErrno::ERROR_OPERATION_FAILED, ToString() + ": Failed to dump to disk.");
    }

    return ret;
}

}
