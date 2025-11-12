/*
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <map>
#include <climits>
#include <ctime>
#include <cstdlib>

#include "include/Macro.h"
#include "utils/FileUtils.h"
#include "utils/FileOperation.h"
#include "third_party/ACL/AclApi.h"
#include "base/Environment.h"
#include "base/ErrorInfosManager.h"
#include "AclDumper.h"

namespace MindStudioDebugger {

constexpr const char* ACL_DUMP_SCENE = "dump_scene";
constexpr const char* SCENE_NORMAL = "normal";
constexpr const char* SCENE_EXCEPTION = "lite_exception";

constexpr const char* ACL_DUMP_PATH = "dump_path";
constexpr const char* ACL_DUMP_STEP = "dump_step";

constexpr const char* ACL_DUMP_LIST = "dump_list";
constexpr const char* ACL_DUMP_LAYER = "layer";
constexpr const char* ACL_DUMP_MODEL_NAME = "model_name";

constexpr const char* ACL_DUMP_MODE = "dump_mode";
constexpr const char* ACL_MODE_INPUT = "input";
constexpr const char* ACL_MODE_OUTPUT = "output";
constexpr const char* ACL_MODE_ALL = "all";

constexpr const char* DUMP_OP_SWITCH = "dump_op_switch";
constexpr const char* ACL_DUMP_DEBUG = "dump_debug";
constexpr const char* ACL_SWITCH_ON = "on";
constexpr const char* ACL_SWITCH_OFF = "off";

constexpr const char* ACL_DUMP_DATA = "dump_data";
constexpr const char* ACL_DUMP_TENSOR = "tensor";
constexpr const char* ACL_DUMP_STATS = "stats";

constexpr const char* ACL_DUMP_STATS_OPT = "dump_stats";
constexpr const char* ACL_DUMP_STATS_MAX = "Max";
constexpr const char* ACL_DUMP_STATS_MIN = "Min";
constexpr const char* ACL_DUMP_STATS_AVG = "Avg";
constexpr const char* ACL_DUMP_STATS_NORM = "L2norm";
constexpr const char* ACL_DUMP_STATS_NAN = "Nan";
constexpr const char* ACL_DUMP_STATS_NEG_INF = "Negative Inf";
constexpr const char* ACL_DUMP_STATS_POS_INF = "Positive Inf";

constexpr const size_t PROCESSOR_NUM_MAX = 100;

inline std::string GenAclJsonPath(const std::string& dumpPath, uint32_t rank)
{
    return std::move(dumpPath + "/acl_dump_" + std::to_string(rank) + "." + JSON_SUFFIX);
}

/* 这里几个转换函数，映射和DebuggerConfigFieldMap类似，但是此处是对接ACL规则的，本质上不是一回事，因此单写一套 */
static std::string GenDumpInoutString(DebuggerDataInOut mode)
{
    static std::map<DebuggerDataInOut, std::string> dumpModeMap = {
        {DebuggerDataInOut::INOUT_INPUT, ACL_MODE_INPUT},
        {DebuggerDataInOut::INOUT_OUTPUT, ACL_MODE_OUTPUT},
        {DebuggerDataInOut::INOUT_BOTH, ACL_MODE_ALL},
    };

    auto it = dumpModeMap.find(mode);
    if (it == dumpModeMap.end()) {
        return ACL_MODE_ALL;
    } else {
        return it->second;
    }
}

static std::vector<std::string> GenStatsOptions(const std::vector<DebuggerSummaryOption>& options)
{
    static std::map<DebuggerSummaryOption, std::string> summaryOptMap = {
        {DebuggerSummaryOption::MAX, ACL_DUMP_STATS_MAX},
        {DebuggerSummaryOption::MIN, ACL_DUMP_STATS_MIN},
        {DebuggerSummaryOption::MEAN, ACL_DUMP_STATS_AVG},
        {DebuggerSummaryOption::L2NORM, ACL_DUMP_STATS_NORM},
        {DebuggerSummaryOption::NAN_CNT, ACL_DUMP_STATS_NAN},
        {DebuggerSummaryOption::NEG_INF_CNT, ACL_DUMP_STATS_NEG_INF},
        {DebuggerSummaryOption::POS_INF_CNT, ACL_DUMP_STATS_POS_INF},
    };

    std::vector<std::string> output;
    for (auto& ele : options) {
        auto it = summaryOptMap.find(ele);
        if (it != summaryOptMap.end()) {
            output.emplace_back(it->second);
        }
    }
    return output;
}

static std::string GenDumpPath(const std::string& path)
{
    std::string timestamp;
    std::string dumpPath;

    time_t pTime;
    time (&pTime);
    char cTime[15];
    strftime(cTime, sizeof(cTime), "%Y%m%d%H%M%S", localtime(&pTime));
    timestamp = cTime;

    int32_t rankId = Environment::GetRankID();
    if (rankId < 0) {
        rankId = 0;
    }

    dumpPath = path + "/rank_" + std::to_string(rankId) + "/" + timestamp;
    return dumpPath;
}

bool AclDumper::IsIterNeedDump(uint32_t iterId)
{
    const DebuggerConfig& cfg = DebuggerConfig::GetInstance();
    if (!cfg.IsCfgLoaded()) {
        return false;
    }

    return cfg.IsStepHits(iterId);
}

bool AclDumper::IsCfgEnableAclDumper()
{
    DebuggerConfig& cfg = DebuggerConfig::GetInstance();
    if (!cfg.IsCfgLoaded() || cfg.GetDebugLevel() != DebuggerLevel::L2) {
        return false;
    }
    const std::vector<DebuggerTaskType>& tasks = cfg.GetTaskList();
    return (ELE_IN_VECTOR(tasks, DebuggerTaskType::TASK_DUMP_TENSOR) ||
            ELE_IN_VECTOR(tasks, DebuggerTaskType::TASK_DUMP_STATISTICS) ||
            ELE_IN_VECTOR(tasks, DebuggerTaskType::TASK_OVERFLOW_CHECK));
}

bool AclDumper::IsOverflowCompleted()
{
    return overflowNums != -1 && realOverflowNums > overflowNums;
}

void AclDumper::CountOverflowNumbers(const AclDumpChunk* chunk)
{
    if (IsOverflowCompleted() || !isOverflowDump || !chunk->isLastChunk) {
        return;
    }
    const std::string fileName = chunk->fileName;
    auto separator = fileName.rfind("/");
    auto fileBaseName = fileName.substr(separator + 1);
    if (fileBaseName.rfind("Opdebug.Node_OpDebug.") == 0) {
        // count according to the first file: Node_OpDebug
        realOverflowNums++;
    }
    return;
}

std::string AclDumper::GetDumpPath(uint32_t curStep) const
{
    if (!initialized || foreDumpPath.empty()) {
        return "";
    }
    return foreDumpPath + "/step_" + std::to_string(curStep);
}

DebuggerErrno AclDumper::AclDumpGenTensorJson(std::shared_ptr<const DumpTensorCfg> dumpTensorCfg, uint32_t rank,
                                              uint32_t curStep, const char** kernels)
{
    DEBUG_FUNC_TRACE();
    nlohmann::json aclDumpJson;
    bool needDump = AclDumper::IsIterNeedDump(curStep);
    const std::string& dumpPath = DebuggerConfig::GetInstance().GetOutputPath();
    std::string fullDumpPath;
    if (needDump) {
        fullDumpPath = GetDumpPath(curStep);
        FileUtils::CreateDir(fullDumpPath, true);
    } else {
        fullDumpPath = dumpPath;
    }

    aclDumpJson[ACL_DUMP_PATH] = fullDumpPath;
    aclDumpJson[ACL_DUMP_MODE] = GenDumpInoutString(dumpTensorCfg->inout);
    aclDumpJson[ACL_DUMP_DATA] = ACL_DUMP_TENSOR;
    aclDumpJson[ACL_DUMP_LIST] = nlohmann::json::array();
    aclDumpJson[DUMP_OP_SWITCH] = ACL_SWITCH_ON;

    if (!needDump) {
        /* 这里沿用mindspore框架的方案，用一个大数0x7FFFFFFF表示不需要dump；这个方案非常奇怪，后续可以看下能否优化 */
        aclDumpJson[ACL_DUMP_STEP] = std::to_string(INT_MAX);
    } else {
        std::vector<std::string> kernelsList = dumpTensorCfg->matcher.GenRealKernelList(kernels);
        if (!kernelsList.empty()) {
            aclDumpJson[ACL_DUMP_LIST].push_back({{ACL_DUMP_LAYER, kernelsList}});
        }
    }

    nlohmann::json content = {{"dump", aclDumpJson}};
    LOG_DEBUG("AclDumpGenTensorJson dump json to " + GenAclJsonPath(dumpPath, rank));
    return FileOperation::DumpJson(GenAclJsonPath(dumpPath, rank), content);
}

DebuggerErrno AclDumper::AclDumpGenStatJson(std::shared_ptr<const StatisticsCfg> statisticsCfg, uint32_t rank,
                                            uint32_t curStep, const char** kernels)
{
    DEBUG_FUNC_TRACE();
    nlohmann::json aclDumpJson;
    bool needDump = AclDumper::IsIterNeedDump(curStep);
    const std::string& dumpPath = DebuggerConfig::GetInstance().GetOutputPath();
    std::string fullDumpPath;
    if (needDump) {
        fullDumpPath = GetDumpPath(curStep);
        FileUtils::CreateDir(fullDumpPath, true);
    } else {
        fullDumpPath = dumpPath;
    }

    aclDumpJson[ACL_DUMP_PATH] = fullDumpPath;
    aclDumpJson[ACL_DUMP_MODE] = GenDumpInoutString(statisticsCfg->inout);
    aclDumpJson[ACL_DUMP_LIST] = nlohmann::json::array();
    aclDumpJson[DUMP_OP_SWITCH] = ACL_SWITCH_ON;

    /* 如果需要host侧分析，下给acl的任务还是dump tensor，然后在host侧转成统计量 */
    if (!hostAnalysisOpt.empty()) {
        aclDumpJson[ACL_DUMP_DATA] = ACL_DUMP_TENSOR;
    } else {
        aclDumpJson[ACL_DUMP_DATA] = ACL_DUMP_STATS;
        aclDumpJson[ACL_DUMP_STATS_OPT] = GenStatsOptions(statisticsCfg->summaryOption);
    }

    if (!needDump) {
        aclDumpJson[ACL_DUMP_STEP] = std::to_string(INT_MAX);
    } else {
        std::vector<std::string> kernelsList = statisticsCfg->matcher.GenRealKernelList(kernels);
        if (!kernelsList.empty()) {
            aclDumpJson[ACL_DUMP_LIST].push_back({{ACL_DUMP_LAYER, kernelsList}});
        }
    }

    nlohmann::json content = {{"dump", aclDumpJson}};
    LOG_DEBUG("AclDumpGenStatJson dump json to " + GenAclJsonPath(dumpPath, rank));
    return FileOperation::DumpJson(GenAclJsonPath(dumpPath, rank), content);
}

DebuggerErrno AclDumper::AclDumpGenOverflowJson(std::shared_ptr<const OverflowCheckCfg> overflowCfg, uint32_t rank,
                                                uint32_t curStep)
{
    DEBUG_FUNC_TRACE();
    nlohmann::json aclDumpJson;
    bool needDump = AclDumper::IsIterNeedDump(curStep);
    const std::string& dumpPath = DebuggerConfig::GetInstance().GetOutputPath();
    std::string fullDumpPath;
    if (needDump) {
        fullDumpPath = GetDumpPath(curStep);
        FileUtils::CreateDir(fullDumpPath, true);
    } else {
        fullDumpPath = dumpPath;
    }

    DebuggerErrno ret = FileUtils::CreateDir(fullDumpPath, true);
    if (ret != DebuggerErrno::OK) {
        return ret;
    }

    aclDumpJson[ACL_DUMP_PATH] =  fullDumpPath;
    aclDumpJson[ACL_DUMP_DEBUG] = ACL_SWITCH_ON;
    if (!needDump) {
        aclDumpJson[ACL_DUMP_STEP] = std::to_string(INT_MAX);
    }
    nlohmann::json content = {{"dump", aclDumpJson}};
    LOG_DEBUG("AclDumpGenOverflowJson dump json to " + GenAclJsonPath(dumpPath, rank));
    return FileOperation::DumpJson(GenAclJsonPath(dumpPath, rank), content);
}

static DebuggerErrno InitAcl()
{
    DEBUG_FUNC_TRACE();
    nlohmann::json aclInitJson;
    std::string aclInitJsonPath = FileUtils::GetAbsPath("./aclinit.json");
    if (aclInitJsonPath.empty()) {
        LOG_ERROR(DebuggerErrno::ERROR_CANNOT_PARSE_PATH, "Failed to get full path of aclinit.json.");
        return DebuggerErrno::ERROR_CANNOT_PARSE_PATH;
    }

    constexpr const char* aclErrMsgOn = "1";
    aclInitJson["err_msg_mode"] = aclErrMsgOn;
    LOG_DEBUG("InitAcl dump json to " + aclInitJsonPath);
    DebuggerErrno result = FileOperation::DumpJson(aclInitJsonPath, aclInitJson);
    if (result != DebuggerErrno::OK) {
        LOG_ERROR(result, "Failed to dump data to " + aclInitJsonPath + ".");
        return result;
    }
    aclError ret;
    try {
        ret = CALL_ACL_API(AclInit, aclInitJsonPath.c_str());
    } catch (const std::runtime_error& e) {
        LOG_ERROR(DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND, "Cannot find function aclInit.");
        return DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND;
    }

    /* 此处框架可能会初始化，如果报重复初始化错误，忽略即可 */
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
        LOG_ERROR(DebuggerErrno::ERROR_EXTERNAL_API_ERROR, "Failed to init acl(" + std::to_string(ret) + ").");
        return DebuggerErrno::ERROR_EXTERNAL_API_ERROR;
    }

    LOG_DEBUG("InitAcl succeed");
    return DebuggerErrno::OK;
}

int32_t AclDumpCallBack(const AclDumpChunk* chunk, int32_t len)
{
    AclDumper& dumper = AclDumper::GetInstance();
    dumper.OnAclDumpCallBack(chunk, len);
    return 0;
}

DebuggerErrno AclDumper::Initialize()
{
    DEBUG_FUNC_TRACE();
    DebuggerErrno ret;
    aclError aclRet;
    const DebuggerConfig& cfg = DebuggerConfig::GetInstance();
    std::shared_ptr<const StatisticsCfg> statsCfg = cfg.GetStatisticsCfg();
    std::shared_ptr<const DumpTensorCfg> tensorCfg = cfg.GetDumpTensorCfg();
    std::shared_ptr<const OverflowCheckCfg> overflowCheckCfg = cfg.GetOverflowCheckCfg();

    ret = InitAcl();
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Failed to call InitAcl.");
        return ret;
    }

    foreDumpPath = GenDumpPath(cfg.GetOutputPath());

    bool needCallback = false;
    if (statsCfg != nullptr) {
        if (ELE_IN_VECTOR(statsCfg->summaryOption, DebuggerSummaryOption::MD5)) {
            hostAnalysisOpt = {DebuggerSummaryOption::MD5};
        }
        needCallback = true;
    }

    if (tensorCfg != nullptr && tensorCfg->fileFormat == DebuggerDumpFileFormat::FILE_FORMAT_NPY) {
        needCallback = true;
    }

    if (overflowCheckCfg != nullptr) {
        needCallback = true;
    }

    if (needCallback) {
        LOG_DEBUG("Register acl dump callback.");
        /* 上面aclInit成功，此处认为acldumpRegCallback符号也存在，不会抛出异常 */
        aclRet = CALL_ACL_API(AcldumpRegCallback, AclDumpCallBack, 0);
        if (aclRet != ACL_SUCCESS) {
            LOG_ERROR(DebuggerErrno::ERROR_EXTERNAL_API_ERROR,
                      "Failed to register acldump callback(" + std::to_string(aclRet) + ").");
            return DebuggerErrno::ERROR_EXTERNAL_API_ERROR;
        }
    }
    LOG_DEBUG("AclDumper::Initialize succeed");
    return DebuggerErrno::OK;
}

void AclDumper::OnAclDumpCallBack(const AclDumpChunk* chunk, int32_t len)
{
    DEBUG_FUNC_TRACE();
    CountOverflowNumbers(chunk);
    if (IsOverflowCompleted()) {
        return;
    }

    std::string dumpPath = FileUtils::GetAbsPath(chunk->fileName);
    auto it = dataProcessors.find(dumpPath);
    if (it == dataProcessors.end()) {
        if (dataProcessors.size() > PROCESSOR_NUM_MAX) {
            LOG_ERROR(DebuggerErrno::ERROR_BUFFER_OVERFLOW, "The number of processors has reached the upper limit.");
            return;
        }
        dataProcessors[dumpPath] = std::make_shared<AclDumpDataProcessor>(dumpPath, hostAnalysisOpt);
    }

    std::shared_ptr<AclDumpDataProcessor> processor = dataProcessors[dumpPath];
    DebuggerErrno ret = processor->PushData(chunk);
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Failed to push data " + dumpPath + ".");
    }

    LOG_DEBUG("Acl dump data processor " + dumpPath + " receive data, len=" +
              std::to_string(chunk->bufLen));

    if (!processor->IsCompleted()) {
        return;
    }

    if (!processor->ErrorOccurred()) {
        ret = processor->DumpToDisk();
    } else {
        ret = DebuggerErrno::ERROR;
    }

    dataProcessors.erase(dumpPath);
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Failed to write data " + dumpPath + " to disk.");
    }
    return;
}

void AclDumper::SetDump(uint32_t rank, uint32_t curStep, ExtArgs& args)
{
    DEBUG_FUNC_TRACE();
    DebuggerErrno ret;
    DebuggerConfig& cfg = DebuggerConfig::GetInstance();
    if (aclDumpHasSet || !cfg.IsRankHits(rank) || !IsCfgEnableAclDumper()) {
        return;
    }

    if (!initialized) {
        ret = Initialize();
        if (ret != DebuggerErrno::OK) {
            LOG_ERROR(ret, "AclDumper initialization failed.");
            return;
        }
        initialized = true;
    }

    /* 和acl dump相关的三个任务 */
    std::shared_ptr<const DumpTensorCfg> dumpTensorCfg = cfg.GetDumpTensorCfg();
    std::shared_ptr<const StatisticsCfg> statisticsCfg = cfg.GetStatisticsCfg();
    std::shared_ptr<const OverflowCheckCfg> overflowCheckCfg = cfg.GetOverflowCheckCfg();

    /* 当前只能三选一 */
    const char** kernels = GetExtArgs<const char**>(args, MindStudioExtensionArgs::ALL_KERNEL_NAMES);
    if (dumpTensorCfg != nullptr) {
        ret = AclDumpGenTensorJson(dumpTensorCfg, rank, curStep, kernels);
    } else if (statisticsCfg != nullptr) {
        ret = AclDumpGenStatJson(statisticsCfg, rank, curStep, kernels);
    } else if (overflowCheckCfg != nullptr) {
        ret = AclDumpGenOverflowJson(overflowCheckCfg, rank, curStep);
        overflowNums = overflowCheckCfg->overflowNums;
        isOverflowDump = true;
    }

    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "AclDumper failed to generate cfg file.");
        return;
    }

    aclError aclRet = CALL_ACL_API(AclmdlInitDump);
    if (aclRet != ACL_SUCCESS) {
        LOG_ERROR(DebuggerErrno::ERROR_EXTERNAL_API_ERROR,
                  "Failed to init acldump(" + std::to_string(aclRet) + ").");
        return;
    }

    const std::string& dumpPath = DebuggerConfig::GetInstance().GetOutputPath();
    aclRet = CALL_ACL_API(AclmdlSetDump, GenAclJsonPath(dumpPath, rank).c_str());
    if (aclRet != ACL_SUCCESS) {
        LOG_ERROR(DebuggerErrno::ERROR_EXTERNAL_API_ERROR,
                  "Failed to enable acldump(" + std::to_string(aclRet) + ").");
        return;
    }

    aclDumpHasSet = true;
    return;
}

void AclDumper::FinalizeDump(ExtArgs& args)
{
    DEBUG_FUNC_TRACE();
    if (!aclDumpHasSet) {
        return;
    }

    CALL_ACL_API(AclrtSynchronizeDevice);
    aclError aclRet = CALL_ACL_API(AclmdlFinalizeDump);
    if (aclRet != ACL_SUCCESS) {
        LOG_ERROR(DebuggerErrno::ERROR_EXTERNAL_API_ERROR,
                  "Failed to finalize acldump(" + std::to_string(aclRet) + ").");
    }

    aclDumpHasSet = false;
}

void KernelInitDump()
{
    if (AscendCLApi::LoadAclApi() != DebuggerErrno::OK) {
        return;
    }

    DebuggerErrno ret = InitAcl();
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Failed to call InitAcl.");
        return;
    }
    auto aclRet = CALL_ACL_API(AclmdlInitDump);
    if (aclRet != ACL_SUCCESS) {
    LOG_ERROR(DebuggerErrno::ERROR_EXTERNAL_API_ERROR,
              "Failed to init acldump(" + std::to_string(aclRet) + ").");
    return;
    }
}

void KernelSetDump(const std::string &filePath)
{
    std::string dumpPath = FileUtils::GetAbsPath(filePath);
    auto aclRet = CALL_ACL_API(AclmdlSetDump, dumpPath.c_str());
    if (aclRet != ACL_SUCCESS) {
    LOG_ERROR(DebuggerErrno::ERROR_EXTERNAL_API_ERROR,
              "Failed to enable acldump(" + std::to_string(aclRet) + ").");
    return;
    }
}

void KernelFinalizeDump()
{
    CALL_ACL_API(AclrtSynchronizeDevice);
    auto aclRet = CALL_ACL_API(AclmdlFinalizeDump);
    if (aclRet != ACL_SUCCESS) {
    LOG_ERROR(DebuggerErrno::ERROR_EXTERNAL_API_ERROR,
              "Failed to finalize acldump(" + std::to_string(aclRet) + ").");
    }
}
}