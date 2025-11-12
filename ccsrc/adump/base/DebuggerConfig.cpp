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
#include <fstream>
#include <sstream>
#include <cstring>
#include <numeric>
#include <re2/re2.h>

#include "include/ErrorCode.h"
#include "include/Macro.h"
#include "utils/FileUtils.h"
#include "base/ErrorInfosManager.h"
#include "DebuggerConfigFieldMap.h"
#include "DebuggerConfig.h"

namespace MindStudioDebugger {

template<typename T>
DebuggerErrno ParseJsonBaseObj2Var(const nlohmann::json& content, const std::string& field, T& output,
                                   bool mandatory = false)
{
    nlohmann::json::const_iterator iter = content.find(field);
    if (iter == content.end()) {
        if (mandatory) {
            return DebuggerErrno::ERROR_FIELD_NOT_EXISTS;
        } else {
            return DebuggerErrno::OK;
        }
    }

    try {
        output = iter->get<T>();
        return DebuggerErrno::OK;
    } catch (const nlohmann::detail::type_error& e) {
        /* 数据类型不匹配异常 */
        return DebuggerErrno::ERROR_INVALID_FORMAT;
    }
}

template<typename T>
DebuggerErrno ParseJsonStringAndTrans(const nlohmann::json& content,
                                      const std::string& field,
                                      const std::map<int32_t, std::string>& enum2name,
                                      T& output,
                                      bool mandatory = false)
{
    DebuggerErrno ret;
    std::string value;

    ret = ParseJsonBaseObj2Var<std::string>(content, field, value, true);
    if (ret == DebuggerErrno::ERROR_FIELD_NOT_EXISTS && !mandatory) {
        return DebuggerErrno::OK;
    }

    if (ret != DebuggerErrno::OK) {
        return ret;
    }

    int32_t enumId = GetEnumIdFromName(enum2name, value);
    if (enumId == DEBUGGER_INVALID_ENUM) {
        return DebuggerErrno::ERROR_UNKNOWN_VALUE;
    }

    output = static_cast<T>(enumId);
    return DebuggerErrno::OK;
}

#define PARSE_OPTIONAL_FIELD_CHECK_RET(content, field, output)                                                        \
    {                                                                                                                 \
        if (ParseJsonBaseObj2Var<decltype(output)>(content, field, output) != DebuggerErrno::OK) {                    \
            LOG_ERROR(DebuggerErrno::ERROR_UNKNOWN_VALUE,                                                             \
                      "Field " + std::string(field) + " cannot be parsed.");                                          \
        }                                                                                                             \
    }

#define PARSE_OPTIONAL_FIELD_TRANS_CHECK_RET(content, field, transMap, output)                                        \
    {                                                                                                                 \
        if (ParseJsonStringAndTrans<decltype(output)>(content, field, transMap, output) != DebuggerErrno::OK) {       \
            LOG_ERROR(DebuggerErrno::ERROR_UNKNOWN_VALUE,                                                             \
                      "Value of field " + std::string(field) + " is unknown.");                                       \
        }                                                                                                             \
    }

static bool DebuggerCfgParseUIntRangeGetBorder(const std::string& exp, uint32_t& left, uint32_t& right)
{
    if (std::count(exp.begin(), exp.end(), '-') != 1) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT,
                  "When using a range expression, it should be formatted as \"a-b\".");
        return false;
    }
    std::istringstream iss(exp);
    char dash;
    iss >> left >> dash >> right;
    if (iss.fail() || dash != '-') {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT,
                  "When using a range expression, it should be formatted as \"a-b\".");
        return false;
    }
    if (left >= right) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT,
                  "When using a range expression, the left border should be smaller than the right.");
        return false;
    }
    return true;
}

void DebuggerCfgParseUIntRange(const nlohmann::json& content, const std::string& name, std::vector<uint32_t>& range)
{
    if (!content.contains(name)) {
        return;
    }

    const nlohmann::json& array = content[name];
    if (!array.is_array()) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT, name + " should be empty or an array.");
        return;
    }

    range.clear();
    range.reserve(array.size());
    std::vector<std::pair<uint32_t, uint32_t>> buf;
    buf.reserve(array.size());
    uint32_t realLen = 0;
    /* a-b表示的范围可能很大，此处为了减少反复申请内存，对于a-b形式先预留空间再解析 */
    for (const auto& element : array) {
        if (element.is_number()) {
            range.emplace_back(element.get<uint32_t>());
            realLen++;
        } else if (element.is_string()) {
            std::string exp = element.get<std::string>();
            uint32_t begin;
            uint32_t end;
            if (!DebuggerCfgParseUIntRangeGetBorder(exp, begin, end)) {
                LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT, "Failed to parse " + name + ".");
                return;
            }
            uint32_t rangeSize = end - begin;
            if (realLen > UINT32_MAX - (rangeSize + 1)) {
                LOG_ERROR(DebuggerErrno::ERROR_VALUE_OVERFLOW, name + " size exceeds limit");
                return;
            }
            realLen += (rangeSize + 1);
            buf.emplace_back(std::make_pair(begin, end));
        }
    }

    constexpr uint32_t maxEleNum = 65536;
    if (realLen > maxEleNum) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT,
                  "When using a range expression in " + name + ", maximum of 65536 elements can be expressed.");
        return;
    }

    if (!buf.empty()) {
        range.reserve(realLen);
        for (const auto& border : buf) {
            for (uint32_t i = border.first; i <= border.second; ++i) {
                range.emplace_back(i);
            }
        }
    }
    return;
}

/* 老规则此处只能指定一个task，新规则允许task列表，出于兼容性考虑，此处允许输入string或list格式 */
void CommonCfgParseTasks(const nlohmann::json& content, std::vector<DebuggerTaskType>& tasks)
{
    std::vector<std::string> taskNameList;
    std::string taskName;
    DebuggerErrno ret;

    ret = ParseJsonBaseObj2Var<std::string>(content, TASK, taskName, true);
    if (ret == DebuggerErrno::ERROR_FIELD_NOT_EXISTS) {
        ret = ParseJsonBaseObj2Var<std::vector<std::string>>(content, TASKS, taskNameList, true);
    } else {
        taskNameList.emplace_back(taskName);
    }

    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Value of field task(s) should be string or list.");
        return;
    }

    for (auto& ele : taskNameList) {
        int32_t enumId = GetEnumIdFromName(TASK_TYPE_ENUM_2_NAME, ele);
        if (enumId == DEBUGGER_INVALID_ENUM) {
            LOG_WARNING(DebuggerErrno::ERROR_UNKNOWN_VALUE, "Task " + ele + " is unknown.");
            continue;
        }
        if (!ELE_IN_VECTOR(tasks, static_cast<DebuggerTaskType>(enumId))) {
            tasks.emplace_back(static_cast<DebuggerTaskType>(enumId));
        }
    }
    return;
}

constexpr char REGEX_PREFIX[] = "name-regex(";
constexpr char REGEX_SUFFIX[] = ")";
constexpr size_t REGEX_PREFIX_LEN = sizeof(REGEX_PREFIX) - 1;
constexpr size_t REGEX_SUFFIX_LEN = sizeof(REGEX_SUFFIX) - 1;
constexpr size_t REGEX_INDEX = REGEX_PREFIX_LEN - 1;

void KernelListMatcher::Parse(const std::vector<std::string>& expressions)
{
    for (auto& expression : expressions) {
        size_t len = expression.size();
        if (len < REGEX_SUFFIX_LEN) {
            LOG_WARNING(DebuggerErrno::ERROR_INVALID_VALUE, "Invalid expression when kernal match, too short: " + expression);
            continue;
        }
        if (strncmp(expression.c_str(), REGEX_PREFIX, REGEX_PREFIX_LEN) == 0 &&
            strncmp(expression.c_str() + (len - REGEX_SUFFIX_LEN), REGEX_SUFFIX, REGEX_SUFFIX_LEN) == 0) {
            /* name-regex(xxx)表示正则表达式 */
            regexList.emplace_back(expression.substr(REGEX_INDEX, len - REGEX_INDEX));
        } else {
            /* 否则认为是full scope name */
            fullNameList.emplace_back(expression);
        }
    }
}

std::vector<std::string> KernelListMatcher::GenRealKernelList(const char** fullKernelList) const
{
    std::vector<std::string> output;
    /* 返回空列表表示全部dump，返回一个空字符串表示没有匹配上的，都不dump */
    if (this->Empty() || fullKernelList == nullptr) {
        return output;
    }
    output = fullNameList;

    for (auto& pattern : regexList) {
        re2::RE2 reg(pattern, re2::RE2::Quiet);
        if (reg.ok()) {
            for (const char** ss = fullKernelList; *ss != nullptr; ++ss) {
                std::string ret;
                if (re2::RE2::FullMatch(*ss, reg, &ret)) {
                    output.emplace_back(*ss);
                }
            }
        }
    }

    if (output.empty()) {
        output.emplace_back("");
        LOG_INFO("No kernel matches, so nothing will be dumped.");
    }

    return output;
}

void CommonCfg::Parse(const nlohmann::json& content)
{
    CommonCfgParseTasks(content, tasks);
    if (tasks.empty()) {
        return;
    }

    PARSE_OPTIONAL_FIELD_CHECK_RET(content, OUTPUT_PATH, outputPath);
    outputPath = FileUtils::GetAbsPath(outputPath);
    DebuggerCfgParseUIntRange(content, RANK, rank);
    DebuggerCfgParseUIntRange(content, STEP, step);
    PARSE_OPTIONAL_FIELD_TRANS_CHECK_RET(content, LEVEL, DEBUGGER_LEVEL_ENUM_2_NAME, level);
    PARSE_OPTIONAL_FIELD_CHECK_RET(content, SEED, seed);
    PARSE_OPTIONAL_FIELD_CHECK_RET(content, IS_DETERMINISTIC, isDeterministic);
    PARSE_OPTIONAL_FIELD_CHECK_RET(content, ENABLE_DATALOADER, enableDataloader);
    PARSE_OPTIONAL_FIELD_CHECK_RET(content, ACL_CONFIG, aclConfig);
}

void DebuggerCfgParseDataMode(const nlohmann::json& content, DebuggerDataDirection& direction, DebuggerDataInOut& inout)
{
    std::vector<std::string> buf;
    bool fw;
    bool bw;
    bool in;
    bool out;
    bool all;

    direction = DebuggerDataDirection::DIRECTION_BOTH;
    inout = DebuggerDataInOut::INOUT_BOTH;
    PARSE_OPTIONAL_FIELD_CHECK_RET(content, DATA_MODE, buf);
    all = static_cast<bool>(std::find(buf.begin(), buf.end(), DATA_MODE_ALL) != buf.end());
    if (buf.empty() || all) {
        return;
    }

    fw = static_cast<bool>(std::find(buf.begin(), buf.end(), DIRECTION_FORWARD) != buf.end());
    bw = static_cast<bool>(std::find(buf.begin(), buf.end(), DIRECTION_BACKWARD) != buf.end());
    in = static_cast<bool>(std::find(buf.begin(), buf.end(), INOUT_INPUT) != buf.end());
    out = static_cast<bool>(std::find(buf.begin(), buf.end(), INOUT_OUTPUT) != buf.end());

    /* 互补项都配或都不配都表示both，因此关注不同的场景就行 */
    if (fw != bw) {
        if (fw) {
            direction = DebuggerDataDirection::DIRECTION_FORWARD;
        } else {
            direction = DebuggerDataDirection::DIRECTION_BACKWARD;
        }
    }
    if (in != out) {
        if (in) {
            inout = DebuggerDataInOut::INOUT_INPUT;
        } else {
            inout = DebuggerDataInOut::INOUT_OUTPUT;
        }
    }
    return;
}

void StatisticsCfgParseSummary(const nlohmann::json& content, std::vector<DebuggerSummaryOption>& summaryOption)
{
    /* 老规则支持"statistics"或"md5"，新规则支持"max"/"min"/"l2norm"/"md5"组合，此处兼容 */
    DebuggerErrno ret;
    std::string mode = STATISTICS;
    std::vector<std::string> modeListName;

    /* 若无该字段，认为是statistic，因此这里给mode设个默认值 */
    ret = ParseJsonBaseObj2Var<std::string>(content, SUMMARY_MODE, mode);
    if (ret == DebuggerErrno::OK) {
        if (mode == STATISTICS) {
            summaryOption.push_back(DebuggerSummaryOption::MAX);
            summaryOption.push_back(DebuggerSummaryOption::MIN);
            summaryOption.push_back(DebuggerSummaryOption::MEAN);
            summaryOption.push_back(DebuggerSummaryOption::L2NORM);
        } else if (mode == MD5) {
            summaryOption.push_back(DebuggerSummaryOption::MD5);
        } else {
            LOG_ERROR(DebuggerErrno::ERROR_UNKNOWN_VALUE, "Summary mode " + mode + " is unknown.");
        }
        return;
    }

    ret = ParseJsonBaseObj2Var<std::vector<std::string>>(content, SUMMARY_MODE, modeListName);
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Value of field summary_mode should be string or list.");
        return;
    }

    /* 若有该字段但值为空，认为是statistic */
    if (modeListName.empty()) {
        summaryOption.push_back(DebuggerSummaryOption::MAX);
        summaryOption.push_back(DebuggerSummaryOption::MIN);
        summaryOption.push_back(DebuggerSummaryOption::MEAN);
        summaryOption.push_back(DebuggerSummaryOption::L2NORM);
        return;
    }

    for (auto& ele : modeListName) {
        int32_t enumId = GetEnumIdFromName(SUMMARY_OPTION_ENUM_2_NAME, ele);
        if (enumId == DEBUGGER_INVALID_ENUM) {
            LOG_ERROR(DebuggerErrno::ERROR_UNKNOWN_VALUE, "Summary mode " + ele + " is unknown.");
            return;
        }
        summaryOption.push_back(static_cast<DebuggerSummaryOption>(enumId));
    }

    return;
}

void StatisticsCfg::Parse(const nlohmann::json& content)
{
    std::vector<std::string> filter;
    PARSE_OPTIONAL_FIELD_CHECK_RET(content, SCOPE, scope);
    PARSE_OPTIONAL_FIELD_CHECK_RET(content, LIST, filter);
    filter.erase(std::remove_if(filter.begin(), filter.end(),
                                [](const std::string& s) { return s.find_first_not_of(' ') == std::string::npos; }),
                 filter.end());
    list = std::move(filter);
    if (DebuggerConfig::GetInstance().GetDebugLevel() == DebuggerLevel::L2) {
        matcher.Parse(list);
    }
    DebuggerCfgParseDataMode(content, direction, inout);
    StatisticsCfgParseSummary(content, summaryOption);
}

void DumpTensorCfg::Parse(const nlohmann::json& content)
{
    std::vector<std::string> filter;
    PARSE_OPTIONAL_FIELD_CHECK_RET(content, SCOPE, scope);
    PARSE_OPTIONAL_FIELD_CHECK_RET(content, LIST, filter);
    filter.erase(std::remove_if(filter.begin(), filter.end(),
                                [](const std::string& s) { return s.find_first_not_of(' ') == std::string::npos; }),
                 filter.end());
    list = std::move(filter);
    if (DebuggerConfig::GetInstance().GetDebugLevel() == DebuggerLevel::L2) {
        matcher.Parse(list);
    }
    DebuggerCfgParseDataMode(content, direction, inout);
    PARSE_OPTIONAL_FIELD_TRANS_CHECK_RET(content, FILE_FORMAT, DUMP_FILE_FORMAT_ENUM_2_NAME, fileFormat);
    PARSE_OPTIONAL_FIELD_CHECK_RET(content, BACKWARD_INPUT, backwardInput);
}

void OverflowCheckCfg::Parse(const nlohmann::json& content)
{
    PARSE_OPTIONAL_FIELD_CHECK_RET(content, OVERFLOW_NUMS, overflowNums);
    PARSE_OPTIONAL_FIELD_TRANS_CHECK_RET(content, CHECK_MODE, OP_CHECK_LEVEL_ENUM_2_NAME, checkMode);
}

void DebuggerConfig::Reset()
{
    LOG_INFO("Reset configuration.");
    commonCfg = CommonCfg();
    statisticCfg.reset();
    dumpTensorCfg.reset();
    overflowCheckCfg.reset();
    loaded = false;
}

void DebuggerConfig::Parse()
{
    std::ifstream cfgFile;
    DebuggerErrno ret = FileUtils::OpenFile(cfgFilePath_, cfgFile);
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Failed to open file " + cfgFilePath_ + ".");
        return;
    }

    nlohmann::json content;
    nlohmann::json::const_iterator iter;
    try {
        cfgFile >> content;
    } catch (const nlohmann::json::parse_error& e) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT, "Failed to parse json file " + cfgFilePath_ + ".");
        return;
    }

    commonCfg.Parse(content);

#define PARSE_SUBTASK_CONFIG(enumeration, name, member, basetype) \
    do {                                                          \
        if (ELE_IN_VECTOR(commonCfg.tasks, enumeration)) {        \
            iter = content.find(name);                            \
            if (iter != content.end()) {                          \
                member = std::make_shared<basetype>();            \
                ((member)->Parse(*(iter)));                             \
            }                                                     \
        }                                                         \
    } while (0)

    PARSE_SUBTASK_CONFIG(DebuggerTaskType::TASK_DUMP_STATISTICS, TASK_STATISTICS, statisticCfg, StatisticsCfg);
    PARSE_SUBTASK_CONFIG(DebuggerTaskType::TASK_DUMP_TENSOR, TASK_DUMP_TENSOR, dumpTensorCfg, DumpTensorCfg);
    PARSE_SUBTASK_CONFIG(DebuggerTaskType::TASK_OVERFLOW_CHECK, TASK_OVERFLOW_CHECK, overflowCheckCfg, OverflowCheckCfg);

#undef PARSE_SUBTASK_CONFIG
    return;
}

int32_t DebuggerConfig::LoadConfig(const std::string& framework, const std::string& cfgFilePath)
{
    if (loaded) {
        LOG_WARNING(DebuggerErrno::ERROR, "Repeated initialization, which may lead to errors.");
        Reset();
    }

    cfgFilePath_ = FileUtils::GetAbsPath(cfgFilePath);
    if (cfgFilePath_ == "") {
        LOG_ERROR(DebuggerErrno::ERROR_CANNOT_PARSE_PATH, "Cannot parse path " + cfgFilePath + ".");
        return -1;
    }

    DebuggerErrno ret = FileUtils::CheckFileBeforeRead(cfgFilePath_, "r", FileType::JSON);
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Config file " + cfgFilePath + " is invalid.");
        return -1;
    }

    int32_t enumId = GetEnumIdFromName(FRAMEWORK_ENUM_2_NAME, framework);
    if (enumId == DEBUGGER_INVALID_ENUM) {
        LOG_ERROR(DebuggerErrno::ERROR_UNKNOWN_VALUE, "Unknown framework " + framework +  ".");
        return -1;
    }
    framework_ = static_cast<DebuggerFramework>(enumId);

    Parse();
    if (ErrorInfosManager::GetTopErrLevelInDuration() >= DebuggerErrLevel::LEVEL_ERROR) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to parse config file " + cfgFilePath + ".");
        return -1;
    }

    CheckConfigValidity();
    if (ErrorInfosManager::GetTopErrLevelInDuration() >= DebuggerErrLevel::LEVEL_ERROR) {
        LOG_ERROR(DebuggerErrno::ERROR, "Config file " + cfgFilePath + " is invalid.");
        return -1;
    }

    loaded = true;
    return 0;
}

bool DebuggerConfig::CheckConfigValidity()
{
    if (commonCfg.tasks.empty()) {
        LOG_WARNING(DebuggerErrno::ERROR, "No task configured. MsProbe will do nothing.");
        return true;
    }

    /* 解析时已做格式有效性校验，数值有效性放在python前端校验 */
    return true;
}

}
