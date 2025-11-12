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

#ifndef DEBUGGERCONFIG_H
#define DEBUGGERCONFIG_H

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <memory>
#include <set>
#include <stdexcept>
#include <nlohmann/json.hpp>

#include "include/Macro.h"

namespace MindStudioDebugger {

constexpr int DEBUGGER_INVALID_ENUM = -1;

enum class DebuggerFramework {
    FRAMEWORK_PYTORCH,
    FRAMEWORK_MINDSPORE,

    FRAMEWORK_BUTT,
};

enum class DebuggerTaskType {
    TASK_DUMP_TENSOR,
    TASK_DUMP_STATISTICS,
    TASK_OVERFLOW_CHECK,
    TASK_FREE_BENCHMARK,
    TASK_RUN_UT,
    TASK_GRAD_PROBE,

    TASK_BUTT = DEBUGGER_INVALID_ENUM,
};

enum class DebuggerDevType {
    DEVICE_TYPE_NPU,
    DEVICE_TYPE_GPU,
    DEVICE_TYPE_CPU,

    DEVICE_TYPE_BUTT = DEBUGGER_INVALID_ENUM,
};

enum class DebuggerLevel {
    L0,
    L1,
    L2,
    MIX,

    LEVEL_BUTT = DEBUGGER_INVALID_ENUM,
};

enum class DebuggerDataDirection {
    DIRECTION_FORWARD,
    DIRECTION_BACKWARD,
    DIRECTION_BOTH,

    DIRECTION_BUTT = DEBUGGER_INVALID_ENUM,
};

enum class DebuggerDataInOut {
    INOUT_INPUT,
    INOUT_OUTPUT,
    INOUT_BOTH,

    INOUT_BUTT = DEBUGGER_INVALID_ENUM,
};

enum class DebuggerDumpFileFormat {
    FILE_FORMAT_BIN,
    FILE_FORMAT_NPY,

    FILE_FORMAT_BUTT = DEBUGGER_INVALID_ENUM,
};

enum class DebuggerOpCheckLevel {
    CHECK_LEVEL_AICORE,
    CHECK_LEVEL_ATOMIC,
    CHECK_LEVEL_ALL,

    CHECK_LEVEL_BUTT = DEBUGGER_INVALID_ENUM,
};

enum class DebuggerSummaryOption {
    MAX,
    MIN,
    MEAN,
    L2NORM,
    NAN_CNT,
    NEG_INF_CNT,
    POS_INF_CNT,
    MD5,

    SUMMARY_BUTT = DEBUGGER_INVALID_ENUM,
};

class KernelListMatcher {
public:
    KernelListMatcher() = default;
    ~KernelListMatcher() = default;

    void Parse(const std::vector<std::string>& expressions);
    std::vector<std::string> GenRealKernelList(const char** fullKernelList) const;

    inline bool Empty() const {return fullNameList.empty() && regexList.empty();}
    inline bool NeedAllKernels() const {return !regexList.empty();}

private:
    std::vector<std::string> fullNameList;
    std::vector<std::string> regexList;
};

/* 说明：config类作为基础的配置解析查询类，对外应该是只读的，外部仅能通过Parse接口解析配置文件，而不应该直接修改配置字段，此处用以下方式防止外部误操作
 * 1、外部统一调用单例类DebuggerConfig的Parse解析配置文件，无法创建子配置类并调用其Parse函数
 * 2、子配置类通过添加DebuggerConfig为友元类允许其调用子配置类的Parse
 * 3、DebuggerConfig对外提供获取子配置类的方法，返回的是const类型指针，实现外部只读（而非将成员变量都写为private并提供get函数）
 */
class DebuggerConfig;

class CommonCfg {
public:
    friend class DebuggerConfig;
    CommonCfg() = default;
    ~CommonCfg() = default;

    std::vector<DebuggerTaskType> tasks;
    std::string outputPath{"./output"};
    std::vector<uint32_t> rank;
    std::vector<uint32_t> step;
    DebuggerLevel level{DebuggerLevel::L1};
    int32_t seed{1234};
    bool isDeterministic{false};
    bool enableDataloader{false};
    std::string aclConfig;

private:
    void Parse(const nlohmann::json &content);
};

class StatisticsCfg {
public:
    friend class DebuggerConfig;
    StatisticsCfg() = default;
    ~StatisticsCfg() = default;

    std::vector<std::string> scope;
    std::vector<std::string> list;
    KernelListMatcher matcher;
    DebuggerDataDirection direction{DebuggerDataDirection::DIRECTION_BOTH};
    DebuggerDataInOut inout{DebuggerDataInOut::INOUT_BOTH};
    std::vector<DebuggerSummaryOption> summaryOption;

private:
    void Parse(const nlohmann::json &content);
};

class DumpTensorCfg {
public:
    friend class DebuggerConfig;
    DumpTensorCfg() = default;
    ~DumpTensorCfg() = default;

    std::vector<std::string> scope;
    std::vector<std::string> list;
    KernelListMatcher matcher;
    DebuggerDataDirection direction{DebuggerDataDirection::DIRECTION_BOTH};
    DebuggerDataInOut inout{DebuggerDataInOut::INOUT_BOTH};
    DebuggerDumpFileFormat fileFormat{DebuggerDumpFileFormat::FILE_FORMAT_NPY};
    std::vector<std::string> backwardInput;

private:
    void Parse(const nlohmann::json &content);
};

class OverflowCheckCfg {
public:
    friend class DebuggerConfig;
    OverflowCheckCfg() = default;
    ~OverflowCheckCfg() = default;

    int32_t overflowNums{1};
    DebuggerOpCheckLevel checkMode{DebuggerOpCheckLevel::CHECK_LEVEL_ALL};

private:
    void Parse(const nlohmann::json &content);
};


class DebuggerConfig {
public:
    static DebuggerConfig& GetInstance()
    {
        static DebuggerConfig configInstance;
        return configInstance;
    }

    int32_t LoadConfig(const std::string& framework, const std::string& cfgFilePath);
    void Reset();

    bool IsCfgLoaded() const {return loaded;}
    DebuggerFramework GetFramework() const {return framework_;}
    const std::vector<DebuggerTaskType>& GetTaskList() const {return commonCfg.tasks;}
    const std::string& GetOutputPath() const {return commonCfg.outputPath;}
    const std::vector<uint32_t>& GetRankRange() const {return commonCfg.rank;};
    const std::vector<uint32_t>& GetStepRange() const {return commonCfg.step;};
    DebuggerLevel GetDebugLevel() const {return commonCfg.level;}
    int32_t GetRandSeed() const {return commonCfg.seed;}
    bool IsDeterministic() const {return commonCfg.isDeterministic;}
    bool IsDataloaderEnable() const {return commonCfg.enableDataloader;}
    std::string GetAclConfigPath() const {return commonCfg.aclConfig;}

    std::shared_ptr<const StatisticsCfg> GetStatisticsCfg() const
        {return std::const_pointer_cast<const StatisticsCfg>(statisticCfg);}
    std::shared_ptr<const DumpTensorCfg> GetDumpTensorCfg() const
        {return std::const_pointer_cast<const DumpTensorCfg>(dumpTensorCfg);}
    std::shared_ptr<const OverflowCheckCfg> GetOverflowCheckCfg() const
        {return std::const_pointer_cast<const OverflowCheckCfg>(overflowCheckCfg);}

    bool IsRankHits(uint32_t rankId) const
        {return commonCfg.rank.empty() || ELE_IN_VECTOR(commonCfg.rank, rankId);}
    bool IsStepHits(uint32_t stepId) const
        {return commonCfg.step.empty() || ELE_IN_VECTOR(commonCfg.step, stepId);}

private:
    DebuggerConfig() = default;
    ~DebuggerConfig() = default;
    explicit DebuggerConfig(const DebuggerConfig &obj) = delete;
    DebuggerConfig& operator=(const DebuggerConfig &obj) = delete;
    explicit DebuggerConfig(DebuggerConfig &&obj) = delete;
    DebuggerConfig& operator=(DebuggerConfig &&obj) = delete;

    void Parse();
    bool CheckConfigValidity();

    DebuggerFramework framework_;
    std::string cfgFilePath_;
    bool loaded{false};
    CommonCfg commonCfg;
    std::shared_ptr<StatisticsCfg> statisticCfg{nullptr};
    std::shared_ptr<DumpTensorCfg> dumpTensorCfg{nullptr};
    std::shared_ptr<OverflowCheckCfg> overflowCheckCfg{nullptr};
};

}

#endif