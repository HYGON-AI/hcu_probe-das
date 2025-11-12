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

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "include/ExtArgs.h"
#include "base/DebuggerConfig.h"
#include "AclDumpDataProcessor.h"

namespace MindStudioDebugger {

class AclDumper {
public:
    static AclDumper& GetInstance()
    {
        static AclDumper dumperInstance;
        return dumperInstance;
    }

    static bool IsIterNeedDump(uint32_t iterId);
    static bool IsCfgEnableAclDumper();

    void SetDump(uint32_t rank, uint32_t curStep, ExtArgs& args);
    void FinalizeDump(ExtArgs& args);
    void OnAclDumpCallBack(const AclDumpChunk* chunk, int32_t len);

    std::string GetDumpPath(uint32_t curStep) const;

private:
    AclDumper() = default;
    ~AclDumper() = default;
    explicit AclDumper(const AclDumper &obj) = delete;
    AclDumper& operator=(const AclDumper &obj) = delete;
    explicit AclDumper(AclDumper &&obj) = delete;
    AclDumper& operator=(AclDumper &&obj) = delete;

    DebuggerErrno Initialize();
    DebuggerErrno AclDumpGenTensorJson(std::shared_ptr<const DumpTensorCfg> dumpTensorCfg, uint32_t rank,
                                       uint32_t curStep, const char** kernels);
    DebuggerErrno AclDumpGenStatJson(std::shared_ptr<const StatisticsCfg> statisticsCfg, uint32_t rank,
                                     uint32_t curStep, const char** kernels);
    DebuggerErrno AclDumpGenOverflowJson(std::shared_ptr<const OverflowCheckCfg> overflowCfg, uint32_t rank,
                                         uint32_t curStep);
    void CountOverflowNumbers(const AclDumpChunk* chunk);
    bool IsOverflowCompleted();

    bool initialized{false};
    bool aclDumpHasSet{false};
    std::string foreDumpPath;
    std::vector<DebuggerSummaryOption> hostAnalysisOpt;
    std::map<std::string, std::shared_ptr<AclDumpDataProcessor>> dataProcessors;
    bool isOverflowDump{false};
    int32_t overflowNums{1};
    int32_t realOverflowNums{0};
};

void KernelInitDump();
void KernelSetDump(const std::string &filePath);
void KernelFinalizeDump();
}