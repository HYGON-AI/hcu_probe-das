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