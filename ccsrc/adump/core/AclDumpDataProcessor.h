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

#ifndef ACLDUMPDATAPROCESSOR_H
#define ACLDUMPDATAPROCESSOR_H

#include <string>
#include <vector>
#include <queue>

#include "include/ErrorCode.h"
#include "base/DebuggerConfig.h"
#include "third_party/ACL/AclApi.h"

namespace MindStudioDebugger {

constexpr size_t MAX_DATA_LEN = 4ULL * 1024 * 1024 * 1024;

class AclDumpDataProcessor {
public:
    AclDumpDataProcessor(const std::string& path, const std::vector<DebuggerSummaryOption>& opts)
        : dumpPath{path}, hostAnalysisOpts{opts} {};
    ~AclDumpDataProcessor();

    bool IsCompleted() const {return completed;}
    bool ErrorOccurred() const {return errorOccurred;}
    DebuggerErrno PushData(const AclDumpChunk *chunk);
    DebuggerErrno DumpToDisk();
    std::string ToString() const;

private:
    DebuggerErrno ConcatenateData();

    std::string dumpPath;
    bool completed{false};
    bool errorOccurred{false};
    size_t totalLen{0};
    size_t headerSegOffset{0};
    size_t headerSegLen{0};
    size_t dataSegOffset{0};
    size_t dataSegLen{0};
    std::queue<std::vector<uint8_t>*> buffer;
    std::vector<DebuggerSummaryOption> hostAnalysisOpts;
};

}

#endif

