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

