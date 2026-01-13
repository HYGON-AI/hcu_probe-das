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

#include <sstream>

#include "utils/CPythonUtils.h"
#include "DebuggerConfig.h"
#include "Environment.h"

namespace MindStudioDebugger {
namespace Environment {

static int32_t GetPTRankID()
{
    /*  if torch.distributed.is_initialized():
     *     return torch.distributed.get_rank()
     */
    CPythonUtils::PythonObject torch = CPythonUtils::PythonObject::Import("torch");
    if (!torch.IsModule()) {
        return -1;
    }

    CPythonUtils::PythonObject distributed = torch.Get("distributed");
    if (distributed.IsNone()) {
        return -1;
    }

    if (!distributed.Get("is_initialized").Call()) {
        return -1;
    }

    CPythonUtils::PythonObject rank = distributed.Get("get_rank").Call();
    int32_t id;
    if (rank.To(id) != 0) {
        return -1;
    }
    return id;
}

static int32_t GetMSRankID()
{
    constexpr const char* RANK_ID = "RANK_ID";
    const char* rankIdEnv =  getenv(RANK_ID);
    if (rankIdEnv == nullptr) {
        return -1;
    }

    std::string rankId(rankIdEnv);
    std::istringstream iss(rankId);
    int32_t id = -1;
    if (!(iss >> id) || id < 0) {
        return -1;
    }

    return id;
}

int32_t GetRankID()
{
    if (!DebuggerConfig::GetInstance().IsCfgLoaded()) {
        return -1;
    }

    static int32_t id = -1;
    if (id >= 0) {
        return id;
    }

    if (DebuggerConfig::GetInstance().GetFramework() == DebuggerFramework::FRAMEWORK_PYTORCH) {
        id = GetPTRankID();
    } else if (DebuggerConfig::GetInstance().GetFramework() == DebuggerFramework::FRAMEWORK_MINDSPORE) {
        id = GetMSRankID();
    }

    return id;
}

}
}