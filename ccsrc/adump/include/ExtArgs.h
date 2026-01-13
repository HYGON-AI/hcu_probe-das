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

#include <map>

namespace MindStudioDebugger {

enum class MindStudioExtensionArgs {
    ALL_KERNEL_NAMES = 0,         /* const std::vector<std::string> --> char** */
    IS_KBK = 1,                   /* bool */

    /* Add before this line */
    ARG_MAX,
};

using ExtArgs = std::map<MindStudioExtensionArgs, void*>;

template <typename T>
T GetExtArgs(ExtArgs& args, MindStudioExtensionArgs id)
{
    auto it = args.find(id);
    if (it == args.end()) {
        return nullptr;
    }

    return reinterpret_cast<T>(it->second);
}

}