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

#include <ios>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <vector>

#include "include/ErrorCode.h"

namespace MindStudioDebugger {

constexpr const char PATH_SEPARATOR = '/';
constexpr const uint32_t FULL_PATH_LENGTH_MAX = 4096;
constexpr const uint32_t FILE_NAME_LENGTH_MAX = 255;
constexpr const uint32_t PATH_DEPTH_MAX = 32;
constexpr const char* FILE_VALID_PATTERN = "^[a-zA-Z0-9_.:/-]+$";

constexpr size_t MAX_PKL_SIZE = 1024ULL * 1024 * 1024;
constexpr size_t MAX_NUMPY_SIZE = 10ULL * 1024 * 1024 * 1024;
constexpr size_t MAX_JSON_SIZE = 1024ULL * 1024 * 1024;
constexpr size_t MAX_PT_SIZE = 10ULL * 1024 * 1024 * 1024;
constexpr size_t MAX_CSV_SIZE = 1024ULL * 1024 * 1024;
constexpr size_t MAX_YAML_SIZE = 10ULL * 1024 * 1024;
constexpr size_t MAX_FILE_SIZE_DEFAULT = 10ULL * 1024 * 1024 * 1024;

constexpr mode_t NORMAL_FILE_MODE_DEFAULT = 0640;
constexpr mode_t READONLY_FILE_MODE_DEFAULT = 0440;
constexpr mode_t SCRIPT_FILE_MODE_DEFAULT = 0550;
constexpr mode_t NORMAL_DIR_MODE_DEFAULT = 0750;

enum class FileType {
    PKL,
    NUMPY,
    JSON,
    PT,
    CSV,
    YAML,

    /* Add new type before this line. */
    COMMON
};

namespace  FileUtils {

constexpr const uint32_t FILE_NAME_MAX = 255;

/* 基础检查函数库，不做过多校验，路径有效性由调用者保证 */
bool IsPathExist(const std::string& path);
std::vector<std::string> SplitPath(const std::string &path, char separator = PATH_SEPARATOR);
std::string GetAbsPath(const std::string &originpath);
bool IsDir(const std::string& path);
bool IsRegularFile(const std::string& path);
bool IsFileSymbolLink(const std::string& path);
bool IsPathCharactersValid(const std::string& path);
bool IsFileReadable(const std::string& path);
bool IsFileWritable(const std::string& path);
bool IsFileExecutable(const std::string& path);
bool IsDirReadable(const std::string& path);
std::string GetParentDir(const std::string& path);
std::string GetFileName(const std::string& path);
std::string GetFileBaseName(const std::string& path);
std::string GetFileSuffix(const std::string& path);
bool CheckFileRWX(const std::string& path, const std::string& permissions);
bool IsPathLengthLegal(const std::string& path);
bool IsPathDepthValid(const std::string& path);
bool IsFileOwner(const std::string& path);

/* 文件操作函数库，会对入参做基本检查 */
DebuggerErrno DeleteFile(const std::string &path);
DebuggerErrno DeleteDir(const std::string &path, bool recursion = false);
DebuggerErrno CreateDir(const std::string &path, bool recursion = false, mode_t mode = NORMAL_DIR_MODE_DEFAULT);
DebuggerErrno Chmod(const std::string& path, const mode_t& mode);
DebuggerErrno GetFileSize(const std::string &path, size_t& size);
DebuggerErrno OpenFile(const std::string& path, std::ifstream& ifs, std::ios::openmode mode = std::ios::in);
DebuggerErrno OpenFile(const std::string& path, std::ofstream& ofs, std::ios::openmode mode = std::ios::out,
                       mode_t permission = NORMAL_FILE_MODE_DEFAULT);

/* 通用检查函数 */
DebuggerErrno CheckFileSuffixAndSize(const std::string &path, FileType type);
DebuggerErrno CheckDirCommon(const std::string &path);
DebuggerErrno CheckFileBeforeRead(const std::string &path, const std::string& authority = "r",
                                  FileType type = FileType::COMMON);
DebuggerErrno CheckFileBeforeCreateOrWrite(const std::string &path, bool overwrite = false);
}
}