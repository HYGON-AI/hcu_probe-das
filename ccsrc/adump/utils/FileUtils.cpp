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

#include <string>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <map>
#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <cstdlib>
#include <dirent.h>
#include <fcntl.h>

#include "include/ErrorCode.h"
#include "FileUtils.h"

/* 部分环境上c++版本比较老，这里不用filesystem库实现 */

namespace MindStudioDebugger {
namespace  FileUtils {

using namespace  MindStudioDebugger;

/********************* 基础检查函数库，不做过多校验，路径有效性由调用者保证 ******************/
bool IsPathExist(const std::string& path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

static std::string GetFullPath(const std::string &originPath)
{
    if (originPath.empty()) {
        return "";
    }
    if (originPath[0] == '/') {
        return originPath;
    }

    std::string cwd;
    char cwdBuf[PATH_MAX];

    if (getcwd(cwdBuf, PATH_MAX) == nullptr) {
        return "";
    }

    cwd = cwdBuf;
    std::string fullPath = std::move(cwd + PATH_SEPARATOR + originPath);

    return fullPath;
}

std::vector<std::string> SplitPath(const std::string &path, char separator)
{
    std::vector<std::string> tokens;
    size_t len = path.length();
    size_t start = 0;

    while (start < len) {
        size_t end = path.find(separator, start);
        if (end == std::string::npos) {
            end = len;
        }
        if (start != end) {
            tokens.push_back(path.substr(start, end - start));
        }
        start = end + 1;
    }
    return tokens;
}

std::string GetAbsPath(const std::string &originpath)
{
    std::string fullPath = GetFullPath(originpath);
    if (fullPath.empty()) {
        return "";
    }

    std::vector<std::string> tokens = SplitPath(fullPath);
    std::vector<std::string> tokensRefined;

    for (std::string& token : tokens) {
        if (token.empty() || token == ".") {
            continue;
        } else if (token == "..") {
            if (tokensRefined.empty()) {
                return "";
            }
            tokensRefined.pop_back();
        } else {
            tokensRefined.emplace_back(token);
        }
    }

    if (tokensRefined.empty()) {
        return "/";
    }

    std::string resolvedPath("");
    for (std::string& token : tokensRefined) {
        resolvedPath.append("/").append(token);
    }

    return resolvedPath;
}

bool IsDir(const std::string& path)
{
    struct stat buffer;
    if (stat(path.c_str(), &buffer) == 0) {
        return (buffer.st_mode & S_IFDIR) != 0;
    }
    return false;
}

bool IsRegularFile(const std::string& path)
{
    struct stat pathStat;
    if (stat(path.c_str(), &pathStat) == 0) {
        return S_ISREG(pathStat.st_mode);
    }
    return false;
}

bool IsFileSymbolLink(const std::string& path)
{
    struct stat buffer;
    if (lstat(path.c_str(), &buffer) == 0) {
        if (S_ISLNK(buffer.st_mode)) {
            return true;
        }
    }
    return false;
}

bool IsPathCharactersValid(const std::string& path)
{
    for (const char& ch : path) {
        if (!std::isalnum(ch) && ch != '_' && ch != '.' && ch != ':' && ch != '/' && ch != '-') {
            return false;
        }
    }
    return true;
}

bool IsFileReadable(const std::string& path)
{
    return access(path.c_str(), R_OK) == 0;
}

bool IsFileWritable(const std::string& path)
{
    return access(path.c_str(), W_OK) == 0;
}

bool IsFileExecutable(const std::string& path)
{
    return (access(path.c_str(), R_OK) == 0) && (access(path.c_str(), X_OK) == 0);
}

bool IsDirReadable(const std::string& path)
{
    return (access(path.c_str(), R_OK) == 0) && (access(path.c_str(), X_OK) == 0);
}

std::string GetParentDir(const std::string& path)
{
    size_t found = path.find_last_of('/');
    if (found != std::string::npos) {
        return path.substr(0, found);
    }
    return ".";
}

std::string GetFileName(const std::string& path)
{
    size_t found = path.find_last_of('/');
    if (found != std::string::npos) {
        return path.substr(found + 1);
    }
    return path;
}

std::string GetFileBaseName(const std::string& path)
{
    std::string fileName = GetFileName(path);
    size_t dotPos = fileName.find_last_of('.');
    if (dotPos != std::string::npos) {
        return fileName.substr(0, dotPos);
    }
    return fileName;
}

std::string GetFileSuffix(const std::string& path)
{
    std::string fileName = GetFileName(path);
    size_t dotPos = fileName.find_last_of('.');
    if (dotPos != std::string::npos && dotPos + 1 < fileName.size()) {
        return fileName.substr(dotPos + 1);
    }
    return "";
}

bool CheckFileRWX(const std::string& path, const std::string& permissions)
{
    if (permissions.find('r') != std::string::npos && !IsFileReadable(path)) {
        return false;
    }
    if (permissions.find('w') != std::string::npos && !IsFileWritable(path)) {
        return false;
    }
    if (permissions.find('x') != std::string::npos && !IsFileExecutable(path)) {
        return false;
    }
    return true;
}

bool IsPathLengthLegal(const std::string& path)
{
    if (path.length() > FULL_PATH_LENGTH_MAX || path.length() == 0) {
        return false;
    }

    std::vector<std::string> tokens = SplitPath(path);
    for (auto token : tokens) {
        if (token.length() > FILE_NAME_LENGTH_MAX) {
            return false;
        }
    }

    return true;
}

bool IsPathDepthValid(const std::string& path)
{
    auto depth = static_cast<uint32_t>(std::count(path.begin(), path.end(), PATH_SEPARATOR));
    return depth <= PATH_DEPTH_MAX;
}

bool IsFileOwner(const std::string& path)
{
    struct stat fileStat;
    if (stat(path.c_str(), &fileStat) == 0) {
        if (fileStat.st_uid == getuid()) {
            return true;
        }
    }
    return false;
}

/****************** 文件操作函数库，会对入参做基本检查 ************************/
DebuggerErrno DeleteFile(const std::string &path)
{
    if (!IsPathExist(path)) {
        return DebuggerErrno::OK;
    }
    if (IsFileSymbolLink(path)) {
        return DebuggerErrno::ERROR_NOT_ALLOW_SOFTLINK;
    }

    if (remove(path.c_str()) == 0) {
        return DebuggerErrno::OK;
    } else {
        return DebuggerErrno::ERROR_SYSCALL_FAILED;
    }
}

static DebuggerErrno DeleteDirRec(const std::string &path, uint32_t depth)
{
    if (depth > PATH_DEPTH_MAX) {
        return DebuggerErrno::ERROR_PATH_TOO_DEEP;
    }

    DebuggerErrno ret;
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        return DebuggerErrno::ERROR_SYSCALL_FAILED;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || (strcmp(entry->d_name, "..") == 0)) {
            continue;
        }
        std::string entryPath = path + "/" + entry->d_name;
        if (entry->d_type == DT_DIR) {
            ret = DeleteDirRec(entryPath, depth + 1);
            if (ret != DebuggerErrno::OK) {
                closedir(dir);
                return ret;
            }
        } else if (entry->d_type == DT_REG || entry->d_type == DT_LNK) {
            if (remove(entryPath.c_str()) != 0) {
                closedir(dir);
                return DebuggerErrno::ERROR_SYSCALL_FAILED;
            }
        } else {
                closedir(dir);
                return DebuggerErrno::ERROR_ILLEGAL_FILE_TYPE;
        }
    }

    closedir(dir);
    if (rmdir(path.c_str()) != 0) {
        if (errno == EACCES || errno == EROFS) {
            return DebuggerErrno::ERROR_PERMISSION_DENINED;
        } else {
            return DebuggerErrno::ERROR_SYSCALL_FAILED;
        }
    }

    return DebuggerErrno::OK;
}

DebuggerErrno DeleteDir(const std::string &path, bool recursion)
{
    if (!IsPathExist(path)) {
        return DebuggerErrno::OK;
    }
    if (IsFileSymbolLink(path)) {
        return DebuggerErrno::ERROR_NOT_ALLOW_SOFTLINK;
    }

    if (recursion) {
        return DeleteDirRec(path, 0);
    }

    if (rmdir(path.c_str()) != 0) {
        return DebuggerErrno::ERROR_SYSCALL_FAILED;
    }

    return DebuggerErrno::OK;
}

static DebuggerErrno CreateDirAux(const std::string& path, bool recursion, mode_t mode)
{
    std::string parent = GetParentDir(path);
    DebuggerErrno ret;

    if (!IsPathExist(parent)) {
        if (!recursion) {
            return DebuggerErrno::ERROR_DIR_NOT_EXISTS;
        }
        /* 递归创建父目录，由于前面已经判断过目录深度，此处递归是安全的 */
        ret = CreateDirAux(parent, recursion, mode);
        if (ret != DebuggerErrno::OK) {
            return ret;
        }
    }

    if (mkdir(path.c_str(), mode) != 0) {
        if (errno == EACCES || errno == EROFS) {
            return DebuggerErrno::ERROR_PERMISSION_DENINED;
        } else {
            return DebuggerErrno::ERROR_SYSCALL_FAILED;
        }
    }
    return DebuggerErrno::OK;
}

DebuggerErrno CreateDir(const std::string &path, bool recursion, mode_t mode)
{
    if (IsPathExist(path)) {
        return DebuggerErrno::OK;
    }

    std::string realPath = GetAbsPath(path);
    if (realPath.empty()) {
        return DebuggerErrno::ERROR_CANNOT_PARSE_PATH;
    }
    if (!IsPathLengthLegal(realPath)) {
        return DebuggerErrno::ERROR_PATH_TOO_LOOG;
    }
    if (!IsPathCharactersValid(realPath)) {
        return DebuggerErrno::ERROR_PATH_CONTAINS_INVALID_CHAR;
    }
    if (!IsPathDepthValid(realPath)) {
        return DebuggerErrno::ERROR_PATH_TOO_DEEP;
    }

    return CreateDirAux(realPath, recursion, mode);
}

DebuggerErrno Chmod(const std::string& path, const mode_t& mode)
{
    if (!IsPathExist(path)) {
        return DebuggerErrno::ERROR_FILE_NOT_EXISTS;
    }
    if (IsFileSymbolLink(path)) {
        return DebuggerErrno::ERROR_NOT_ALLOW_SOFTLINK;
    }

    std::string absPath = GetAbsPath(path);
    if (absPath.empty()) {
        return DebuggerErrno::ERROR_CANNOT_PARSE_PATH;
    }
    return chmod(absPath.c_str(), mode) == 0 ? DebuggerErrno::OK : DebuggerErrno::ERROR_SYSCALL_FAILED;
}

DebuggerErrno GetFileSize(const std::string &path, size_t& size)
{
    struct stat pathStat;
    if (stat(path.c_str(), &pathStat) != 0) {
        return DebuggerErrno::ERROR_FILE_NOT_EXISTS;
    }
    if (!S_ISREG(pathStat.st_mode)) {
        return DebuggerErrno::ERROR_ILLEGAL_FILE_TYPE;
    }

    size = static_cast<size_t>(pathStat.st_size);
    return DebuggerErrno::OK;
}

DebuggerErrno OpenFile(const std::string& path, std::ifstream& ifs, std::ios::openmode mode)
{
    std::string realPath = GetAbsPath(path);
    DebuggerErrno ret = CheckFileBeforeRead(realPath);
    if (ret != DebuggerErrno::OK) {
        return ret;
    }

    std::ifstream tmpifs(realPath, mode);
    if (!tmpifs.is_open()) {
        return DebuggerErrno::ERROR_FAILED_TO_OPEN_FILE;
    }

    ifs = std::move(tmpifs);
    return DebuggerErrno::OK;
}

DebuggerErrno OpenFile(const std::string& path, std::ofstream& ofs, std::ios::openmode mode, mode_t permission)
{
    DebuggerErrno ret;
    std::string realPath = GetAbsPath(path);
    if (realPath.empty()) {
        return DebuggerErrno::ERROR_CANNOT_PARSE_PATH;
    }

    std::string parent = GetParentDir(realPath);
    ret = CheckFileBeforeCreateOrWrite(realPath, true);
    if (ret != DebuggerErrno::OK) {
        return ret;
    }

    if (!IsPathExist(parent)) {
        ret = CreateDir(parent, true);
        if (ret != DebuggerErrno::OK) {
            return ret;
        }
    }

    if (!IsPathExist(realPath)) {
        int fd = open(realPath.c_str(), O_CREAT | O_WRONLY, permission);
        if (fd < 0) {
            return DebuggerErrno::ERROR_FAILED_TO_OPEN_FILE;
        }
        close(fd);
    }

    std::ofstream tmpofs(realPath, mode);
    if (!tmpofs.is_open()) {
        return DebuggerErrno::ERROR_FAILED_TO_OPEN_FILE;
    }

    ofs = std::move(tmpofs);
    return DebuggerErrno::OK;
}

/******************************* 通用检查函数 **********************************/
DebuggerErrno CheckFileSuffixAndSize(const std::string &path, FileType type)
{
    static const std::map<FileType, std::pair<std::string, size_t>> FileTypeCheckTbl = {
        {FileType::PKL, {"kpl", MAX_PKL_SIZE}},
        {FileType::NUMPY, {"npy", MAX_NUMPY_SIZE}},
        {FileType::JSON, {"json", MAX_JSON_SIZE}},
        {FileType::PT, {"pt", MAX_PT_SIZE}},
        {FileType::CSV, {"csv", MAX_CSV_SIZE}},
        {FileType::YAML, {"yaml", MAX_YAML_SIZE}},
    };

    size_t size;
    DebuggerErrno ret = GetFileSize(path, size);
    if (ret != DebuggerErrno::OK) {
        return ret;
    }

    if (type == FileType::COMMON) {
        if (size > MAX_FILE_SIZE_DEFAULT) {
            return DebuggerErrno::ERROR_FILE_TOO_LARGE;
        }
        return DebuggerErrno::OK;
    }

    auto iter = FileTypeCheckTbl.find(type);
    if (iter == FileTypeCheckTbl.end()) {
        return DebuggerErrno::ERROR_UNKNOWN_FILE_SUFFIX;
    }

    std::string suffix = GetFileSuffix(path);
    if (suffix != iter->second.first) {
        return DebuggerErrno::ERROR_UNKNOWN_FILE_SUFFIX;
    }
    if (size > iter->second.second) {
        return DebuggerErrno::ERROR_FILE_TOO_LARGE;
    }

    return DebuggerErrno::OK;
}

DebuggerErrno CheckDirCommon(const std::string &path)
{
    std::string realPath = GetAbsPath(path);
    if (realPath.empty()) {
        return DebuggerErrno::ERROR_CANNOT_PARSE_PATH;
    }
    if (!IsPathExist(realPath)) {
        return DebuggerErrno::ERROR_FILE_NOT_EXISTS;
    }
    if (!IsDir(realPath)) {
        return DebuggerErrno::ERROR_ILLEGAL_FILE_TYPE;
    }
    if (!IsPathLengthLegal(realPath)) {
        return DebuggerErrno::ERROR_PATH_TOO_LOOG;
    }
    if (!IsPathCharactersValid(realPath)) {
        return DebuggerErrno::ERROR_PATH_CONTAINS_INVALID_CHAR;
    }
    if (!IsPathDepthValid(realPath)) {
        return DebuggerErrno::ERROR_PATH_TOO_DEEP;
    }
    if (IsFileSymbolLink(path)) {
        return DebuggerErrno::ERROR_NOT_ALLOW_SOFTLINK;
    }
    if (!IsDirReadable(path)) {
        return DebuggerErrno::ERROR_PERMISSION_DENINED;
    }

    return DebuggerErrno::OK;
}

DebuggerErrno CheckFileBeforeRead(const std::string &path, const std::string& authority, FileType type)
{
    std::string realPath = GetAbsPath(path);
    if (realPath.empty()) {
        return DebuggerErrno::ERROR_CANNOT_PARSE_PATH;
    }
    if (!IsPathExist(realPath)) {
        return DebuggerErrno::ERROR_FILE_NOT_EXISTS;
    }
    if (!IsPathLengthLegal(realPath)) {
        return DebuggerErrno::ERROR_PATH_TOO_LOOG;
    }
    if (!IsPathCharactersValid(realPath)) {
        return DebuggerErrno::ERROR_PATH_CONTAINS_INVALID_CHAR;
    }
    if (!IsPathDepthValid(realPath)) {
        return DebuggerErrno::ERROR_PATH_TOO_DEEP;
    }
    if (IsFileSymbolLink(realPath)) {
        return DebuggerErrno::ERROR_NOT_ALLOW_SOFTLINK;
    }
    if (!CheckFileRWX(realPath, authority)) {
        return DebuggerErrno::ERROR_PERMISSION_DENINED;
    }

    /* 如果是/dev/random之类的无法计算size的文件，不要用本函数check */
    return CheckFileSuffixAndSize(path, type);
}

DebuggerErrno CheckFileBeforeCreateOrWrite(const std::string &path, bool overwrite)
{
    std::string realPath = GetAbsPath(path);
    if (realPath.empty()) {
        return DebuggerErrno::ERROR_CANNOT_PARSE_PATH;
    }
    if (!IsPathLengthLegal(realPath)) {
        return DebuggerErrno::ERROR_PATH_TOO_LOOG;
    }
    if (!IsPathCharactersValid(realPath)) {
        return DebuggerErrno::ERROR_PATH_CONTAINS_INVALID_CHAR;
    }
    if (!IsPathDepthValid(realPath)) {
        return DebuggerErrno::ERROR_PATH_TOO_DEEP;
    }
    if (IsPathExist(realPath)) {
        if (!overwrite) {
            return DebuggerErrno::ERROR_FILE_ALREADY_EXISTS;
        }

        /* 默认不允许覆盖其他用户创建的文件，若有特殊需求（如多用户通信管道等）由业务自行校验 */
        if (!IsFileWritable(realPath) || !IsFileOwner(realPath)) {
            return DebuggerErrno::ERROR_PERMISSION_DENINED;
        }
    }
    return DebuggerErrno::OK;
}
}
}