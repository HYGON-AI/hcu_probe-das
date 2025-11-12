/*
 * Copyright (C) 2025-2025. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <cstring>
#include <exception>
#include <stdexcept>

#include "base/ErrorInfosManager.h"
#include "core/AclDumper.h"
#include "utils/CPythonUtils.h"

namespace MindStudioDebugger {

static PyObject *CPythonKernelInitDump(PyObject *module, PyObject *args)
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    KernelInitDump();
    PyGILState_Release(gstate);
    Py_RETURN_NONE;
}

static PyObject *CPythonKernelSetDump(PyObject *module, PyObject *args)
{
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path)) {
    LOG_ERROR(DebuggerErrno::ERROR_INVALID_VALUE,
              "npu set dump error, cfg_file must string");
    return nullptr;
    }
    PyGILState_STATE gstate = PyGILState_Ensure();
    KernelSetDump(std::string(path));
    PyGILState_Release(gstate);
    Py_RETURN_NONE;
}

static PyObject *CPythonKernelFinalizeDump(PyObject *module, PyObject *args)
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    KernelFinalizeDump();
    PyGILState_Release(gstate);
    Py_RETURN_NONE;
}

static PyMethodDef DumpMethods[] = {
    {"init_dump", reinterpret_cast<PyCFunction>(CPythonKernelInitDump),
     METH_NOARGS, "Initialize dump."},
    {"set_dump", reinterpret_cast<PyCFunction>(CPythonKernelSetDump),
     METH_VARARGS, "Set dump."},
    {"finalize_dump", reinterpret_cast<PyCFunction>(CPythonKernelFinalizeDump),
     METH_NOARGS, "Finalize dump."},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef *GetDumpMethods() { return DumpMethods; }
} // namespace MindStudioDebugger
