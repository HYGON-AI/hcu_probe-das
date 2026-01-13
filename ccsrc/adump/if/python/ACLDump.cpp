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
