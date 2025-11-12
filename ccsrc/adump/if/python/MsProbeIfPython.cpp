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

#include <Python.h>

#include "PrecisionDebuggerIfPython.h"
#include "CPythonAgent.h"
#include "ACLDump.h"

namespace MindStudioDebugger {

PyDoc_STRVAR(MsProbeCModuleDoc,
"The part of the module msprobe that is implemented in CXX.\n\
class _PrecisionDebugger: PrecisionDebugger in CXX \n\
class _DebuggerConfig: Configuration data of PrecisionDebugger \n\
class CPythonAgent: Used for front-end and back-end code interactions \n\
    \n\
...");

static struct PyModuleDef g_MsProbeCModule = {
    PyModuleDef_HEAD_INIT,
    "_msprobe_c",                  /* m_name */
    MsProbeCModuleDoc,            /* m_doc */
    -1,                           /* m_size */
    nullptr,                      /* m_methods */
};

}

PyMODINIT_FUNC PyInit__msprobe_c(void)
{
    PyObject* m = PyModule_Create(&MindStudioDebugger::g_MsProbeCModule);
    if (m == nullptr) {
        return nullptr;
    }

    PyTypeObject* precisionDebugger = MindStudioDebugger::GetPyPrecisionDebuggerType();
    if (precisionDebugger == nullptr) {
        PyErr_SetString(PyExc_ImportError, "Failed to create class _PrecisionDebugger.");
        Py_DECREF(m);
        return nullptr;
    }
    if (PyModule_AddObject(m, "_PrecisionDebugger", reinterpret_cast<PyObject*>(precisionDebugger)) < 0) {
        PyErr_SetString(PyExc_ImportError, "Failed to bind class _PrecisionDebugger.");
        Py_DECREF(m);
        return nullptr;
    }

    PyObject* cpyAgent = MindStudioDebugger::GetCPythonAgentModule();
    if (cpyAgent == nullptr) {
        PyErr_SetString(PyExc_ImportError, "Failed to create submodule CPythonAgent.");
        Py_DECREF(m);
        return nullptr;
    }
    if (PyModule_AddObject(m, "CPythonAgent", cpyAgent) < 0) {
        PyErr_SetString(PyExc_ImportError, "Failed to bind submodule CPythonAgent.");
        Py_DECREF(m);
        return nullptr;
    }

    PyMethodDef* dumpmethods = MindStudioDebugger::GetDumpMethods();
    for (PyMethodDef* method = dumpmethods; method->ml_name != nullptr; ++method) {
        PyObject* func = PyCFunction_New(method, nullptr);
        if (func == nullptr) {
            PyErr_SetString(PyExc_ImportError, "Failed to create dump method.");
            Py_DECREF(m);
            return nullptr;
        }
        if (PyModule_AddObject(m, method->ml_name, func) < 0) {
            Py_DECREF(func); // 释放未被模块接管的方法对象
            PyErr_SetString(PyExc_ImportError, "Failed to bind dump method.");
            Py_DECREF(m);
            return nullptr;
        }
    }
    return m;
}