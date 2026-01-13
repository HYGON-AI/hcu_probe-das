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

#include <stdexcept>
#include <exception>
#include <cstring>

#include "utils/CPythonUtils.h"

namespace MindStudioDebugger {

PyDoc_STRVAR(CPythonAgentModuleDoc,
"A module for Python code to interact with C++ code.\n\
 \n\
...");

static PyObject* CPythonAgentRegister(PyObject *module, PyObject *args)
{
    if (args == nullptr || !PyTuple_Check(args)) {
        PyErr_SetString(PyExc_TypeError, "Expect a tuple.");
        Py_RETURN_NONE;
    }
    /* 预期2个参数，name和obj */
    if (PyTuple_GET_SIZE(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "\'register_context\' expects 2 arguments.");
        Py_RETURN_NONE;
    }

    PyObject* obj = nullptr;
    const char* name = nullptr;
    if (!PyArg_ParseTuple(args, "sO", &name, &obj)) {
        PyErr_SetString(PyExc_TypeError, "\"name\" should be a string and \"obj\" should be a python object.");
        Py_RETURN_NONE;
    }

    if (CPythonUtils::RegisterPythonObject(name, obj) != 0) {
        if (CPythonUtils::IsPyObjRegistered(name)) {
            PyErr_Format(PyExc_RuntimeError, "\"%s\" has been registered already.", name);
        } else {
            PyErr_Format(PyExc_RuntimeError, "Failed to register \"%s\".", name);
        }
    }

    Py_RETURN_NONE;
}

static PyObject* CPythonAgentUnRegister(PyObject *module, PyObject *obj)
{
    CPythonUtils::PythonStringObject name(obj);
    if (name.IsNone()) {
        PyErr_SetString(PyExc_TypeError, "\"name\" should be a string.");
        Py_RETURN_NONE;
    }

    CPythonUtils::UnRegisterPythonObject(name);
    Py_RETURN_NONE;
}

static PyObject* CPythonAgentGetContext(PyObject *module, PyObject *obj)
{
    CPythonUtils::PythonStringObject name(obj);
    if (name.IsNone()) {
        PyErr_SetString(PyExc_TypeError, "\"name\" should be a string.");
        Py_RETURN_NONE;
    }

    return CPythonUtils::GetRegisteredPyObj(name).NewRef();
}

PyDoc_STRVAR(RegisterDoc,
"register_context(name, obj)\n--\n\nRegister a python object, which will be available on the backend.");
PyDoc_STRVAR(UnregisterDoc,
"unregister_context(name)\n--\n\nUnregister a python object.");
PyDoc_STRVAR(GetDoc,
"get_context(name)\n--\n\nGet a python object, which may be register by the backend.");

static PyMethodDef CPythonAgentMethods[] = {
    {"register_context", reinterpret_cast<PyCFunction>(CPythonAgentRegister), METH_VARARGS, RegisterDoc},
    {"unregister_context", reinterpret_cast<PyCFunction>(CPythonAgentUnRegister), METH_O, UnregisterDoc},
    {"get_context", reinterpret_cast<PyCFunction>(CPythonAgentGetContext), METH_O, GetDoc},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef g_CPythonAgentModule = {
    PyModuleDef_HEAD_INIT,
    "_msprobe_c.CPythonAgent",     /* m_name */
    CPythonAgentModuleDoc,        /* m_doc */
    -1,                           /* m_size */
    CPythonAgentMethods,          /* m_methods */
};

PyObject* GetCPythonAgentModule()
{
    return PyModule_Create(&g_CPythonAgentModule);
}

}