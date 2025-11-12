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

#include <stdexcept>
#include <exception>
#include <cstring>

#include "utils/CPythonUtils.h"
#include "core/PrecisionDebugger.h"

namespace MindStudioDebugger {

static PyObject* NewPrecisionDebugger(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (type == nullptr || type->tp_alloc == nullptr) {
        throw std::runtime_error("PrecisionDebugger: type or alloc is nullptr.");
    }

    /* 单例,减少重复构造 */
    static PyObject *self = nullptr;
    if (self == nullptr) {
        self = type->tp_alloc(type, 0);
    }

    Py_XINCREF(self);
    return self;
}

static int InitPrecisionDebugger(PyObject *self, PyObject *args, PyObject *kws)
{
    if (PrecisionDebugger::GetInstance().HasInitialized()) {
        return 0;
    }

    if (kws == nullptr) {
        PyErr_SetString(PyExc_TypeError, "Need keywords arg'framework\'and \'config_path\'.");
        return -1;
    }

    CPythonUtils::PythonDictObject kwArgs(kws);
    std::string framework = kwArgs.GetItem("framework");
    std::string cfgFile = kwArgs.GetItem("config_path");
    if (PrecisionDebugger::GetInstance().Initialize(framework, cfgFile) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to load config, read log for more details.");
        return -1;
    }

    return 0;
}

static PyObject* PrecisionDebuggerGetAttr(PyObject *self, PyObject *name)
{
    CPythonUtils::PythonStringObject attr(name);
    
    if (attr.IsNone()) {
        PyErr_SetString(PyExc_TypeError, "Attribution should be a string.");
        Py_RETURN_NONE;
    }

    const char* s = attr.ToString().c_str();
    if (strcmp(s, "enable") == 0) {
        return CPythonUtils::PythonObject::From(PrecisionDebugger::GetInstance().IsEnable()).NewRef();
    } else if (strcmp(s, "current_step") == 0) {
        return CPythonUtils::PythonObject::From(PrecisionDebugger::GetInstance().GetCurStep()).NewRef();
    }
    
    PyObject* ret =  PyObject_GenericGetAttr(self, name);
    if (ret == nullptr) {
        PyErr_Format(PyExc_AttributeError, "\'PrecisionDebugger\' object has no attribute \'%s\'", attr);
        Py_RETURN_NONE;
    }

    return ret;
}

static PyObject* PrecisionDebuggerStart(PyObject *self)
{
    PrecisionDebugger::GetInstance().Start();
    Py_RETURN_NONE;
}

static PyObject* PrecisionDebuggerStop(PyObject *self)
{
    PrecisionDebugger::GetInstance().Stop();
    Py_RETURN_NONE;
}

static PyObject* PrecisionDebuggerStep(PyObject *self)
{
    PrecisionDebugger::GetInstance().Step();
    Py_RETURN_NONE;
}

PyDoc_STRVAR(StartDoc,
"start($self, /)\n--\n\nEnable debug.");
PyDoc_STRVAR(StopDoc,
"stop($self, /)\n--\n\nDisable debug.");
PyDoc_STRVAR(StepDoc,
"step($self, [increment])\n--\n\nUpdata step.");

static PyMethodDef PrecisionDebuggerMethods[] = {
    {"start", reinterpret_cast<PyCFunction>(PrecisionDebuggerStart), METH_NOARGS, StartDoc},
    {"stop", reinterpret_cast<PyCFunction>(PrecisionDebuggerStop), METH_NOARGS, StopDoc},
    {"step", reinterpret_cast<PyCFunction>(PrecisionDebuggerStep), METH_NOARGS, StepDoc},
    {nullptr, nullptr, 0, nullptr}
};

PyTypeObject PyPrecisionDebuggerType = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "_msprobe_c._PrecisionDebugger",             /* tp_name */
    0,                                          /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    PrecisionDebuggerGetAttr,                   /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    PrecisionDebuggerMethods,                   /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    &PyBaseObject_Type,                         /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    InitPrecisionDebugger,                      /* tp_init */
    0,                                          /* tp_alloc */
    NewPrecisionDebugger,                       /* tp_new */
    PyObject_Del,                               /* tp_free */
};

PyTypeObject* GetPyPrecisionDebuggerType()
{
    static bool init = false;
    if (!init) {
        if (PyType_Ready(&PyPrecisionDebuggerType) < 0) {
            return nullptr;
        }
        init = true;
    }
    return &PyPrecisionDebuggerType;
}
}