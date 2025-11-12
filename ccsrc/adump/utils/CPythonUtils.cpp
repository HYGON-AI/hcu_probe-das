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

#include <cstring>
#include <string>
#include <map>

#include "CPythonUtils.h"

namespace MindStudioDebugger {
namespace  CPythonUtils {

static std::map<std::string, PythonObject> PyObjMap = {};

int32_t RegisterPythonObject(const std::string& name, PythonObject obj)
{
    if (PyObjMap.find(name) != PyObjMap.end()) {
        return -1;
    }

    PyObjMap[name] = obj;
    return 0;
}

void UnRegisterPythonObject(const std::string& name)
{
    auto it = PyObjMap.find(name);
    if (it == PyObjMap.end()) {
        return;
    }

    PyObjMap.erase(it);
}

bool IsPyObjRegistered(const std::string& name)
{
    return PyObjMap.find(name) != PyObjMap.end();
}

PythonObject GetRegisteredPyObj(const std::string& name)
{
    auto it = PyObjMap.find(name);
    if (it == PyObjMap.end()) {
        return PythonObject();
    }
    return it->second;
}

PythonObject PythonObject::From(const PythonObject& input)
{
    return PythonObject(input);
}

PythonObject PythonObject::From(const int32_t& input)
{
    return PythonNumberObject::From(input);
}

PythonObject PythonObject::From(const uint32_t& input)
{
    return PythonNumberObject::From(input);
}

PythonObject PythonObject::From(const double& input)
{
    return PythonNumberObject::From(input);
}
PythonObject PythonObject::From(const std::string& input)
{
    return PythonStringObject::From(input);
}

PythonObject PythonObject::From(const char* input)
{
    return PythonStringObject::From(input);
}

PythonObject PythonObject::From(const bool& input)
{
    return PythonBoolObject::From(input);
}

int32_t PythonObject::To(int32_t& output) const
{
    if (!PyLong_Check(ptr)) {
        return -1;
    }
    output = static_cast<int32_t>(PyLong_AsLong(ptr));
    return 0;
}

int32_t PythonObject::To(uint32_t& output) const
{
    if (!PyLong_Check(ptr)) {
        return -1;
    }
    output = static_cast<uint32_t>(PyLong_AsUnsignedLong(ptr));
    return 0;
}

int32_t PythonObject::To(double& output) const
{
    if (!PyFloat_Check(ptr)) {
        return -1;
    }

    output = PyFloat_AsDouble(ptr);
    return 0;
}

int32_t PythonObject::To(std::string& output) const
{
    PyObject* strObj = PyObject_Str(ptr);
    if (strObj == nullptr) {
        return -1;
    }
    const char* s = PyUnicode_AsUTF8(strObj);
    if (s == nullptr) {
        Py_DECREF(strObj);
        return -1;
    }
    output = std::string(s);
    Py_DECREF(strObj);
    return 0;
}

int32_t PythonObject::To(bool& output) const
{
    output = static_cast<bool>(PyObject_IsTrue(ptr));
    return 0;
}

PythonObject PythonObject::Get(const std::string& name, bool ignore) const
{
    PyObject* o = PyObject_GetAttrString(ptr, name.c_str());
    if (o == nullptr && ignore) {
        PyErr_Clear();
    }
    PythonObject ret(o);
    Py_XDECREF(o);
    return ret;
}

PythonObject PythonObject::Call(bool ignore) noexcept
{
    if (!PyCallable_Check(ptr)) {
        if (!ignore) {
            PyErr_SetString(PyExc_TypeError, "Object is not callable.");
        }
        return PythonObject();
    }

    PyObject* o = PyObject_CallObject(ptr, nullptr);
    if (o == nullptr && ignore) {
        PyErr_Clear();
    }
    PythonObject ret(o);
    Py_XDECREF(o);
    return ret;
}

PythonObject PythonObject::Call(PythonTupleObject& args, bool ignore) noexcept
{
    if (!PyCallable_Check(ptr)) {
        if (!ignore) {
            PyErr_SetString(PyExc_TypeError, "Object is not callable.");
        }
        return PythonObject();
    }

    PyObject* o = PyObject_CallObject(ptr, args.IsNone() ? nullptr : reinterpret_cast<PyObject*>(&args));
    if (o == nullptr && ignore) {
        PyErr_Clear();
    }
    PythonObject ret(o);
    Py_XDECREF(o);
    return ret;
}

PythonObject PythonObject::Call(PythonTupleObject& args, PythonDictObject& kwargs, bool ignore) noexcept
{
    if (!PyCallable_Check(ptr)) {
        if (!ignore) {
            PyErr_SetString(PyExc_TypeError, "Object is not callable.");
        }
        return PythonObject();
    }

    if (args.IsNone() || kwargs.IsNone()) {
        if (!ignore) {
            PyErr_SetString(PyExc_TypeError, "Call python object with invalid parameters.");
        }
        return PythonObject();
    }

    PyObject* o = PyObject_Call(ptr, args, kwargs);
    if (o == nullptr && ignore) {
        PyErr_Clear();
    }
    PythonObject ret(o);
    Py_XDECREF(o);
    return ret;
}

PythonObject PythonObject::GetGlobal(const std::string& name, bool ignore)
{
    PyObject *globals = PyEval_GetGlobals();
    if (globals == nullptr) {
        if (ignore) {
            PyErr_Clear();
        }
        return PythonObject();
    }

    return PythonObject(PyDict_GetItemString(globals, name.c_str()));
}

PythonObject PythonObject::Import(const std::string& name, bool ignore) noexcept
{
    PyObject* m = PyImport_ImportModule(name.c_str());
    if (m == nullptr) {
        if (ignore) {
            PyErr_Clear();
        }
        return PythonObject();
    }
    PythonObject ret(m);
    Py_XDECREF(m);
    return ret;
}

PythonNumberObject::PythonNumberObject() : PythonObject()
{
    PyObject* o = PyLong_FromLong(0);
    SetPtr(o);
    Py_XDECREF(o);
}

PythonNumberObject::PythonNumberObject(PyObject* o) : PythonObject()
{
    if (!PyLong_Check(o) && !PyFloat_Check(o)) {
        return;
    }

    SetPtr(o);
}

PythonNumberObject PythonNumberObject::From(const int32_t& input)
{
    PythonNumberObject ret;
    PyObject* o = PyLong_FromLong(input);
    if (o == nullptr) {
        return ret;
    }
    ret.SetPtr(o);
    Py_DECREF(o);
    return ret;
}

PythonNumberObject PythonNumberObject::From(const uint32_t& input)
{
    PythonNumberObject ret;
    PyObject* o = PyLong_FromUnsignedLong(input);
    if (o == nullptr) {
        return ret;
    }
    ret.SetPtr(o);
    Py_DECREF(o);
    return ret;
}

PythonNumberObject PythonNumberObject::From(const double& input)
{
    PythonNumberObject ret;
    PyObject* o = PyFloat_FromDouble(input);
    if (o == nullptr) {
        return ret;
    }
    ret.SetPtr(o);
    Py_DECREF(o);
    return ret;
}

PythonStringObject::PythonStringObject() : PythonObject()
{
    PyObject* o = PyUnicode_FromString("");
    SetPtr(o);
    Py_XDECREF(o);
}

PythonStringObject::PythonStringObject(PyObject* o) : PythonObject()
{
    if (!PyUnicode_Check(o)) {
        return;
    }

    SetPtr(o);
}

PythonStringObject PythonStringObject::From(const std::string& input)
{
    PythonStringObject ret;
    PyObject* o = PyUnicode_FromString(input.c_str());
    if (o == nullptr) {
        return ret;
    }
    ret.SetPtr(o);
    Py_DECREF(o);
    return ret;
}

PythonStringObject PythonStringObject::From(const char* input)
{
    PythonStringObject ret;
    PyObject* o = PyUnicode_FromString(input);
    if (o == nullptr) {
        return ret;
    }
    ret.SetPtr(o);
    Py_DECREF(o);
    return ret;
}

PythonBoolObject::PythonBoolObject() : PythonObject()
{
    SetPtr(Py_False);
}

PythonBoolObject::PythonBoolObject(PyObject* o) : PythonObject()
{
    if (!PyBool_Check(o)) {
        return;
    }

    SetPtr(o);
}

PythonBoolObject PythonBoolObject::From(const bool& input)
{
    PythonBoolObject ret;
    PyObject* o = PyBool_FromLong(input);
    if (o == nullptr) {
        return ret;
    }
    ret.SetPtr(o);
    Py_DECREF(o);
    return ret;
}

PythonListObject::PythonListObject() : PythonObject()
{
    PyObject* o = PyList_New(0);
    SetPtr(o);
    Py_XDECREF(o);
}

PythonListObject::PythonListObject(size_t size) : PythonObject()
{
    PyObject* o = PyList_New(size);
    SetPtr(o);
    Py_XDECREF(o);
}

PythonListObject::PythonListObject(PyObject* o) : PythonObject()
{
    if (!PyList_Check(o)) {
        return;
    }

    SetPtr(o);
}

size_t PythonListObject::Size() const
{
    if (!PyList_Check(ptr)) {
        return 0;
    }

    return PyList_GET_SIZE(ptr);
}

PythonObject PythonListObject::GetItem(size_t pos, bool ignore)
{
    if (!PyList_Check(ptr)) {
        if (!ignore) {
            PyErr_SetString(PyExc_TypeError, "Expect a list.");
        }
        return PythonObject();
    }
    if (static_cast<size_t>(PyList_GET_SIZE(ptr)) <= pos) {
        if (!ignore) {
            PyErr_SetString(PyExc_IndexError, "list index outof range");
        }
        return PythonObject();
    }

    PyObject* o = PyList_GetItem(ptr, pos);
    if (o == nullptr && ignore) {
        PyErr_Clear();
    }

    return PythonObject(o);
}

PythonListObject& PythonListObject::SetItem(size_t pos, PythonObject& item, bool ignore)
{
    if (!PyList_Check(ptr)) {
        if (!ignore) {
            PyErr_SetString(PyExc_TypeError, "Expect a list.");
        }
        return *this;
    }

    if (static_cast<size_t>(PyList_GET_SIZE(ptr)) <= pos) {
        if (!ignore) {
            PyErr_SetString(PyExc_IndexError, "list index outof range");
        }
        return *this;
    }

    if (PyList_SetItem(ptr, pos, item.NewRef()) != 0) {
        if (ignore) {
            PyErr_Clear();
        }
    }
    return *this;
}

PythonListObject& PythonListObject::Insert(int64_t pos, PythonObject& item, bool ignore)
{
    if (!PyList_Check(ptr)) {
        if (!ignore) {
            PyErr_SetString(PyExc_TypeError, "Expect a list.");
        }
        return *this;
    }

    if (PyList_Insert(ptr, pos, item) != 0) {
        if (ignore) {
            PyErr_Clear();
        }
    }

    return *this;
}

PythonTupleObject PythonListObject::ToTuple(bool ignore)
{
    if (!PyList_Check(ptr)) {
        return PythonTupleObject();
    }

    PyObject* o = PyList_AsTuple(ptr);
    if (o == nullptr && ignore) {
        PyErr_Clear();
    }
    PythonTupleObject ret(o);
    Py_XDECREF(o);
    return ret;
}

PythonTupleObject::PythonTupleObject() : PythonObject()
{
    PyObject* o = PyTuple_New(0);
    SetPtr(o);
    Py_XDECREF(o);
}

PythonTupleObject::PythonTupleObject(PyObject* o) : PythonObject()
{
    if (!o || !PyTuple_Check(o)) {
        return;
    }

    SetPtr(o);
}

size_t PythonTupleObject::Size() const
{
    if (!PyTuple_Check(ptr)) {
        return 0;
    }

    return PyTuple_GET_SIZE(ptr);
}

PythonObject PythonTupleObject::GetItem(size_t pos, bool ignore)
{
    if (!PyTuple_Check(ptr)) {
        if (!ignore) {
            PyErr_SetString(PyExc_TypeError, "Expect a tuple.");
        }
        return PythonObject();
    }
    if (static_cast<size_t>(PyTuple_GET_SIZE(ptr)) <= pos) {
        if (!ignore) {
            PyErr_SetString(PyExc_IndexError, "tuple index outof range");
        }
        return PythonObject();
    }

    PyObject* o = PyTuple_GetItem(ptr, pos);
    if (o == nullptr && ignore) {
        PyErr_Clear();
    }

    return PythonObject(o);
}

PythonDictObject::PythonDictObject() : PythonObject()
{
    PyObject* o = PyDict_New();
    SetPtr(o);
    Py_XDECREF(o);
}

PythonDictObject::PythonDictObject(PyObject* o) : PythonObject()
{
    if (!PyDict_Check(o)) {
        return;
    }

    SetPtr(o);
}

}
}