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

#pragma once

#include <Python.h>
#include <string>
#include <vector>
#include <map>
#include <cstddef>
#include <utility>
#include <stdexcept>

namespace MindStudioDebugger {
namespace  CPythonUtils {

/*
 * 对常用python类型做了cpp对接封装，对应关系为:
 * -------------------------------------------
 * |  python           |  cpp wrapper        |
 * |-----------------------------------------|
 * |  object           |  PythonObject       |
 * |  str              |  PythonStringObject |
 * |  int/float        |  PythonNumberObject |
 * |  bool             |  PythonBoolObject   |
 * |  list             |  PythonListObject   |
 * |  tuple            |  PythonTupleObject  |
 * |  dict             |  PythonDictObject   |
 * -------------------------------------------
 *
 * 创建对象的方式：
 * 1、通过原生PyObject*类型创建，PythonObject生命周期内会持有原生对象的一个引用
 * 2、通过From方法从c++对象创建
 * 3、通过GetGlobal、Import等方法从解释器上下文获取
 * 4、通过GetRegisteredPyObj获取到上下文的python对象
 * 5、通过已有PythonObject对象的Get、GetItem等方法获取子对象
 *
 * 对象转换：
 * 1、对于转换成PyObject*、bool、string的场景，支持隐式转换
 * 2、对于非通用类型转换，调用To方法，返回0表示成功
 * 3、对于list、tuple、dict类型，若元素类型都一样，支持直接转为vector/map类型，否则无法直接转换
 * 4、对于To方法:
 *     python维度支持bool()的都可以转bool（即并非只有bool类型支持转换，下同）
 *     支持str()的都可以转string
 *     可迭代对象(且元素支持转换)都可以转vector
 *
 * 对象传递：
 * 1、子类可以安全传递或拷贝给PythonObject对象
 * 2、PythonObject传给子类时，若类型匹配，可以安全转递，否则会转为None
 * 3、PythonObject或子类传递给需要PyObject*类型的cpthon原生接口时：
 *     若原生接口是接管参数型，需要传递NewRef()
 *     若原生接口是临时引用型，需要确保对象生命周期覆盖被调用的函数(不要原地构造临时对象)
 */

class PythonObject;
class PythonNumberObject;
class PythonStringObject;
class PythonBoolObject;
class PythonListObject;
class PythonTupleObject;
class PythonDictObject;

/* python侧使用_msprobe_c.CPythonAgent，cpp侧使用以下函数，进行python<--->cpp代码交互 */
int32_t RegisterPythonObject(const std::string& name, PythonObject obj);
void UnRegisterPythonObject(const std::string& name);
bool IsPyObjRegistered(const std::string& name);
PythonObject GetRegisteredPyObj(const std::string& name);

class PythonObject {
public:
    PythonObject()
    {
        Py_INCREF(Py_None);
        ptr = Py_None;
    }
    PythonObject(PyObject* o) : ptr(o)
    {
        if (ptr == nullptr) {
            ptr = Py_None;
        }
        Py_XINCREF(ptr);
    }
    ~PythonObject()
    {
        Py_XDECREF(ptr);
    }
    explicit PythonObject(const PythonObject &obj) : PythonObject(static_cast<PyObject*>(obj)) {}
    PythonObject& operator=(const PythonObject &obj)
    {
        SetPtr(static_cast<PyObject*>(obj));
        return *this;
    }

    /* 获取全局对象 */
    static PythonObject GetGlobal(const std::string& name, bool ignore = true);
    /* 获取模块对象；若其还未加载至缓存，则加载一遍 */
    static PythonObject Import(const std::string& name, bool ignore = true) noexcept;

    /* From/To转换，统一放一份在基类，用于遍历迭代器等场景 */
    static PythonObject From(const PythonObject& input);
    static PythonObject From(const int32_t& input);
    static PythonObject From(const uint32_t& input);
    static PythonObject From(const double& input);
    static PythonObject From(const std::string& input);
    static PythonObject From(const char* input);
    static PythonObject From(const bool& input);
    template <typename T>
    static PythonObject From(const std::vector<T>& input);
    template <typename T1, typename T2>
    static PythonObject From(const std::map<T1, T2>& input);
    int32_t To(int32_t& output) const;
    int32_t To(uint32_t& output) const;
    int32_t To(double& output) const;
    int32_t To(std::string& output) const;
    int32_t To(bool& output) const;
    template <typename T>
    int32_t To(std::vector<T>& output)const;

    bool IsNone() const {return ptr == Py_None;}
    bool IsNumber() const {return PyLong_Check(ptr) || PyFloat_Check(ptr);}
    bool IsString() const {return PyUnicode_Check(ptr);}
    bool IsBool() const {return PyBool_Check(ptr);}
    bool IsList() const {return PyList_Check(ptr);}
    bool IsTuple() const {return PyTuple_Check(ptr);}
    bool IsDict() const {return PyDict_Check(ptr);}
    bool IsModule() const {return PyModule_Check(ptr);}
    bool IsCallable() const {return PyCallable_Check(ptr);}

    /* 用于调用可调用对象，相当于python代码中的obj()，为了简单只实现了args+kwargs参数形式 */
    PythonObject Call(bool ignore = true) noexcept;
    PythonObject Call(PythonTupleObject& args, bool ignore = true) noexcept;
    PythonObject Call(PythonTupleObject& args, PythonDictObject& kwargs, bool ignore = true) noexcept;

    /* 用于获取对象属性，相当于python代码中的obj.xx */
    PythonObject Get(const std::string& name, bool ignore = true) const;
    PythonObject& NewRef()
    {
        Py_XINCREF(ptr);
        return *this;
    }
    std::string ToString() const
    {
        std::string ret;
        if (To(ret) == 0) {
            return ret;
        }
        return std::string();
    }

    operator PyObject*() const {return ptr;}
    operator bool() const {return static_cast<bool>(PyObject_IsTrue(ptr));}
    operator std::string() const
    {
        return ToString();
    }
    PythonObject operator()(bool ignore = true) {return Call(ignore);}
    PythonObject operator()(PythonTupleObject& args, bool ignore = true) {return Call(args, ignore);}
    PythonObject operator()(PythonTupleObject& args, PythonDictObject& kwargs, bool ignore = true)
    {
        return Call(args, kwargs, ignore);
    }

protected:
    void SetPtr(PyObject* o)
    {
        Py_XDECREF(ptr);
        if (o == nullptr) {
            o = Py_None;
        }
        Py_INCREF(o);
        ptr = o;
    }

    PyObject* ptr{nullptr};
    
private:
    explicit PythonObject(PythonObject &&obj) = delete;
    PythonObject& operator=(PythonObject &&obj) = delete;
};

class PythonNumberObject : public PythonObject {
public:
    PythonNumberObject();
    explicit PythonNumberObject(PyObject* o);

    static PythonNumberObject From(const int32_t& input);
    static PythonNumberObject From(const uint32_t& input);
    static PythonNumberObject From(const double& input);
};

class PythonStringObject : public PythonObject {
public:
    PythonStringObject();
    explicit PythonStringObject(PyObject* o);

    static PythonStringObject From(const std::string& input);
    static PythonStringObject From(const char* input);
};

class PythonBoolObject : public PythonObject {
public:
    PythonBoolObject();
    explicit PythonBoolObject(PyObject* o);

    static PythonBoolObject From(const bool& input);
};

class PythonListObject : public PythonObject {
public:
    PythonListObject();
    explicit PythonListObject(size_t size);
    explicit PythonListObject(PyObject* o);

    template <typename T>
    static PythonListObject From(const std::vector<T>& input);

    size_t Size() const;
    template <typename T>
    PythonListObject& Append(T value, bool ignore = true);
    PythonObject GetItem(size_t pos, bool ignore = true);
    PythonListObject& SetItem(size_t pos, PythonObject& item, bool ignore = true);
    PythonListObject& Insert(int64_t pos, PythonObject& item, bool ignore = true);
    PythonTupleObject ToTuple(bool ignore = true);
};

class PythonTupleObject : public PythonObject {
public:
    PythonTupleObject();
    explicit PythonTupleObject(PyObject* o);

    template <typename T>
    static PythonTupleObject From(const std::vector<T>& input);

    size_t Size() const;
    PythonObject GetItem(size_t pos, bool ignore = true);
};

class PythonDictObject : public PythonObject {
public:
    PythonDictObject();
    explicit PythonDictObject(PyObject* o);

    template <typename T1, typename T2>
    static PythonDictObject From(const std::map<T1, T2>& input);

    template <typename T1, typename T2>
    PythonDictObject& Add(T1 key, T2 value, bool ignore = true);
    template <typename T>
    PythonDictObject& Delete(T key, bool ignore = true);
    template <typename T>
    PythonObject GetItem(T key, bool ignore = true);
};

/**************************************************************************************************/
/**************************** 以下为模板函数的实现，调用者无需关注 ***********************************/
/**************************************************************************************************/
template <typename T>
PythonObject PythonObject::From(const std::vector<T>& input)
{
    return PythonListObject::From(input);
}

template <typename T1, typename T2>
PythonObject PythonObject::From(const std::map<T1, T2>& input)
{
    return PythonDictObject::From(input);
}

template <typename T>
int32_t PythonObject::To(std::vector<T>& output) const
{
    PyObject* item = nullptr;
    PyObject* iter = PyObject_GetIter(ptr);
    if (iter == nullptr) {
        return -1;
    }

    while ((item = PyIter_Next(iter)) != nullptr) {
        T tmp;
        if (PythonObject(item).To(tmp) != 0) {
            Py_DECREF(item);
            Py_DECREF(iter);
            return -1;
        }
        output.emplace_back(tmp);
        Py_DECREF(item);
    }

    Py_DECREF(iter);
    return 0;
}

template <typename T>
PythonListObject PythonListObject::From(const std::vector<T>& input)
{
    PyObject* o = PyList_New(input.size());
    if (o == nullptr) {
        return PythonListObject();
    }

    Py_ssize_t i = 0;
    for (const T& ele : input) {
        if (PyList_SetItem(o, i, PythonObject::From(ele).NewRef()) != 0) {
            Py_DECREF(o);
            return PythonListObject();
        }
        i++;
    }

    PythonListObject ret(o);
    Py_DECREF(o);
    return ret;
}

template <typename T>
PythonListObject& PythonListObject::Append(T value, bool ignore)
{
    if (!PyList_Check(ptr)) {
        if (!ignore) {
            PyErr_SetString(PyExc_TypeError, "Expect a list.");
        }
        return *this;
    }

    PythonObject o = PythonObject::From(value);
    PyList_Append(ptr, o);
    return *this;
}

template <typename T>
PythonTupleObject PythonTupleObject::From(const std::vector<T>& input)
{
    PyObject* o = PyTuple_New(input.size());
    if (o == nullptr) {
        return PythonTupleObject();
    }

    Py_ssize_t i = 0;

    for (const T& ele : input) {
        if (PyTuple_SetItem(o, i, PythonObject::From(ele).NewRef()) != 0) {
            Py_DECREF(o);
            return PythonTupleObject();
        }
        i++;
    }

    PythonTupleObject ret(o);
    Py_DECREF(o);
    return ret;
}

template <typename T1, typename T2>
PythonDictObject PythonDictObject::From(const std::map<T1, T2>& input)
{
    PyObject* o = PyDict_New();
    if (o == nullptr) {
        return PythonDictObject();
    }
    for (const std::pair<T1, T2>& pair : input) {
        PythonObject key = PythonObject::From(pair.first);
        PythonObject value = PythonObject::From(pair.second);
        if (PyDict_SetItem(o, key.NewRef(), value.NewRef()) != 0) {
            Py_DECREF(o);
            return PythonDictObject();
        }
    }

    PythonDictObject ret(o);
    Py_DECREF(o);
    return ret;
}

template <typename T1, typename T2>
PythonDictObject& PythonDictObject::Add(T1 key, T2 value, bool ignore)
{
    if (!PyDict_Check(ptr)) {
        if (!ignore) {
            PyErr_SetString(PyExc_TypeError, "Expect a dict.");
        }
        return *this;
    }

    if (PyDict_SetItem(ptr, PythonObject::From(key).NewRef(), PythonObject::From(value).NewRef()) != 0) {
        if (ignore) {
            PyErr_Clear();
        }
    }
    return *this;
}

template <typename T>
PythonDictObject& PythonDictObject::Delete(T key, bool ignore)
{
    if (!PyDict_Check(ptr)) {
        if (!ignore) {
            PyErr_SetString(PyExc_TypeError, "Expect a dict.");
        }
        return *this;
    }

    PythonObject o = PythonObject::From(key);
    if (PyDict_DelItem(ptr, o) != 0) {
        if (ignore) {
            PyErr_Clear();
        }
    }
    return *this;
}

template <typename T>
PythonObject PythonDictObject::GetItem(T key, bool ignore)
{
    if (!PyDict_Check(ptr)) {
        if (!ignore) {
            PyErr_SetString(PyExc_TypeError, "Expect a dict.");
        }
        return *this;
    }

    PythonObject o = PythonObject::From(key);
    PyObject* item = PyDict_GetItem(ptr, o);
    if (item == nullptr && ignore) {
        PyErr_Clear();
    }
    return PythonObject(item);
}

}
}
