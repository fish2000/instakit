
#include "pyimgc.h"
#include "numpypp/structcode.hpp"
#include <iostream>
#include <vector>
#include <string>
using namespace std;

#define PyImgC_STRUCTCODE_MODULE
#include "PyImgC_StructCodeAPI.h"

static PyObject *PyImgC_ParseStructCode(PyObject *self, PyObject *args) {
    char *structcode = None;

    if (!PyArg_ParseTuple(args, "s", &structcode)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot get structcode string (bad argument)");
        return NULL;
    }

    vector<pair<string, string>> pairvec = parse(string(structcode));
    string byteorder = "";

    if (!pairvec.size()) {
        PyErr_Format(PyExc_ValueError,
            "Struct typecode string %.200s parsed to zero-length pair vector",
            structcode);
        return NULL;
    }

    /// get special values
    for (size_t idx = 0; idx < pairvec.size(); idx++) {
        if (pairvec[idx].first == "__byteorder__") {
            byteorder = string(pairvec[idx].second);
            pairvec.erase(pairvec.begin()+idx);
        }
    }

    /// Make python list of tuples
    PyObject *list = PyList_New((Py_ssize_t)0);
    for (size_t idx = 0; idx < pairvec.size(); idx++) {
        PyList_Append(list,
            PyTuple_Pack((Py_ssize_t)2,
                PyString_InternFromString(string(pairvec[idx].first).c_str()),
                PyString_InternFromString(string(byteorder + pairvec[idx].second).c_str())));
    }

    return Py_BuildValue("O", list);
}

static PyObject *PyImgC_ParseSingleStructAtom(PyObject *self, PyObject *args) {
    char *structcode = None;

    if (!PyArg_ParseTuple(args, "s", &structcode)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot get structcode string (bad argument)");
        return NULL;
    }

    vector<pair<string, string>> pairvec = parse(string(structcode));
    string byteorder = "=";

    if (!pairvec.size()) {
        PyErr_Format(PyExc_ValueError,
            "Structcode string %.200s parsed to zero-length pair vector",
            structcode);
        return NULL;
    }

    /// get special values
    for (size_t idx = 0; idx < pairvec.size(); idx++) {
        if (pairvec[idx].first == "__byteorder__") {
            byteorder = string(pairvec[idx].second);
            pairvec.erase(pairvec.begin()+idx);
        }
    }

    /// Get singular value
    PyObject *dtypecode = PyString_InternFromString(
        string(byteorder + pairvec[0].second).c_str());

    return Py_BuildValue("O", dtypecode);
}

static int PyImgC_NPYCodeFromStructAtom(PyObject *self, PyObject *args) {
    PyObject *dtypecode = PyImgC_ParseSingleStructAtom(self, args);
    PyArray_Descr *descr;
    int npy_type_num = 0;

    if (!dtypecode) {
        PyErr_SetString(PyExc_ValueError,
            "cannot get structcode string (bad argument)");
        return -1;
    }

    if (!PyArray_DescrConverter(dtypecode, &descr)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot convert string to PyArray_Descr");
        return -1;
    }

    npy_type_num = (int)descr->type_num;
    Py_XDECREF(dtypecode);
    Py_XDECREF(descr);

    return npy_type_num;
}

static PyObject *PyImgC_NumpyCodeFromStructAtom(PyObject *self, PyObject *args) {
    return Py_BuildValue("i", PyImgC_NPYCodeFromStructAtom(self, args));
}

static PyMethodDef _structcode_methods[] = {
    {
        "parse",
            (PyCFunction)PyImgC_ParseStructCode,
            METH_VARARGS,
            "Parse struct code into list of dtype-string tuples"},
    {
        "parse_one",
            (PyCFunction)PyImgC_ParseSingleStructAtom,
            METH_VARARGS,
            "Parse unary struct code into a singular dtype string"},
    {
        "to_numpy_typenum",
            (PyCFunction)PyImgC_NumpyCodeFromStructAtom,
            METH_VARARGS,
            "Parse unary struct code into a NumPy typenum"},
    SENTINEL
};

PyMODINIT_FUNC init_structcode(void) {
    PyObject *module;
    static void *PyImgC_API[PyImgC_API_pointers];
    PyObject *api_ptr;

    module = Py_InitModule3(
        "_structcode",
        _structcode_methods,
        "PyImgC structcode decoder");
    if (module == None) { return; }
    
    /// set up PyCapsule API
    PyImgC_API[PyImgC_NPYCodeFromStructAtom_NUM] = (void *)PyImgC_NPYCodeFromStructAtom;
    api_ptr = PyCapsule_New((void *)PyImgC_API, "_structcode._C_API", NULL);
    if (api_ptr != NULL) {
        PyModule_AddObject(module, "_C_API", api_ptr);
    }
    
    /// bring in numpy
    import_array();
}




