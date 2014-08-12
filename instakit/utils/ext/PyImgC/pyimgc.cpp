#include <Python.h>
#include "PyImgC_Interface.h"

typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
} PyImgC_Image;

static PyTypeObject PyImgC_ImageType = {
    PyObject_HEAD_INIT(NULL)
    0,                                                          /*ob_size*/
    "PyImgC.Image",                                             /*tp_name*/
    sizeof(PyImgC_Image),                                       /*tp_basicsize*/
    0,                                                          /*tp_itemsize*/
    0,                                                          /*tp_dealloc*/
    0,                                                          /*tp_print*/
    0,                                                          /*tp_getattr*/
    0,                                                          /*tp_setattr*/
    0,                                                          /*tp_compare*/
    0,                                                          /*tp_repr*/
    0,                                                          /*tp_as_number*/
    0,                                                          /*tp_as_sequence*/
    0,                                                          /*tp_as_mapping*/
    0,                                                          /*tp_hash */
    0,                                                          /*tp_call*/
    0,                                                          /*tp_str*/
    0,                                                          /*tp_getattro*/
    0,                                                          /*tp_setattro*/
    0,                                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,                                         /*tp_flags*/
    "PyImgC object wrapper for CImg instances",                 /* tp_doc */
};

static PyMethodDef PyImgC_methods[] = {
    { NULL }
};

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
initPyImgC(void) {
    PyObject* module;
    PyImgC_ImageType.tp_new = PyType_GenericNew;
    
    if (PyType_Ready(&PyImgC_ImageType) < 0) { return; }

    module = Py_InitModule3(
        "PyImgC",
        PyImgC_methods,
        "PyImgC buffer interface module");

    Py_INCREF(&PyImgC_ImageType);
    PyModule_AddObject(module, "Image", (PyObject *)&PyImgC_ImageType);
}