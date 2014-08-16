#include <Python.h>
#include <structmember.h>

/// lil' bit pythonic
#ifndef False
#define False 0
#endif
#ifndef True
#define True 1
#endif
#ifndef None
#define None NULL
#endif

/// UGH
#ifndef SENTINEL
#define SENTINEL {NULL}
#endif
#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif



typedef struct {
    PyObject_HEAD
    PyObject *oldbuf;
    Py_buffer *newbuf;
    PyObject *source;
    PyObject *dtype;
} Image;

static PyObject *Image_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Image *self;
    self = (Image *)type->tp_alloc(type, 0);
    if (self != None) {
        self->oldbuf = None;
        self->newbuf = None;
        self->source = None;
        self->dtype = None;
    }
    return (PyObject *)self;
}

static int Image_init(Image *self, PyObject *args, PyObject *kwargs) {
    PyObject *source=None, *dtype=None, *fake;
    static char *keywords[] = { "source", "dtype", None };
    
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "|OO",
        keywords,
        &source, &dtype)) { return -1; }
    
    if (!self->source || source != self->source) {
        
#if PY_VERSION_HEX <= 0x03000000
        /// Try the legacy buffer interface while it's here
        if (PyBuffer_Check(source)) {
            PyBuffer_FromObject(source, (Py_ssize_t)0, Py_END_OF_BUFFER);
            goto through;
        } else {
            printf("YO DOGG: legacy buffer check failed");
        }
#endif
        /// In the year 3000 the old ways are long gone
        if (PyObject_CheckBuffer(source)) {
            PyMemoryView_FromObject(source);
            goto through;
        } else {
            printf("YO DOGG: buffer3000 check failed");
        }
        
        /// return before 'through' cuz IT DIDNT WORK DAMNIT
        return 0;
        
    through:
        printf("YO DOGG WERE THROUGH");
        fake = self->source;        Py_INCREF(source);
        self->source = source;      Py_XDECREF(fake);
    }
    
    if (!self->dtype || dtype != self->dtype) {
        fake = self->dtype;         Py_INCREF(dtype);
        self->dtype = dtype;        Py_XDECREF(fake);
    }
    
    return 0;
}

static void Image_dealloc(Image *self) {
    Py_XDECREF(self->memview);
    Py_XDECREF(self->dtype);
    self->ob_type->tp_free((PyObject *)self);
}

static PyMemberDef Image_members[] = {
    {
        "memview", T_OBJECT_EX,
            offsetof(Image, memview), 0,
            "Memory View"},
    {
        "dtype", T_OBJECT_EX,
            offsetof(Image, dtype), 0,
            "Data Type (numpy.dtype)"},
    SENTINEL
};

static PyMethodDef Image_methods[] = {
    SENTINEL
};

static Py_ssize_t Image_TypeFlags = Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE |
    Py_TPFLAGS_HAVE_GETCHARBUFFER;

static PyTypeObject ImageType = {
    PyObject_HEAD_INIT(NULL)
    0,                                                          /* ob_size */
    "PyImgC.Image",                                             /* tp_name */
    sizeof(PyImgC_Image),                                       /* tp_basicsize */
    0,                                                          /* tp_itemsize */
    (destructor)Image_dealloc,                                  /* tp_dealloc */
    0,                                                          /* tp_print */
    0,                                                          /* tp_getattr */
    0,                                                          /* tp_setattr */
    0,                                                          /* tp_compare */
    0,                                                          /* tp_repr */
    0,                                                          /* tp_as_number */
    0,                                                          /* tp_as_sequence */
    0,                                                          /* tp_as_mapping */
    0,                                                          /* tp_hash */
    0,                                                          /* tp_call */
    0,                                                          /* tp_str */
    0,                                                          /* tp_getattro */
    0,                                                          /* tp_setattro */
    0,                                                          /* tp_as_buffer */
    Image_TypeFlags,                                            /* tp_flags*/
    "PyImgC object wrapper for CImg instances",                 /* tp_doc */
    0,                                                          /* tp_traverse */
    0,                                                          /* tp_clear */
    0,                                                          /* tp_richcompare */
    0,                                                          /* tp_weaklistoffset */
    0,                                                          /* tp_iter */
    0,                                                          /* tp_iternext */
    Image_methods,                                              /* tp_methods */
    Image_members,                                              /* tp_members */
    0,                                                          /* tp_getset */
    0,                                                          /* tp_base */
    0,                                                          /* tp_dict */
    0,                                                          /* tp_descr_get */
    0,                                                          /* tp_descr_set */
    0,                                                          /* tp_dictoffset */
    (initproc)Image_init,                                       /* tp_init */
    0,                                                          /* tp_alloc */
    Image_new,                                                  /* tp_new */
};

static PyMethodDef PyImgC_methods[] = {
    SENTINEL
};

PyMODINIT_FUNC initPyImgC(void) {
    PyObject* module;
    
    //ImageType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ImageType) < 0) { return; }

    module = Py_InitModule3(
        "PyImgC",
        PyImgC_methods,
        "PyImgC buffer interface module");

    import_array();

    Py_INCREF(&ImageType);
    PyModule_AddObject(
        module, "Image",
        (PyObject *)&ImageType);
}


