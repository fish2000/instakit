#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

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
    PyObject *buffer;
    //Py_buffer *pybuffer;
    PyObject *source;
    PyArray_Descr *dtype;
} Image;

static PyObject *Image_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Image *self;
    self = (Image *)type->tp_alloc(type, 0);
    if (self != None) {
        self->buffer = None;
        //self->pybuffer = None;
        self->source = None;
        self->dtype = None;
    }
    return (PyObject *)self;
}

static int Image_init(Image *self, PyObject *args, PyObject *kwargs) {
    PyObject *source=None, *dtypeobj=None, *fake;
    static char *keywords[] = { "source", "dtype", None };
    
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "|OO",
        keywords,
        &source, &dtypeobj)) { return -1; }
        
#if PY_VERSION_HEX <= 0x03000000
        /// Try the legacy buffer interface while it's here
        if (PyObject_CheckReadBuffer(source)) {
            self->buffer = PyBuffer_FromObject(source, (Py_ssize_t)0, Py_END_OF_BUFFER);
            goto through;
#ifdef PYIMGC_DEBUG
        } else {
            printf("YO DOGG: legacy buffer check failed");
#endif
        }
#endif
        /// In the year 3000 the old ways are long gone
        if (PyObject_CheckBuffer(source)) {
            self->buffer = PyMemoryView_FromObject(source);
            goto through;
#ifdef PYIMGC_DEBUG
        } else {
            printf("YO DOGG: buffer3000 check failed");
#endif
        }
        
        /// return before 'through' cuz IT DIDNT WORK DAMNIT
        return 0;
        
    through:
#ifdef PYIMGC_DEBUG
        printf("YO DOGG WERE THROUGH");
#endif
        fake = self->source;        Py_INCREF(source);
        self->source = source;      Py_XDECREF(fake);
    }
    
    if ((source && !self->source) || source != self->source) {
        PyArray_Descr *dtype;
        if (!dtypeobj && PyArray_Check(source)) {
            dtype = ((PyArrayObject *)source)->descr;
        } else if (dtypeobj && !self->dtype) {
            if (!PyArray_DescrConverter(dtypeobj, &dtype)) {
                printf("Couldn't convert dtype arg");
            }
            Py_DECREF(dtypeobj);
        }
    
    if ((dtype && !self->dtype) || dtype != self->dtype) {
        fake = self->dtype;         Py_INCREF(dtype);
        self->dtype = dtype;        Py_XDECREF(fake);
    }
    
    return 0;
}

static void Image_dealloc(Image *self) {
    Py_XDECREF(self->buffer);
    Py_XDECREF(self->source);
    Py_XDECREF(self->dtype);
    self->ob_type->tp_free((PyObject *)self);
}

static PyMemberDef Image_members[] = {
    {
        "buffer", T_OBJECT_EX,
            offsetof(Image, buffer), 0,
            "Buffer or MemoryView"},
    {
        "source", T_OBJECT_EX,
            offsetof(Image, source), 0,
            "Buffer Source Object"},
    {
        "dtype", T_OBJECT_EX,
            offsetof(Image, dtype), 0,
            "Data Type (numpy.dtype)"},
    SENTINEL
};

typedef struct {
    Py_ssize_t len;
    void *buf;
} rawbuffer_t;

static void *PyImgC_rawbuffer(PyObject *buffer) {
    rawbuffer_t *raw = (rawbuffer_t)malloc(sizeof(rawbuffer_t));
    if (PyObject_CheckBuffer(buffer)) {
        /// buffer3000
        Py_buffer *buf = PyObject_GetBuffer(buffer);
        raw->len = buf->len;
        raw->buf = buf->buf;
        PyBuffer_Release(buf);
        return raw;
    } else if (PyBuffer_Check(buffer)) {
        /// legacybuf
        PyObject *bufferobj = PyBuffer_FromObject(buffer, (Py_ssize_t)0, Py_END_OF_BUFFER);
        const void *buf;
        Py_ssize_t len;
        PyObject_AsReadBuffer(bufferobj, &buf, &len);
        raw->buf = buf;
        raw->len = len;
        Py_XDECREF(bufferobj);
        return raw;
    }
    return None;
}

static PyObject *Image_as_ndarray(Image *self) {
    if (self->source && self->dtype) {
        rawbuffer_t *raw = PyImgC_rawbuffer(self->buffer);
        int ndims = 1;
        int *shape = { raw->len, };
        int typenum = (int)self->dtype->type_num;
        //Py_ssize_t *dims = { PySequence_Length(self->buffer), };
        
        PyArrayObject *ndarray = PyArray_SimpleNewFromData(
            ndims, shape, typenum, raw->buf);
        Py_INCREF(ndarray);
        return (PyObject *)ndarray;
    }
    return None;
}

static PyMethodDef Image_methods[] = {
    {
        "as_ndarray",
            (PyCFunction)Image_as_ndarray,
            METH_NOARGS,
            "Cast to NumPy array"},
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


