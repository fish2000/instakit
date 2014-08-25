
#include "pyimgc.h"
#include "PyImgC_CImage.h"
using namespace cimg_library;


static PyObject *CImage_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Image *self;
    self = (Image *)type->tp_alloc(type, 0);
    if (self != None) {
        self->dtype = None;
        self->cimage = None;
    }
    return (PyObject *)self;
}

static int CImage_init(Image *self, PyObject *args, PyObject *kwargs) {
    PyObject *source=None, *dtype=None, *fake;
    //static char *keywords[] = { "source", "dtype", None };
    static char *keywords[] = {
        "shape", "dtype",
        "buffer", "offset", "strides",
        "order", NULL
    };

    PyArray_Descr *descr = NULL;
    int itemsize;
    PyArray_Dims dims = {NULL, 0};
    PyArray_Dims strides = {NULL, 0};
    PyArray_Chunk buffer;
    npy_longlong offset = 0;
    NPY_ORDER order = NPY_CORDER;
    int is_f_order = 0;
    //CImg<>

    if (!PyArg_ParseTupleAndKeywords(
                                        args, kwargs,
                                        "O&|O&O&LO&O&", keywords,
        PyArray_IntpConverter,          &dims,
        PyArray_DescrConverter,         &descr,
        PyArray_BufferConverter,        &buffer,
                                        &offset,
        &PyArray_IntpConverter,         &strides,
        &PyArray_OrderConverter,        &order)) {
            Py_XDECREF(dtype);
            return -1;
        }
    if (order == NPY_FORTRANORDER) {
        is_f_order = 1;
    }
    if (descr == NULL) {
        descr = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }


    return 0;
}

static void Image_dealloc(Image *self) {
    Py_XDECREF(self->buffer);
    Py_XDECREF(self->source);
    Py_XDECREF(self->dtype);
    self->ob_type->tp_free((PyObject *)self);
}

/*
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
*/
#define Image_members 0


static PyObject     *Image_GET_buffer(Image *self, void *closure) {
    BAIL_WITHOUT(self->buffer);
    Py_INCREF(self->buffer);
    return self->buffer;
}
static int           Image_SET_buffer(Image *self, PyObject *value, void *closure) {
    if (self->buffer) { Py_DECREF(self->buffer); }
    Py_INCREF(value);
    self->buffer = value;
    return 0;
}

static PyObject     *Image_GET_source(Image *self, void *closure) {
    BAIL_WITHOUT(self->source);
    Py_INCREF(self->source);
    return self->source;
}
static int           Image_SET_source(Image *self, PyObject *value, void *closure) {
    if (self->buffer) { Py_DECREF(self->source); }
    Py_INCREF(value);
    self->source = value;
    return 0;
}

static PyObject     *Image_GET_dtype(Image *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
    Py_INCREF(self->dtype);
    return self->dtype;
}
static int           Image_SET_dtype(Image *self, PyObject *value, void *closure) {
    if (self->buffer) { Py_DECREF(self->dtype); }
    Py_INCREF(value);
    self->dtype = value;
    return 0;
}

static PyGetSetDef Image_getset[] = {
    {
        "buffer",
            (getter)Image_GET_buffer,
            (setter)Image_SET_buffer,
            "Buffer or MemoryView", None},
    {
        "source",
            (getter)Image_GET_source,
            (setter)Image_SET_source,
            "Buffer Source Object", None},
    {
        "dtype",
            (getter)Image_GET_dtype,
            (setter)Image_SET_dtype,
            "Data Type (numpy.dtype)", None},
    SENTINEL
};

static rawbuffer_t *PyImgC_rawbuffer(PyObject *buffer) {

    rawbuffer_t *raw = (rawbuffer_t *)malloc(sizeof(rawbuffer_t));

    if (PyObject_CheckBuffer(buffer)) {
        /// buffer3000
        Py_buffer *buf = 0;
        PyObject_GetBuffer(buffer, buf, PyBUF_SIMPLE); BAIL_WITHOUT(buf);

        raw->len = buf->len;
        raw->buf = buf->buf;
        PyBuffer_Release(buf);

        return raw;
    } else if (PyBuffer_Check(buffer)) {
        /// legacybuf
        PyObject *bufferobj = PyBuffer_FromObject(buffer, (Py_ssize_t)0, Py_END_OF_BUFFER);
        const void *buf = 0;
        Py_ssize_t len;
        PyObject_AsReadBuffer(bufferobj, &buf, &len); BAIL_WITHOUT(buf);

        raw->buf = (void *)buf;
        raw->len = len;
        Py_XDECREF(bufferobj);

        return raw;
    }

    return None;
}

static PyObject *Image_as_ndarray(Image *self) {

    if (self->source && self->dtype) {
        rawbuffer_t *raw = PyImgC_rawbuffer(self->buffer);

        npy_intp *shape = &raw->len;
        PyArray_Descr *descr = 0;
        PyArray_DescrConverter(self->dtype, &descr); BAIL_WITHOUT(descr);

        int ndims = 1;
        int typenum = (int)descr->type_num;

        PyObject *ndarray = PyArray_SimpleNewFromData(
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
    sizeof(Image),                                              /* tp_basicsize */
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
    Image_getset,                                               /* tp_getset */
    0,                                                          /* tp_base */
    0,                                                          /* tp_dict */
    0,                                                          /* tp_descr_get */
    0,                                                          /* tp_descr_set */
    0,                                                          /* tp_dictoffset */
    (initproc)Image_init,                                       /* tp_init */
    0,                                                          /* tp_alloc */
    Image_new,                                                  /* tp_new */
};

static PyMethodDef _PyImgC_methods[] = {
    SENTINEL
};

PyMODINIT_FUNC init_PyImgC(void) {
    PyObject* module;

    if (PyType_Ready(&ImageType) < 0) { return; }

    module = Py_InitModule3(
        "_PyImgC",
        _PyImgC_methods,
        "PyImgC buffer interface module");

    if (module == None) {
        return;
    }

    import_array();

    Py_INCREF(&ImageType);
    PyModule_AddObject(
        module, "Image", (PyObject *)&ImageType);
}


