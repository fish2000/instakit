
#ifndef cimg_plugin_pybuffer
#define cimg_plugin_pybuffer
/// INSERT PLUGIN HERE

/// check for Python.h prior include
#ifndef Py_PYTHON_H
#error You need Python.h hash-included before PyImgC_PyBuffer.h (it's wonky I know)
#endif

#ifndef NDARRAYTYPES_H
    #ifndef Py_ARRAYOBJECT_H
    #error Numpy includes have to come before including CImg.h (it's wonky I know)
    #endif
#endif

/// check for CImg.h prior include
#ifndef cimg_version
#error You need to hash-define PyImgC_PyBuffer.h as a plugin and then include CImg.h (it's wonky I know)
#endif

/// conversion constructor
CImg(const Py_buffer *pybuffer):_width(0),_height(0),_depth(0),_spectrum(0),_is_shared(false),_data(0) {
    assign(pybuffer);
}

/// in-place constructor

template <typename YoDogg>
CImg<T>& assign(const Py_buffer *pybuffer) {
    if (!pybuffer) { return assign(); }
    int colordepth = (pybuffer->ndim > 2) ? pybuffer->shape[2] : 1;
    
    CImg<unsigned char>(
        pybuffer->buf,
        pybuffer->shape[1],
        pybuffer->shape[0],
        1,
        colordepth,
        True);

    return *this;
}


template <typename YoDogg>
CImg<T>& assign(const Py_buffer *pybuffer, bool wat) {
    if (!pybuffer) { return assign(); }
    int colordepth = (pybuffer->ndim > 2) ? pybuffer->shape[2] : 1;
    
    PyObject *memoryview = PyMemoryView_FromBuffer(pybuffer);
    if (!PyMemoryView_Check(structmod)) { printf("ERROR: BAD MEMORY VIEW"); }
    
    PyArrayObject *array = PyArray_FromBuffer(
        memoryview, None, /// dtype
        (npy_intp)pybuffer->itemsize,
        NPY_ARRAY_F_CONTIGUOUS);
    
    CImg<unsigned char>(
        pybuffer->buf,
        pybuffer->shape[1],
        pybuffer->shape[0],
        1,
        colordepth,
        True);

    return *this;
}



/// END OF PLUGIN IFDEF
#endif

