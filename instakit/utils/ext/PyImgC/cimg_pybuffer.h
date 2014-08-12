
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

//////////////////// CONSTANTS4U ////////////////////
//#include "PyImgC_Constants.h"

//////////////////// EXTEND CIMG ////////////////////

// Check if this CImg<T> instance and a given PyBuffer have identical pixel types.
/*
bool not_pixel_type_of(const Py_buffer *const pybuffer) const {
    const char *format = (char *)pybuffer->format;
    const char format_key = format[0];
    
    return (
        0 != strcmp(*format_key, "b") && typeid(T) != typeid(char)) ||
        0 != strcmp(*format_key, "h") && typeid(T) != typeid(short)) ||
        0 != strcmp(*format_key, "i") && typeid(T) != typeid(int)) ||
        0 != strcmp(*format_key, "l") && typeid(T) != typeid(long)) ||
        0 != strcmp(*format_key, "q") && typeid(T) != typeid(long long)) ||

        0 != strcmp(*format_key, "B") && typeid(T) != typeid(unsigned char)) ||
        0 != strcmp(*format_key, "H") && typeid(T) != typeid(unsigned short)) ||
        0 != strcmp(*format_key, "I") && typeid(T) != typeid(unsigned int)) ||
        0 != strcmp(*format_key, "L") && typeid(T) != typeid(unsigned long)) ||
        0 != strcmp(*format_key, "Q") && typeid(T) != typeid(unsigned long long)) ||

        0 != strcmp(*format_key, "d") && typeid(T) != typeid(float)) ||
        0 != strcmp(*format_key, "g") && typeid(T) != typeid(double))
    );
}
*/

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

