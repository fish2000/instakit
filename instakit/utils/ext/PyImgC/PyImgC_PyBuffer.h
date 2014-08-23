#ifndef PyImgC_PYBUFFER_H
#define PyImgC_PYBUFFER_H
/// INSERT PYTHON C-API STUFF HERE

#include <Python.h>
#include "numpypp/numpy.hpp"



//////////////////// CREATE AN IMAGEVIEW //////////////////
/// CImg<decoder<NPY_UBYTE>::type> image(...);
/// PyArray_DESCR() -> (PyArray_Descr *)descr_struct
/// PyArray_BYTES() | PyArray_DATA() [?]
/// PyArray_NDIM(a) -> len(a.shape)
/// PyArray_STRIDES() -> *strides
/// PyArray_DIM(a, dim) -> shape of a at `dim`
/// PyArray_SHAPE(a) | PyArray_DIMS(a) -> (npy_intp *)a.shape
/// PyArray_TYPE(a) -> NPY_UBYTE (and friends)
/// PyArray_SIZE(a) -> len(a.flatten())
/// PyArray_NBYTES(a) ~> sizeof(a)
/// PyArray_GetPtr(a, dim) -> (void *)a.data

template <typename T, typename shape>
CImg<T> PyImgC_ImageFromBuffer(Py_buffer *buffer) {
    return CImg<>
}





template <typename DTYPE>
CImg<typename no_ptr<T>::type>

template <> inline
CImg<type> cimage_view<constant>() {
    return CImg<type>;
}

int sX = 640;
int sY = 480;
int channels = 3;
CImg<unsigned char> view(charbuffer, sX, sY, 1, channels, is_shared=True);



//////////////////// CREATE A PYBUFFER ////////////////////

template <typename T>
Py_buffer *PyImgC_BufferFromImage(CImg<T>& cimage) {
    int was_buffer_filled = -2;
    Py_buffer pybuffer;
    Py_ssize_t raw_buffer_size = (Py_ssize_t)(
        cimage._width * cimage._height * cimage._spectrum * sizeof(T)); /// our cimage._depth should be zero
    
    was_buffer_filled = PyBuffer_FillInfo(
            &pybuffer, None,                                /// Output struct ref, and null PyObject ptr
            (T*)cimage._data,                               /// Input raw-data ptr
            raw_buffer_size,                                /// Size of *_data in bytes 
            True,                                           /// Buffer is read-only
            PyBUF_F_CONTIGUOUS);                            /// I *think* CImg instances are fortran-style planar
    
    if (was_buffer_filled > -1) {
        return pybuffer;
    }
    
    throw CImgInstanceException(
        "PyBuffer_FillInfo() returned %i (which is wack)",
        was_buffer_filled);
}

//////////////////// CREATE AN NDARRAY ////////////////////

template <typename T>
Py_buffer *PyImgC_ArrayFromImage(CImg<T>& cimage) {
    int was_buffer_filled = -2;
    Py_buffer pybuffer;
    Py_ssize_t raw_buffer_size = (Py_ssize_t)(
        cimage._width * cimage._height * cimage._spectrum * sizeof(T)); /// our cimage._depth should be zero
    
    was_buffer_filled = PyArray_SimpleNewFromData(
            &pybuffer, None,                                /// Output struct ref, and null PyObject ptr
            (T*)cimage._data,                               /// Input raw-data ptr
            raw_buffer_size,                                /// Size of *_data in bytes 
            True,                                           /// Buffer is read-only
            PyBUF_F_CONTIGUOUS);                            /// I *think* CImg instances are fortran-style planar
    
    if (was_buffer_filled > -1) {
        return pybuffer;
    }
    
    throw CImgInstanceException(
        "PyArray_SimpleNewFromData() returned %i (which is wack)",
        was_buffer_filled);
}




/// END OF PYTHON C-API STUFF'S LIMINAL HASH-IFNDEF
#endif