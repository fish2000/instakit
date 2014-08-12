#ifndef PyImgC_PYBUFFER_H
#define PyImgC_PYBUFFER_H
/// INSERT PYTHON C-API STUFF HERE

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