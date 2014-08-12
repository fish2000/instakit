
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

//////////////////// EXTEND CIMG ////////////////////

// Check if this CImg<T> instance and a given PyBuffer have identical pixel types.
bool not_pixel_type_of(const Py_buffer *const pybuffer) const {
    const char *const format = (char *)pybuffer->format;
    const char format_key = (char)format[0];
    
    return (
        format_key == "b" && typeid(T) != typeid(char)) ||
        format_key == "h" && typeid(T) != typeid(short)) ||
        format_key == "i" && typeid(T) != typeid(int)) ||
        format_key == "l" && typeid(T) != typeid(long)) ||
        format_key == "q" && typeid(T) != typeid(long long)) ||

        format_key == "B" && typeid(T) != typeid(unsigned char)) ||
        format_key == "H" && typeid(T) != typeid(unsigned short)) ||
        format_key == "I" && typeid(T) != typeid(unsigned int)) ||
        format_key == "L" && typeid(T) != typeid(unsigned long)) ||
        format_key == "Q" && typeid(T) != typeid(unsigned long long)) ||

        format_key == "d" && typeid(T) != typeid(float)) ||
        format_key == "g" && typeid(T) != typeid(double))
    );
}

/// conversion constructor
CImg(const PyBuffer *pybuffer):_width(0),_height(0),_depth(0),_spectrum(0),_is_shared(false),_data(0) {
    assign(pybuffer);
}

/// in-place constructor

template <typename TYPE>
CImg<T>& assign(const Py_buffer *pybuffer) {
    if (!pybuffer) { return assign() }
    typename TYPE = typecodes[pybuffer->format];
    int colordepth = (pybuffer->ndim > 2) ? pybuffer->shape[2] : 1;
    CImg<TYPE>(
        (TYPE)pybuffer->buf,            /// data ptr
        pybuffer->shape[1],             /// width
        pybuffer->shape[0],             /// height
        1,                              /// z-depth (unused)
        colordepth,                     /// color depth
        True);                          /// it is shared
    return *this;
}




/// END OF PLUGIN IFDEF
#endif

