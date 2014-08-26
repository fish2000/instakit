#ifndef PyImgC_CIMAGE_H
#define PyImgC_CIMAGE_H

#define cimg_OS 1                       /// unix-like
#define cimg_verbosity 1                /// log to the console
#define cimg_display 0                  /// don't need this

#define cimg_use_jpeg 1                 /// jpeg
#define cimg_use_zlib 1                 /// compressed output

//#define cimg_use_png 1                /// png (via setup.py)
//#define cimg_use_tiff 1               /// tiff (via setup.py)
//#define cimg_use_magick 1             /// ImageMagick++ I/O (via setup.py)
//#define cimg_use_fftw3 1              /// libFFTW3 (via setup.py)
//#define cimg_use_openexr 1            /// OpenEXR (via setup.py)
//#define cimg_use_lapack 1             /// LAPACK

#include <map>
#include <type_traits>
#include <Python.h>
#include <structmember.h>

#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include "numpypp/numpy.hpp"

#include "cimg/CImg.h"
using namespace cimg_library;
using namespace std;
#ifndef cimg_imagepath
#define cimg_imagepath "cimg/img/"
#endif

/// I hate the way the name 'CImg' looks written out --
/// all stubby-looking and typographically cramped. And so.
#ifndef CImage
#define CImage CImg
#endif

template <typename T>
CImage<T> from_pybuffer(Py_buffer *pybuffer, int sW, int sH,
                    int channels, bool is_shared=true) {
    CImage<T> view(pybuffer->buf,
        sW, sH, 1,
        channels, is_shared);
    return view;
}

template <typename T>
CImg<T> from_pybuffer(Py_buffer *pybuffer, bool is_shared=true) {
    CImg<T> view(
        pybuffer->buf,
        pybuffer->shape[1],
        pybuffer->shape[0],
        1, 3, is_shared);
    return view;
}

template <typename T>
CImg<T> from_pyarray(PyArrayObject *pyarray, bool is_shared=true) {
    int sW = 0;
    int sH = 0;
    int channels = 0;
    switch (PyArray_NDIM(pyarray)) {
        case 3:
        {
            channels = PyArray_DIM(pyarray, 2); /// starts from zero, right?...
            sW = PyArray_DIM(pyarray, 1);
            sH = PyArray_DIM(pyarray, 0);
        }
        break;
        case 2:
        {
            channels = 1;
            sW = PyArray_DIM(pyarray, 1);
            sH = PyArray_DIM(pyarray, 0);
        }
        break;
        default:
        {
            return NULL;
        }
        break;
    }
    CImg<T> view(PyArray_DATA(pyarray), sW, sH,
        1, channels, is_shared);
    return view;
}

template <typename T>
CImg<T> from_pyobject(PyObject *datasource, int sW, int sH,
                    int channels, bool is_shared=true) {
    CImg<T> view(sW, sH, 1, channels, is_shared);
    return view;
}

template <typename T>
CImg<T> from_pyobject(PyObject *datasource, bool is_shared=true) {
    CImg<T> view(640, 480, 1, 3, is_shared);
    return view;
}

#define NILCODE '~'

struct CImage_SubBase {};

template <typename dT>
struct CImage_Traits;

template <typename dT>
struct CImage_Base : public CImage_SubBase {
    typedef typename CImage_Traits<dT>::value_type value_type;

    CImg<value_type> as_pybuffer(Py_buffer *pybuffer, bool is_shared=true) {
        return from_pybuffer<bool>(pybuffer, is_shared); }

    CImg<value_type> as_pybuffer_with_dims(Py_buffer *pybuffer,
        int sW, int sH, int channels=3,
        bool is_shared=true) {
        return from_pybuffer<value_type>(pybuffer, sW, sH, channels, is_shared); }

    CImg<value_type> as_pyarray(PyArrayObject *pyarray, bool is_shared=true) {
        return from_pyarray<value_type>(pyarray, is_shared); }

    CImg<value_type> as_datasource(PyObject *datasource, bool is_shared=true) {
        return from_pyobject<value_type>(datasource, is_shared); }

    CImg<value_type> as_datasource_with_dims(Py_buffer *pybuffer,
        int sW, int sH, int channels=3,
        bool is_shared=true) {
        return from_pybuffer<value_type>(pybuffer, sW, sH, channels, is_shared); }

    inline bool operator()(const char sc) {
        dT self = static_cast<dT*>(this);
        for (int idx = 0; self->structcode[idx] != NILCODE; ++idx) {
            if (self->structcode[idx] == sc) { return true; }
        }
        return false; }
    
    inline bool operator[](const npy_intp tc) {
        dT self = static_cast<dT*>(this);
        return tc == self->typecode();
    }
    
    template <typename... Args>
    static dT& get_instance(Args... args) {
        static dT instance{std::forward<Args>(args)...};
        return instance;
    }
    
};

template <typename T>
struct CImage_Type : public CImage_Base<CImage_Type<T>> {
    typedef typename CImage_Traits<CImage_Type>::value_type value_type;
    int typecode() {
        return numpy::dtype_code<value_type>();
    }
};

template <typename T>
struct CImage_Traits<CImage_Type<T>> {
    typedef T value_type;
};

struct CImage_FunctorType {
    //static const NPY_TYPES operator()(int check) {
    static const NPY_TYPES typecode_cast(int to_cast) {
        static NPY_TYPES interim = static_cast<NPY_TYPES>(to_cast);
        return const_cast<NPY_TYPES&>(interim);
    }
};

template <NPY_TYPES>
struct CImage_Functor : public CImage_FunctorType {};

/////////////////////////////////// AUTOGENERATED ///////////////////////////////////////
/////////////////////////////////// AUTOGENERATED ///////////////////////////////////////
/////////////////////////////////// AUTOGENERATED ///////////////////////////////////////

struct CImage_NPY_BOOL : public CImage_Type<bool> {
    const char structcode[2] = { '?', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_BOOL() {}
};

template <>
struct CImage_Functor<NPY_BOOL> {
    typedef CImage_NPY_BOOL impl;
};


struct CImage_NPY_BYTE : public CImage_Type<char> {
    const char structcode[2] = { 'b', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_BYTE() {}
};

template <>
struct CImage_Functor<NPY_BYTE> {
    typedef CImage_NPY_BYTE impl;
};


struct CImage_NPY_HALF : public CImage_Type<npy_half> {
    const char structcode[2] = { 'e', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = false;
    CImage_NPY_HALF() {}
};

template <>
struct CImage_Functor<NPY_HALF> {
    typedef CImage_NPY_HALF impl;
};


struct CImage_NPY_SHORT : public CImage_Type<short> {
    const char structcode[2] = { 'h', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_SHORT() {}
};

template <>
struct CImage_Functor<NPY_SHORT> {
    typedef CImage_NPY_SHORT impl;
};


struct CImage_NPY_INT : public CImage_Type<int> {
    const char structcode[2] = { 'i', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_INT() {}
};

template <>
struct CImage_Functor<NPY_INT> {
    typedef CImage_NPY_INT impl;
};


struct CImage_NPY_LONG : public CImage_Type<long> {
    const char structcode[2] = { 'l', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_LONG() {}
};

template <>
struct CImage_Functor<NPY_LONG> {
    typedef CImage_NPY_LONG impl;
};


struct CImage_NPY_LONGLONG : public CImage_Type<long long> {
    const char structcode[2] = { 'q', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_LONGLONG() {}
};

template <>
struct CImage_Functor<NPY_LONGLONG> {
    typedef CImage_NPY_LONGLONG impl;
};


struct CImage_NPY_UBYTE : public CImage_Type<unsigned char> {
    const char structcode[2] = { 'B', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_UBYTE() {}
};

template <>
struct CImage_Functor<NPY_UBYTE> {
    typedef CImage_NPY_UBYTE impl;
};


struct CImage_NPY_USHORT : public CImage_Type<unsigned short> {
    const char structcode[2] = { 'H', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_USHORT() {}
};

template <>
struct CImage_Functor<NPY_USHORT> {
    typedef CImage_NPY_USHORT impl;
};


struct CImage_NPY_UINT : public CImage_Type<unsigned int> {
    const char structcode[2] = { 'I', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_UINT() {}
};

template <>
struct CImage_Functor<NPY_UINT> {
    typedef CImage_NPY_UINT impl;
};


struct CImage_NPY_ULONG : public CImage_Type<unsigned long> {
    const char structcode[2] = { 'L', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_ULONG() {}
};

template <>
struct CImage_Functor<NPY_ULONG> {
    typedef CImage_NPY_ULONG impl;
};


struct CImage_NPY_ULONGLONG : public CImage_Type<unsigned long long> {
    const char structcode[2] = { 'Q', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_ULONGLONG() {}
};

template <>
struct CImage_Functor<NPY_ULONGLONG> {
    typedef CImage_NPY_ULONGLONG impl;
};


struct CImage_NPY_CFLOAT : public CImage_Type<std::complex<float>> {
    const char structcode[2] = { 'f', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = true;
    CImage_NPY_CFLOAT() {}
};

template <>
struct CImage_Functor<NPY_CFLOAT> {
    typedef CImage_NPY_CFLOAT impl;
};


struct CImage_NPY_CDOUBLE : public CImage_Type<std::complex<double>> {
    const char structcode[2] = { 'd', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = true;
    CImage_NPY_CDOUBLE() {}
};

template <>
struct CImage_Functor<NPY_CDOUBLE> {
    typedef CImage_NPY_CDOUBLE impl;
};


struct CImage_NPY_FLOAT : public CImage_Type<float> {
    const char structcode[2] = { 'f', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = false;
    CImage_NPY_FLOAT() {}
};

template <>
struct CImage_Functor<NPY_FLOAT> {
    typedef CImage_NPY_FLOAT impl;
};


struct CImage_NPY_DOUBLE : public CImage_Type<double> {
    const char structcode[2] = { 'd', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = false;
    CImage_NPY_DOUBLE() {}
};

template <>
struct CImage_Functor<NPY_DOUBLE> {
    typedef CImage_NPY_DOUBLE impl;
};


struct CImage_NPY_CLONGDOUBLE : public CImage_Type<std::complex<long double>> {
    const char structcode[2] = { 'g', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_CLONGDOUBLE() {}
};

template <>
struct CImage_Functor<NPY_CLONGDOUBLE> {
    typedef CImage_NPY_CLONGDOUBLE impl;
};


struct CImage_NPY_LONGDOUBLE : public CImage_Type<std::complex<long double>> {
    const char structcode[2] = { 'g', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = true;
    CImage_NPY_LONGDOUBLE() {}
};

template <>
struct CImage_Functor<NPY_LONGDOUBLE> {
    typedef CImage_NPY_LONGDOUBLE impl;
};

/////////////////////////////////// !AUTOGENERATED ///////////////////////////////////////
/////////////////////////////////// !AUTOGENERATED ///////////////////////////////////////
/////////////////////////////////// !AUTOGENERATED ///////////////////////////////////////

#endif /// PyImgC_INTERFACE_H