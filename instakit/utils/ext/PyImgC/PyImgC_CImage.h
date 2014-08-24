#ifndef PyImgC_CIMAGE_H
#define PyImgC_CIMAGE_H

#define cimg_OS 1                       /// unix-like
#define cimg_verbosity 1                /// log to the console
#define cimg_display 0                  /// don't need this

//#define cimg_use_png 1                  /// png
//#define cimg_use_jpeg 1                 /// jpeg
//#define cimg_use_tiff 1                 /// tiff
//#define cimg_use_zlib 1                 /// compressed output
//#define cimg_use_magick 1               /// ImageMagick++ I/O
//#define cimg_use_fftw3 1                /// libFFTW3
//#define cimg_use_openexr 1              /// OpenEXR
//#define cimg_use_lapack 1               /// LAPACK

#include <Python.h>
#include <structmember.h>

#include "cimg/CImg.h"
using namespace cimg_library;

#ifndef cimg_imagepath
#define cimg_imagepath "cimg/img/"
#endif

#define Tx(NPY_TYPE) typename numpy::decoder<NPY_TYPE>::type

//CImg<TX> view(charbuffer, sX, sY, 1, channels, is_shared=True);

template <npy_intp npy_type>
struct CImageView {
    CImg<Tx(npy_type)> operator()(Py_buffer *pybuffer,
                        int sX, int sY, int channels,
                        short int is_shared=True) {
        CImg<Tx(npy_type)> view(pybuffer->buf, sX, sY, 1, channels, is_shared);
    }
};

template <typename T>
struct CImage {
    PyObject_HEAD
    PyArray_Descr *dtype;
    CImg<T> *cimage;
};



#endif /// PyImgC_INTERFACE_H