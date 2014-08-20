#ifndef PyImgC_INTERFACE_H
#define PyImgC_INTERFACE_H

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

#include "numpypp/array.hpp"
#include "cimg/CImg.h"

template <typename T>
struct Image {
    PyObject_HEAD
    PyArray_Descr *dtype;
    CImg<T> *cimage;
};



#endif /// PyImgC_INTERFACE_H