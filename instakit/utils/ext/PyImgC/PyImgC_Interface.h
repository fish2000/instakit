#ifndef PyImgC_INTERFACE_H
#define PyImgC_INTERFACE_H
/// INSERT PYTHON C-API STUFF HERE

#define cimg_OS 1                       /// unix-like
#define cimg_verbosity 1                /// log to the console
#define cimg_display 0                  /// don't need this

#define cimg_use_png 1                  /// png
#define cimg_use_jpeg 1                 /// jpeg
//#define cimg_use_tiff 1                 /// tiff
#define cimg_use_zlib 1                 /// compressed output
//#define cimg_use_magick 1               /// ImageMagick++ I/O
//#define cimg_use_fftw3 1                /// libFFTW3
//#define cimg_use_openexr 1              /// OpenEXR
//#define cimg_use_lapack 1               /// LAPACK

#define cimg_plugin "cimg_pybuffer.h"   /// Py_buffer CImg plugin interface

#include <Python.h>                     /// Python C API
#include <numpy/ndarrayobject.h>        /// NumPy C API

#include "PyImgC_Constants.h"           /// our library support
#include "PyImgC_PyBuffer.h"            /// our buffer functions
#include "cimg/CImg.h"                  /// the CImg library itself

/// END OF PYTHON C-API STUFF'S LIMINAL HASH-IFNDEF
#endif /// PyImgC_INTERFACE_H