#ifndef PyImgC_INTERFACE_H
#define PyImgC_INTERFACE_H
/// INSERT PYTHON C-API STUFF HERE

/// forward-declare
//struct _cimg_math_parser;
//bool is_empty();

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

#define cimg_plugin "../cimg_pybuffer.h"   /// Py_buffer CImg plugin interface

#include <Python.h>                     /// Python C API
#include "structmember.h"
#include <stdio.h>
#include <string.h>
#include <numpy/ndarrayobject.h>        /// NumPy C API
#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"
#include "PyImgC_Constants.h"
#include "cimg/CImg.h"                  /// the CImg library itself

/// END OF PYTHON C-API STUFF'S LIMINAL HASH-IFNDEF
#endif /// PyImgC_INTERFACE_H