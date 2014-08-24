#ifndef PyImgC_INTERFACE_H
#define PyImgC_INTERFACE_H
/// INSERT PYTHON C-API STUFF HERE

#define cimg_OS 1                       /// unix-like
#define cimg_verbosity 1                /// log to the console
#define cimg_display 0                  /// don't need this

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