
//#include "PyImgC_PyBuffer.h"

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





/// END OF PLUGIN IFDEF
#endif

