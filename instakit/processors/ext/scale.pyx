
from __future__ import division

import numpy
cimport numpy
cimport cython
cimport hqx
from scipy import misc

UINT32 = numpy.uint32
ctypedef numpy.uint32_t UINT32_t

cdef class HQx:

    cdef UINT8_t factor

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __cinit__(object self not None, int factor=2):
        self.factor = <int>factor

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def process(object self not None, object pilimage not None):
        cdef int w, h, factor = self.factor
        cdef numpy.ndarray[UINT32_t, ndim=3, mode="c"] ndimage_in = misc.fromimage(
            pilimage.convert('RGBA')).astype(UINT32)
        cdef numpy.ndarray[UINT32_t, ndim=3, mode="c"] ndimage_out

        w = <int>ndimage_in.shape[0]
        h = <int>ndimage_in.shape[1]
        ndimage_out = numpy.zeros((w * factor, h * factor, 4)), dtype=UINT32)

        if self.factor == 2:
            hqx.hq2x_32(&ndimage_in[0, 0], &ndimage_out[0, 0], w, h)
        elif self.factor == 3:
            hqx.hq3x_32(&ndimage_in[0, 0], &ndimage_out[0, 0], w, h)
        elif self.factor == 4:
            hqx.hq4x_32(&ndimage_in[0, 0], &ndimage_out[0, 0], w, h)
        else:
            pass

        return misc.toimage(ndimage_out).convert('RGB')
