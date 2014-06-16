
from __future__ import division

import numpy
cimport numpy
cimport cython

from scipy import misc

cdef extern from "halftone.h":
    unsigned char adderror(int b, int e)
    unsigned char* threshold_matrix

INT = numpy.int
UINT8 = numpy.uint8
FLOAT32 = numpy.float32

ctypedef numpy.int_t INT_t
ctypedef numpy.uint8_t UINT8_t
ctypedef numpy.float32_t FLOAT32_t


cdef class Atkinson:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __cinit__(self not None, FLOAT32_t threshold=128.0):
        cdef INT_t i
        for i in range(255):
            threshold_matrix[i] = <unsigned char>(<INT_t>(<FLOAT32_t>i / threshold) * 255)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def process(self not None, pilimage not None):
        in_array = misc.fromimage(pilimage.convert('L')).astype(INT)
        self.atkinson(in_array, in_array.shape[0], in_array.shape[1])
        return misc.toimage(in_array)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def atkinson(self not None, numpy.ndarray[INT_t, ndim=2, mode="c"] image_i not None, INT_t w, INT_t h):

        cdef INT_t x, y, i, err, old, new

        for y in range(h):
            for x in range(w):
                old = image_i[x, y]
                new = <INT_t>threshold_matrix[old]
                err = (old - new) >> 3

                image_i[x, y] = new

                if x+1 < w:
                    image_i[x+1, y] = adderror(image_i[x+1, y], err)

                if x+2 < w:
                    image_i[x+2, y] = adderror(image_i[x+2, y], err)

                if (x > 0) and (y+1 < h):
                    image_i[x-1, y+1] = adderror(image_i[x-1, y+1], err)

                if y+1 < h:
                    image_i[x, y+1] = adderror(image_i[x, y+1], err)

                if (x+1 < w) and (y+1 < h):
                    image_i[x+1, y+1] = adderror(image_i[x+1, y+1], err)

                if y+2 < h:
                    image_i[x, y+2] = adderror(image_i[x, y+2], err)

