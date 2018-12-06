
from __future__ import division

import numpy
cimport numpy
cimport cython

from instakit.utils.ndarrays import ndarray_fromimage, ndarray_toimage

cdef extern from "halftone.h" nogil:
    unsigned char atkinson_add_error(int b, int e)
    unsigned char* threshold_matrix

INT = numpy.int
FLOAT32 = numpy.float32

ctypedef numpy.int_t int_t
ctypedef numpy.float32_t float32_t

cdef bint threshold_matrix_allocated = False

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void atkinson_dither(int_t[:, :] input_view, int_t w, int_t h) nogil:
    
    cdef int_t y, x, err, oldpx, newpx
    
    for y in range(h):
        for x in range(w):
            oldpx = input_view[y, x]
            newpx = <int_t>threshold_matrix[oldpx]
            err = (oldpx - newpx) >> 3
            
            input_view[y, x] = newpx
            
            if y + 1 < h:
                input_view[y+1, x] = atkinson_add_error(input_view[y+1, x], err)
            
            if y + 2 < h:
                input_view[y+2, x] = atkinson_add_error(input_view[y+2, x], err)
            
            if (y > 0) and (x + 1 < w):
                input_view[y-1, x+1] = atkinson_add_error(input_view[y-1, x+1], err)
            
            if x + 1 < w:
                input_view[y, x+1] = atkinson_add_error(input_view[y, x+1], err)
            
            if (y + 1 < h) and (x + 1 < w):
                input_view[y+1, x+1] = atkinson_add_error(input_view[y+1, x+1], err)
            
            if x + 2 < w:
                input_view[y, x+2] = atkinson_add_error(input_view[y, x+2], err)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int_t floyd_steinberg_add_error(int_t base,
                                            int_t err,
                                            int_t frac) nogil:
    cdef int_t something = base + err * frac / 16
    return (err < 0) and max(something, 0) or min(something, 255)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void floyd_steinberg_dither(int_t[:, :] input_view, int_t w, int_t h) nogil:
    
    cdef int_t y, x, err, oldpx, newpx
    
    for y in range(h):
        for x in range(w):
            oldpx = input_view[y, x]
            newpx = <int_t>threshold_matrix[oldpx]
            input_view[y, x] = newpx
            err = oldpx - newpx
            
            if (x + 1 < w):
                input_view[y, x+1] = floyd_steinberg_add_error(input_view[y, x+1], err, 7)
            
            if (y + 1 < h) and (x - 1 > 0):
                input_view[y+1, x-1] = floyd_steinberg_add_error(input_view[y+1, x-1], err, 3)
            
            if (y + 1 < h):
                input_view[y+1, x] = floyd_steinberg_add_error(input_view[y+1, x], err, 5)
            
            if (y + 1 < h) and (x + 1 < w):
                input_view[y+1, x+1] = floyd_steinberg_add_error(input_view[y+1, x+1], err, 1)


cdef class Atkinson:
    
    """ Fast cythonized Atkinson-dither halftone image processor """
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __cinit__(self, float32_t threshold = 128.0):
        global threshold_matrix_allocated
        cdef int_t i
        if not threshold_matrix_allocated:
            for i in range(255):
                threshold_matrix[i] = <unsigned char>(<int_t>(<float32_t>i / threshold) * 255)
            threshold_matrix_allocated = True
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def process(self, image not None):
        input_array = ndarray_fromimage(image.convert('L')).astype(INT)
        cdef int_t[:, :] input_view = input_array
        atkinson_dither(input_view, image.size[0], image.size[1])
        output_array = numpy.asarray(input_view.base)
        return ndarray_toimage(output_array)


cdef class FloydSteinberg:
    
    """ Fast cythonized Floyd-Steinberg-dither halftone image processor """
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __cinit__(self, float32_t threshold = 128.0):
        global threshold_matrix_allocated
        cdef int_t i
        if not threshold_matrix_allocated:
            for i in range(255):
                threshold_matrix[i] = <unsigned char>(<int_t>(<float32_t>i / threshold) * 255)
            threshold_matrix_allocated = True
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def process(self, image not None):
        input_array = ndarray_fromimage(image.convert('L')).astype(INT)
        cdef int_t[:, :] input_view = input_array
        floyd_steinberg_dither(input_view, image.size[0], image.size[1])
        output_array = numpy.asarray(input_view.base)
        return ndarray_toimage(output_array)
