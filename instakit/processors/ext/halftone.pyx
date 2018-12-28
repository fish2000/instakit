#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from __future__ import division

cdef extern from "halftone.h" nogil:
    unsigned char atkinson_add_error(int b, int e)
    unsigned char* threshold_matrix

import numpy
cimport numpy
cimport cython

from instakit.utils import ndarrays

INT = numpy.int
UINT8 = numpy.uint8
FLOAT32 = numpy.float32

ctypedef numpy.int_t int_t
ctypedef numpy.uint8_t uint8_t
ctypedef numpy.float32_t float32_t

cdef bint threshold_matrix_allocated = False

cdef void atkinson_dither(uint8_t[:, :] input_view, int_t w, int_t h) nogil:
    
    cdef int_t y, x, err
    cdef uint8_t oldpx, newpx
    
    for y in range(h):
        for x in range(w):
            oldpx = input_view[y, x]
            newpx = <uint8_t>threshold_matrix[oldpx]
            err = (<int_t>oldpx - <int_t>newpx) >> 3
            
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

cdef inline uint8_t floyd_steinberg_add_error_SEVEN(uint8_t base,
                                                      int_t err) nogil:
    cdef int_t something = <int_t>base + err * 7 / 16
    return <uint8_t>max(min(255, something), 0)

cdef inline uint8_t floyd_steinberg_add_error_THREE(uint8_t base,
                                                      int_t err) nogil:
    cdef int_t something = <int_t>base + err * 3 / 16
    return <uint8_t>max(min(255, something), 0)

cdef inline uint8_t floyd_steinberg_add_error_CINCO(uint8_t base,
                                                      int_t err) nogil:
    cdef int_t something = <int_t>base + err * 5 / 16
    return <uint8_t>max(min(255, something), 0)

cdef inline uint8_t floyd_steinberg_add_error_ALONE(uint8_t base,
                                                      int_t err) nogil:
    cdef int_t something = <int_t>base + err * 1 / 16
    return <uint8_t>max(min(255, something), 0)

cdef void floyd_steinberg_dither(uint8_t[:, :] input_view, int_t w, int_t h) nogil:
    
    cdef int_t y, x, err
    cdef uint8_t oldpx, newpx
    
    for y in range(h):
        for x in range(w):
            oldpx = input_view[y, x]
            newpx = <uint8_t>threshold_matrix[oldpx]
            input_view[y, x] = newpx
            err = <int_t>oldpx - <int_t>newpx
            
            if (x + 1 < w):
                input_view[y, x+1] = floyd_steinberg_add_error_SEVEN(input_view[y, x+1], err)
            
            if (y + 1 < h) and (x > 0):
                input_view[y+1, x-1] = floyd_steinberg_add_error_THREE(input_view[y+1, x-1], err)
            
            if (y + 1 < h):
                input_view[y+1, x] = floyd_steinberg_add_error_CINCO(input_view[y+1, x], err)
            
            if (y + 1 < h) and (x + 1 < w):
                input_view[y+1, x+1] = floyd_steinberg_add_error_ALONE(input_view[y+1, x+1], err)

@cython.freelist(4)
cdef class Atkinson:
    
    """ Fast cythonized Atkinson-dither halftone image processor """
    
    def __cinit__(self, float32_t threshold = 128.0):
        global threshold_matrix_allocated
        cdef uint8_t idx
        if not threshold_matrix_allocated:
            for idx in range(255):
                threshold_matrix[idx] = <unsigned char>(<uint8_t>(<float32_t>idx / threshold) * 255)
            threshold_matrix_allocated = True
    
    def process(self, image not None):
        input_array = ndarrays.fromimage(image.convert('L')).astype(UINT8)
        cdef uint8_t[:, :] input_view = input_array
        atkinson_dither(input_view, image.size[0], image.size[1])
        output_array = numpy.asarray(input_view.base)
        return ndarrays.toimage(output_array)

@cython.freelist(4)
cdef class FloydSteinberg:
    
    """ Fast cythonized Floyd-Steinberg-dither halftone image processor """
    
    def __cinit__(self, float32_t threshold = 128.0):
        global threshold_matrix_allocated
        cdef uint8_t idx
        if not threshold_matrix_allocated:
            for idx in range(255):
                threshold_matrix[idx] = <unsigned char>(<uint8_t>(<float32_t>idx / threshold) * 255)
            threshold_matrix_allocated = True
    
    def process(self, image not None):
        input_array = ndarrays.fromimage(image.convert('L')).astype(UINT8)
        cdef uint8_t[:, :] input_view = input_array
        floyd_steinberg_dither(input_view, image.size[0], image.size[1])
        output_array = numpy.asarray(input_view.base)
        return ndarrays.toimage(output_array)
