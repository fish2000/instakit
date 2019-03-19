#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from __future__ import division

ctypedef unsigned char byte_t

cdef extern from "halftone.h" nogil:
    byte_t atkinson_add_error(int b, int e)

import numpy
cimport numpy
cimport cython

from instakit.utils import ndarrays

ctypedef numpy.int_t int_t
ctypedef numpy.uint8_t uint8_t
ctypedef numpy.uint32_t uint32_t
ctypedef numpy.float32_t float32_t

cdef void atkinson_dither(uint8_t[:, :] input_view,
                          int_t width, int_t height,
                          byte_t* threshold_matrix_ptr) nogil:
    
    cdef int_t y, x, err
    cdef uint8_t oldpx, newpx
    
    for y in range(height):
        for x in range(width):
            oldpx = input_view[y, x]
            newpx = <uint8_t>threshold_matrix_ptr[oldpx]
            err = (<int_t>oldpx - <int_t>newpx) >> 3
            
            input_view[y, x] = newpx
            
            if y + 1 < height:
                input_view[y+1, x] = atkinson_add_error(input_view[y+1, x], err)
            
            if y + 2 < height:
                input_view[y+2, x] = atkinson_add_error(input_view[y+2, x], err)
            
            if (y > 0) and (x + 1 < width):
                input_view[y-1, x+1] = atkinson_add_error(input_view[y-1, x+1], err)
            
            if x + 1 < width:
                input_view[y, x+1] = atkinson_add_error(input_view[y, x+1], err)
            
            if (y + 1 < height) and (x + 1 < width):
                input_view[y+1, x+1] = atkinson_add_error(input_view[y+1, x+1], err)
            
            if x + 2 < width:
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

cdef void floyd_steinberg_dither(uint8_t[:, :] input_view,
                                 int_t width, int_t height,
                                 byte_t* threshold_matrix_ptr) nogil:
    
    cdef int_t y, x, err
    cdef uint8_t oldpx, newpx
    
    for y in range(height):
        for x in range(width):
            oldpx = input_view[y, x]
            newpx = <uint8_t>threshold_matrix_ptr[oldpx]
            input_view[y, x] = newpx
            err = <int_t>oldpx - <int_t>newpx
            
            if (x + 1 < width):
                input_view[y, x+1] = floyd_steinberg_add_error_SEVEN(input_view[y, x+1], err)
            
            if (y + 1 < height) and (x > 0):
                input_view[y+1, x-1] = floyd_steinberg_add_error_THREE(input_view[y+1, x-1], err)
            
            if (y + 1 < height):
                input_view[y+1, x] = floyd_steinberg_add_error_CINCO(input_view[y+1, x], err)
            
            if (y + 1 < height) and (x + 1 < width):
                input_view[y+1, x+1] = floyd_steinberg_add_error_ALONE(input_view[y+1, x+1], err)

@cython.freelist(16)
cdef class ThresholdMatrixDitherer:
    
    """ Base ditherer image processor class """
    
    cdef:
        byte_t threshold_matrix[256]
    
    def __cinit__(self, float32_t threshold = 128.0):
        cdef uint8_t idx
        with nogil:
            for idx in range(255):
                self.threshold_matrix[idx] = <unsigned char>(<uint8_t>(<float32_t>idx / threshold) * 255)

cdef class Atkinson(ThresholdMatrixDitherer):
    
    """ Fast cythonized Atkinson-dither halftone image processor """
    
    def process(self, image not None):
        input_array = ndarrays.fromimage(image.convert('L'), dtype=numpy.uint8)
        cdef uint32_t width = image.size[0]
        cdef uint32_t height = image.size[1]
        cdef uint8_t[:, :] input_view = input_array
        with nogil:
            atkinson_dither(input_view, width, height, self.threshold_matrix)
        output_array = numpy.asarray(input_view.base)
        return ndarrays.toimage(output_array)

cdef class FloydSteinberg(ThresholdMatrixDitherer):
    
    """ Fast cythonized Floyd-Steinberg-dither halftone image processor """
    
    def process(self, image not None):
        input_array = ndarrays.fromimage(image.convert('L'), dtype=numpy.uint8)
        cdef uint32_t width = image.size[0]
        cdef uint32_t height = image.size[1]
        cdef uint8_t[:, :] input_view = input_array
        with nogil:
            floyd_steinberg_dither(input_view, width, height, self.threshold_matrix)
        output_array = numpy.asarray(input_view.base)
        return ndarrays.toimage(output_array)
