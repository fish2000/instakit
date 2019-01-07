
from libc.stdint cimport *
from libcpp.vector cimport vector

cdef extern from "butteraugli.h" namespace "butteraugli" nogil:
    
    cppclass Image[ComponentType]:
        Image()
        Image(size_t const, size_t const)       # xsize, ysize
        Image(size_t const, size_t const,       # 
                            ComponentType val)  # xsize, ysize, component-type
        Image(size_t const, size_t const,       # xsize, ysize,
              uint8_t*,     size_t const)       # byteptr, bytes-per-row
        
        Image(Image&&)                          # move constructor
        Image& operator=(Image&&)               # move assignment operator
        
        size_t xsize() const
        size_t ysize() const
        
        const ComponentType* Row(size_t const)
        const uint8_t* byte_ptr "bytes"()
        
        size_t bytes_per_row()
        intptr_t PixelsPerRow() const

ctypedef Image[float]   ImageF
ctypedef Image[uint8_t] Image8

cdef extern from "butteraugli.h" namespace "butteraugli" nogil:
    
    cdef vector[Image[T]] CreatePlanes[T](size_t const,
                                          size_t const,
                                          size_t const)

ctypedef vector[ImageF]     imagefvec
ctypedef vector[Image8]     image8vec
ctypedef vector[float]      floatvec
ctypedef vector[floatvec]   floatvecvec


cdef extern from "butteraugli.h" namespace "butteraugli" nogil:
    
    cdef bint ButteraugliInterface(imagefvec&,
                                   imagefvec&, float,
                                   ImageF&,
                                   double&)
    
    cdef double ButteraugliFuzzyClass(double)
    cdef double ButteraugliFuzzyInverse(double)
    
    cdef bint ButteraugliAdaptiveQuantization(size_t, size_t,
                                              floatvecvec&,
                                              floatvec&)
    
    cdef const double kButteraugliQuantLow
    cdef const double kButteraugliQuantHigh
