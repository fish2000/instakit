
cimport cython
from cython.operator cimport address
from cython cimport view

from instakit.utils.ext cimport funcs

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] hsluv_to_rgb(double[:] hsl_triple) nogil:
    cdef double[:] rgb_triple = hsl_triple
    funcs.hsluv2rgb(        hsl_triple[0],          hsl_triple[1],          hsl_triple[2],
                    address(rgb_triple[0]), address(rgb_triple[1]), address(rgb_triple[2]))
    return rgb_triple

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] rgb_to_hsluv(double[:] rgb_triple) nogil:
    cdef double[:] hsl_triple = rgb_triple
    funcs.rgb2hsluv(        rgb_triple[0],          rgb_triple[1],          rgb_triple[2],
                    address(hsl_triple[0]), address(hsl_triple[1]), address(hsl_triple[2]))
    return hsl_triple

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] hpluv_to_rgb(double[:] hpl_triple) nogil:
    cdef double[:] rgb_triple = hpl_triple
    funcs.hpluv2rgb(        hpl_triple[0],          hpl_triple[1],          hpl_triple[2],
                    address(rgb_triple[0]), address(rgb_triple[1]), address(rgb_triple[2]))
    return rgb_triple

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] rgb_to_hpluv(double[:] rgb_triple) nogil:
    cdef double[:] hpl_triple = rgb_triple
    funcs.rgb2hpluv(        rgb_triple[0],          rgb_triple[1],          rgb_triple[2],
                    address(hpl_triple[0]), address(hpl_triple[1]), address(hpl_triple[2]))
    return hpl_triple
