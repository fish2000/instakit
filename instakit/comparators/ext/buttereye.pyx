#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from cython.operator cimport dereference as deref
from cpython.float cimport PyFloat_AS_DOUBLE
from libcpp.memory cimport unique_ptr

from instakit.comparators.ext.butteraugli cimport Image8, ImageF
from instakit.comparators.ext.butteraugli cimport image8vec, imagefvec
from instakit.comparators.ext.butteraugli cimport floatvecvec, CreatePlanes
from instakit.utils.mode import Mode

ctypedef unique_ptr[ImageF] imagef_ptr
ctypedef unique_ptr[Image8] image8_ptr

cdef imagefvec image_to_planar_vector(object pilimage):
    cdef int width, height, x, y
    cdef double point
    cdef float* planerow
    cdef object pypoint, image, accessor
    cdef object bands, band
    cdef imagef_ptr plane
    cdef imagefvec planes
    cdef int bandcount, idx
    
    width, height = pilimage.size
    bands = Mode.RGB.process(pilimage).split()
    
    with nogil:
        bandcount = 3
        planes = CreatePlanes[float](width,
                                     height,
                                     bandcount)
    
    for idx in range(bandcount):
        band = bands[idx]
        image = Mode.F.process(band)
        accessor = image.load()
        
        for y in range(height):
            planerow = planes[idx].Row(y)
            for x in range(width):
                pypoint = accessor[x, y]
                point = PyFloat_AS_DOUBLE(pypoint)
                planerow[x] = <float>point
    
    # AND SO NOW WHAT??!
