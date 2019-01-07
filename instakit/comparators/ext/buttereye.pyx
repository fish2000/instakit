#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from cython.operator cimport dereference as deref
from cpython.float cimport PyFloat_AS_DOUBLE
from libcpp.memory cimport unique_ptr

from instakit.comparators.ext.butteraugli cimport Image8, ImageF, image8vec, imagefvec, floatvecvec, CreatePlanes
from instakit.utils.mode import Mode

ctypedef unique_ptr[ImageF] imagef_ptr
ctypedef unique_ptr[Image8] image8_ptr

# cdef imagef_ptr image_to_planar(object pilimage):
#     cdef int width, height, x, y
#     cdef double point
#     cdef float* planerow
#     cdef object pypoint, image, accessor
#     cdef imagef_ptr plane
#
#     image = Mode.F.process(pilimage)
#     accessor = image.load()
#     width, height = image.size
#     # plane.reset(new ImageF(width, height))
#
#     for y in range(height):
#         planerow = deref(plane).Row(y)
#         for x in range(width):
#             pypoint = accessor[x, y]
#             point = PyFloat_AS_DOUBLE(pypoint)
#             planerow[x] = <float>point
#
#     return plane

cdef imagefvec image_to_planar_vector(object pilimage):
    cdef int width, height, x, y
    cdef double point
    cdef float* planerow
    cdef object pypoint, image, accessor
    cdef object bands, band
    cdef imagef_ptr plane
    cdef imagefvec planes
    cdef int bandcount, idx
    
    # image = Mode.F.process(pilimage)
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
        # plane.reset(new ImageF(width, height))
        
        for y in range(height):
            # planerow = deref(plane).Row(y)
            planerow = planes[idx].Row(y)
            for x in range(width):
                pypoint = accessor[x, y]
                point = PyFloat_AS_DOUBLE(pypoint)
                planerow[x] = <float>point
        
        # planes.push_back(deref(imagef_ptr))
        # planes[idx] = ImageF(deref(plane))
        # planes.push_back(plane.release())
    
    # return planes