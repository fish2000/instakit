#!/usr/bin/env python
# encoding: utf-8
"""
ndarrays.py

Created by FI$H 2000 on 2013-08-23.
Copyright © 2012-2019 Objects In Space And Time, LLC. All rights reserved.

The `bytescale`[0], `fromimage`[1], and `toimage`[2] functions have been
adapted from the versions published in the now-historic `scipy.misc.pilutils`
module; the last official release of which looks to have been in SciPy 1.1.0:

* [0] https://git.io/fhIoX
* [1] https://git.io/fhIo1
* [2] https://git.io/fhIoD

"""
from __future__ import division, print_function

import numpy
from instakit.utils.mode import Mode
from instakit.abc import NDProcessorBase
from instakit.exporting import Exporter

exporter = Exporter(path=__file__)
export = exporter.decorator()

uint8_t = numpy.uint8
uint32_t = numpy.uint32
float32_t = numpy.float32

@export
def bytescale(data, cmin=None, cmax=None,
                    high=255,  low=0):
    """
    Byte-scales a `numpy.ndarray` of nd-image data.
    
    “Byte scaling” means 1) casting the input image to the ``uint8_t`` dtype, and
                         2) scaling the range to ``(low, high)`` (default 0-255).
    
    If the input image data is already of dtype ``uint8_t``, no scaling is done.
    
    Parameters
    ----------
    data : `numpy.ndarray`
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    
    Returns
    -------
    array : `numpy.ndarray` of dtype ``uint8_t``
        The byte-scaled array.
    """
    if data.dtype == uint8_t:
        return data
    
    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")
    
    if cmin is None:
        cmin = data.min()
    
    if cmax is None:
        cmax = data.max()
    
    cscale = cmax - cmin
    
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1
    
    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(uint8_t)

@export
def fromimage(image, flatten=False,
                        mode=None,
                       dtype=None):
    """
    Return the data from an input PIL image as a `numpy.ndarray`.
    
    Parameters
    ----------
    im : PIL image
        Input image.
    flatten : bool, optional
        If true, convert the output to greyscale. Default is False.
    mode : str / Mode, optional
        Mode to convert image to, e.g. ``'RGB'``. See the Notes of the
        `imread` docstring for more details.
    dtype : str / ``numpy.dtype``, optional
        Numpy dtype to which to cast the output image array data,
        e.g. ``'float64'`` or ``'uint16'``. 
    
    Returns
    -------
    fromimage : ndarray (rank 2..3)
        The individual color channels of the input image are stored in the
        third dimension, such that greyscale (`L`) images are MxN (rank-2),
        `RGB` images are MxNx3 (rank-3), and `RGBA` images are MxNx4 (rank-3).
    """
    from PIL import Image
    
    if not Image.isImageType(image):
        raise TypeError(f"Input is not a PIL image (got {image!r})")
    
    if mode is not None:
        if not Mode.is_mode(mode):
            mode = Mode.for_string(mode)
        image = mode.process(image)
    elif Mode.of(image) is Mode.P:
        # Mode 'P' means there is an indexed "palette".  If we leave the mode
        # as 'P', then when we do `a = numpy.array(im)` below, `a` will be a 2D
        # containing the indices into the palette, and not a 3D array
        # containing the RGB or RGBA values.
        if 'transparency' in image.info:
            image = Mode.RGBA.process(image)
        else:
            image = Mode.RGB.process(image)
    
    if flatten:
        image = Mode.F.process(image)
    elif Mode.of(image) is Mode.MONO:
        # Workaround for crash in PIL. When im is 1-bit, the call numpy.array(im)
        # can cause a seg. fault, or generate garbage. See
        # https://github.com/scipy/scipy/issues/2138 and
        # https://github.com/python-pillow/Pillow/issues/350.
        # This converts im from a 1-bit image to an 8-bit image.
        image = Mode.L.process(image)
    
    out = numpy.array(image)
    
    if dtype is not None:
        return out.astype(
              numpy.dtype(dtype))
    
    return out

_errstr = "Mode unknown or incompatible with input array shape"

@export
def toimage(array,  high=255,    low=0,
                    cmin=None,  cmax=None,
                     pal=None,
                    mode=None,
            channel_axis=None):
    """
    Takes an input `numpy.ndarray` and returns a PIL image.
            
    The mode of the image returned depends on 1) the array shape, and 
                                              2) the `pal` and `mode` keywords.
    
    For 2D arrays, if `pal` is a valid (N, 3) rank-2, ``uint8_t`` bytearray --
    populated with an `RGB` LUT of values from 0 to 255, ``mode='P'`` (256-color
    single-channel palette mode) will be used; otherwise ``mode='L'`` (256-level
    single-channel grayscale mode) will be employed -- unless a “mode” argument
    is given, as either 'F' or 'I'; in which case conversion to either a float
    or an integer rank-3 array will be made.
    
    .. warning::
    
        This function calls `bytescale` under the hood, to rescale the image
        pixel values across the full (0, 255) ``uint8_t`` range if ``mode``
        is one of either: ``None, 'L', 'P', 'l'``.
        
        It will also cast rank-2 image data to ``uint32_t`` when ``mode=None``
        (which is the default).
    
    Notes
    -----
    For 3D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data. If one of the dimensions is 3, the mode
    is 'RGB' by default, or 'YCbCr' if selected.
    
    The input `numpy.ndarray` must be either rank-2 or rank-3.
    """
    from pprint import pformat
    
    data = numpy.asarray(array)
    if numpy.iscomplexobj(data):
        raise ValueError("Cannot convert arrays of complex values")
    
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("input array lacks a suitable shape for any mode")
    
    if mode is not None:
        if not Mode.is_mode(mode):
            mode = Mode.for_string(mode)
    
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        
        if mode is Mode.F:
            return mode.frombytes(shape, data.astype(float32_t).tostring())
        
        if mode in [ None, Mode.L, Mode.P ]:
            bytedata = bytescale(data, high=high,
                                       low=low,
                                       cmin=cmin,
                                       cmax=cmax)
            image = Mode.L.frombytes(shape, bytedata.tostring())
            
            if pal is not None:
                image.putpalette(numpy.asarray(pal,
                                 dtype=uint8_t).tostring()) # Becomes mode='P' automatically
            elif mode is Mode.P:  # default grayscale
                pal = (numpy.arange(0, 256, 1, dtype=uint8_t)[:, numpy.newaxis] *
                       numpy.ones((3,),        dtype=uint8_t)[numpy.newaxis, :])
                image.putpalette(numpy.asarray(pal,
                                 dtype=uint8_t).tostring())
            
            return image
        
        if mode is Mode.MONO:  # high input gives threshold for 1
            bytedata = (data > high)
            return mode.frombytes(shape, bytedata.tostring())
        
        if cmin is None:
            cmin = numpy.amin(numpy.ravel(data))
        
        if cmax is None:
            cmax = numpy.amax(numpy.ravel(data))
        
        data = (data * 1.0 - cmin) * (high - low) / (cmax - cmin) + low
        
        if mode is Mode.I:
            image = mode.frombytes(shape, data.astype(uint32_t).tostring())
        else:
            raise ValueError(_errstr)
        
        return image
    
    # if here then 3D array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = numpy.flatnonzero(numpy.asarray(shape) == 3)[0]
        else:
            ca = numpy.flatnonzero(numpy.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError(
                    f"Could not find a channel dimension (shape = {pformat(shape)})")
    else:
        ca = channel_axis
    
    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError(f"Channel dimension invalid (#channels = {numch})")
    
    bytedata = bytescale(data, high=high,
                               low=low,
                               cmin=cmin,
                               cmax=cmax)
    
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = numpy.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = numpy.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    
    if mode is None:
        if numch == 3:
            mode = Mode.RGB
        else:
            mode = Mode.RGBA
    
    if mode not in [ Mode.RGB, Mode.RGBA, Mode.YCbCr, Mode.CMYK ]:
        raise ValueError(_errstr)
    
    if mode in [ Mode.RGB, Mode.YCbCr ]:
        if numch != 3:
            raise ValueError(f"Invalid shape for mode “{mode}”: {pformat(shape)}")
    if mode in [ Mode.RGBA, Mode.CMYK ]:
        if numch != 4:
            raise ValueError(f"Invalid shape for mode “{mode}”: {pformat(shape)}")
    
    # Here we know both `strdata` and `mode` are correct:
    image = mode.frombytes(shape, strdata)
    return image

@export
class NDProcessor(NDProcessorBase):
    
    """ An image processor ancestor class that represents PIL image
        data in a `numpy.ndarray`. Subclasses can override the
        `process_nd(…)` method to receive, transform, and return
        the image data using NumPy, SciPy, and the like.
    """
    __slots__ = tuple()
    
    def process(self, image):
        """ NDProcessor.process(…) converts its PIL image operand
            to a `numpy.ndarray`, hands it off to the delegate
            method NDProcessor.process_nd(…), and converts whatever
            that method call returns back to a PIL image before
            finally returning it.
        """
        return toimage(self.process_nd(fromimage(image)))
    
    @staticmethod
    def compand(ndimage):
        """ The NDProcessor.compand(…) static method scales a
            `numpy.ndarray` with floating-point values from 0.0»1.0
            to unsigned 8-bit integer values from 0»255.
        """
        return uint8_t(float32_t(ndimage) * 255.0)
    
    @staticmethod
    def uncompand(ndimage):
        """ The NDProcessor.uncompand(…) static method scales a
            `numpy.ndarray` with unsigned 8-bit integer values
            from 0»255 to floating-point values from 0.0»1.0.
        """
        return float32_t(ndimage) / 255.0

# Assign the modules’ `__all__` and `__dir__` using the exporter:
__all__, __dir__ = exporter.all_and_dir()

def test():
    """ Tests for bytescale(¬) adapted from `scipy.misc.pilutil` doctests,
        q.v. https://git.io/fhkHI supra.
    """
    from instakit.utils.static import asset
    
    image_paths = list(map(
        lambda image_file: asset.path('img', image_file),
            asset.listfiles('img')))
    image_inputs = list(map(
        lambda image_path: Mode.RGB.open(image_path),
            image_paths))
    
    # print()
    print("«TESTING: BYTESCALE UTILITY FUNCTION»")
    
    image = numpy.array((91.06794177,   3.39058326,  84.4221549,
                         73.88003259,  80.91433048,   4.88878881,
                         51.53875334,  34.45808177,  27.5873488)).reshape((3, 3))
    assert numpy.all(
           bytescale(image) == numpy.array((255,   0, 236,
                                            205, 225,   4,
                                            140,  90,  70),
                                           dtype=uint8_t).reshape((3, 3)))
    assert numpy.all(
           bytescale(image,
                     high=200,
                     low=100) == numpy.array((200, 100, 192,
                                              180, 188, 102,
                                              155, 135, 128),
                                             dtype=uint8_t).reshape((3, 3)))
    assert numpy.all(
           bytescale(image,
                     cmin=0,
                     cmax=255) == numpy.array((91,  3, 84,
                                               74, 81,  5,
                                               52, 34, 28),
                                              dtype=uint8_t).reshape((3, 3)))
    print("«SUCCESS»")
    print()
    
    # print()
    print("«TESTING: FROMIMAGE»")
    
    image_arrays = list(map(
        lambda image_input: fromimage(image_input),
            image_inputs))
    
    for idx, image_array in enumerate(image_arrays):
        assert image_array.dtype == uint8_t
        assert image_array.shape[0] == image_inputs[idx].size[1]
        assert image_array.shape[1] == image_inputs[idx].size[0]
    
    print("«SUCCESS»")
    print()

if __name__ == '__main__':
    test()