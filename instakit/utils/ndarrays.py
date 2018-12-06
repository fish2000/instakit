#!/usr/bin/env python
# encoding: utf-8
"""
ndarrays.py

Created by FI$H 2000 on 2013-08-23.
Copyright © 2012-2019 Objects In Space And Time, LLC. All rights reserved.
The `bytescale`, `ndarray_fromimage`, and `ndarray_toimage` functions
were originally published in the `scipy.misc.pilutils` codebase.

"""

from __future__ import division

import numpy
from PIL import Image
from pprint import pformat

uint8_t = numpy.uint8
uint32_t = numpy.uint32
float32_t = numpy.float32

def bytescale(data, cmin=None, cmax=None,
                    high=255,  low=0):
    """
    Byte scales an array (image).
    
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255). If the input image already
    has dtype uint8, no scaling is done.
    
    Parameters
    ----------
    data : ndarray
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
    img_array : uint8 ndarray
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

def ndarray_fromimage(image, flatten=False, mode=None):
    """
    Return a copy of a PIL image as a numpy array.
    
    Parameters
    ----------
    im : PIL image
        Input image.
    flatten : bool
        If true, convert the output to grey-scale.
    mode : str, optional
        Mode to convert image to, e.g. ``'RGB'``.  See the Notes of the
        `imread` docstring for more details.
    
    Returns
    -------
    fromimage : ndarray
        The different colour bands/channels are stored in the
        third dimension, such that a grey-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.
    """
    if not Image.isImageType(image):
        raise TypeError("Input is not a PIL image (got %s)" % repr(image))
    
    if mode is not None:
        if mode != image.mode:
            image = image.convert(mode)
    elif image.mode == 'P':
        # Mode 'P' means there is an indexed "palette".  If we leave the mode
        # as 'P', then when we do `a = numpy.array(im)` below, `a` will be a 2D
        # containing the indices into the palette, and not a 3D array
        # containing the RGB or RGBA values.
        if 'transparency' in image.info:
            image = image.convert('RGBA')
        else:
            image = image.convert('RGB')
    
    if flatten:
        image = image.convert('F')
    elif image.mode == '1':
        # Workaround for crash in PIL. When im is 1-bit, the call numpy.array(im)
        # can cause a seg. fault, or generate garbage. See
        # https://github.com/scipy/scipy/issues/2138 and
        # https://github.com/python-pillow/Pillow/issues/350.
        # This converts im from a 1-bit image to an 8-bit image.
        image = image.convert('L')
    
    out = numpy.array(image)
    return out

_errstr = "Mode unknown or incompatible with input array shape"

def ndarray_toimage(array, high=255,  low=0,
                           cmin=None, cmax=None,
                           pal=None,
                           mode=None, channel_axis=None):
    """
    Takes a numpy array and returns a PIL image. The mode of the image returned
    depends on the array shape and the `pal` and `mode` keywords.
    
    For 2D arrays, if `pal` is a valid (N, 3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    
    .. warning::
    
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    
    Notes
    -----
    For 3D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data. If one of the dimensions is 3, the mode
    is 'RGB' by default, or 'YCbCr' if selected.
    
    The numpy array must be either 2- or 3-dimensional.
    """
    data = numpy.asarray(array)
    if numpy.iscomplexobj(data):
        raise ValueError("Cannot convert arrays of complex values")
    
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("input array lacks a suitable shape for any mode")
    
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        
        if mode == 'F':
            return Image.frombytes(mode,
                                   shape,
                                   data.astype(float32_t).tostring())
        
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high,
                                       low=low,
                                       cmin=cmin,
                                       cmax=cmax)
            image = Image.frombytes('L',
                                    shape,
                                    bytedata.tostring())
            
            if pal is not None:
                image.putpalette(numpy.asarray(pal,
                                 dtype=uint8_t).tostring()) # Becomes mode='P' automatically
            elif mode == 'P':  # default grayscale
                pal = (numpy.arange(0, 256, 1, dtype=uint8_t)[:, numpy.newaxis] *
                       numpy.ones((3,),        dtype=uint8_t)[numpy.newaxis, :])
                image.putpalette(numpy.asarray(pal,
                                 dtype=uint8_t).tostring())
            
            return image
        
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            return Image.frombytes(mode,
                                   shape,
                                   bytedata.tostring())
        
        if cmin is None:
            cmin = numpy.amin(numpy.ravel(data))
        
        if cmax is None:
            cmax = numpy.amax(numpy.ravel(data))
        
        data = (data * 1.0 - cmin) * (high - low) / (cmax - cmin) + low
        
        if mode == 'I':
            image = Image.frombytes(mode,
                                    shape,
                                    data.astype(uint32_t).tostring())
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
                    "Could not find a channel dimension (shape = %s)" % pformat(shape))
    else:
        ca = channel_axis
    
    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel dimension invalid (#channels = %s)" % numch)
    
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
            mode = 'RGB'
        else:
            mode = 'RGBA'
    
    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)
    
    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid shape for mode “%s”: %s" % (
                              mode, pformat(shape)))
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid shape for mode “%s”: %s" % (
                              mode, pformat(shape)))
    
    # Here we know both `strdata` and `mode` are correct:
    image = Image.frombytes(mode,
                            shape,
                            strdata)
    return image


class NDProcessor(object):
    
    def process(self, image):
        return ndarray_toimage(
            self.process_ndimage(
                ndarray_fromimage(image)))
    
    def process_ndimage(self, ndimage):
        """ Override me! """
        return ndimage
    
    @staticmethod
    def compand(ndimage):
        return uint8_t(float32_t(ndimage) * 255.0)
    
    @staticmethod
    def uncompand(ndimage):
        return float32_t(ndimage) / 255.0