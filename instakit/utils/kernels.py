#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function

from instakit.exporting import Exporter

exporter = Exporter(path=__file__)
export = exporter.decorator()

@export
def gaussian(size, sizeY=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    from scipy import mgrid, exp
    sizeX = int(size)
    if not sizeY:
        sizeY = sizeX
    else:
        sizeY = int(sizeY)
    x, y = mgrid[-sizeX:sizeX+1,
                 -sizeY:sizeY+1]
    g = exp(-(x**2/float(sizeX)+y**2/float(sizeY)))
    return (g / g.sum()).flatten()

@export
def gaussian_blur_kernel(ndim, kernel):
    from scipy.ndimage import convolve
    return convolve(ndim, kernel, mode='reflect')

@export
def gaussian_blur(ndim, sigma=3, sizeY=None):
    return gaussian_blur_kernel(ndim,
                                gaussian(sigma,
                                sizeY=sizeY))

@export
def gaussian_blur_filter(input, sigmaX=3,
                                sigmaY=3,
                                sigmaZ=0):
    from scipy.ndimage.filters import gaussian_filter
    if not sigmaY:
        sigmaY = sigmaX
    if not sigmaZ:
        sigmaZ = sigmaX
    return gaussian_filter(input, sigma=(sigmaX,
                                         sigmaY,
                                         sigmaZ), order=0,
                                                  mode='reflect')

# Assign the modules’ `__all__` and `__dir__` using the exporter:
__all__, __dir__ = exporter.all_and_dir()
