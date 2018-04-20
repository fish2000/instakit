#!/usr/bin/env python
# encoding: utf-8
"""
blur.py

Created by FI$H 2000 on 2012-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""
from __future__ import print_function
from instakit.utils import kernels

from PIL.ImageFilter import UnsharpMask as PILUnsharpMask
from PIL.ImageFilter import GaussianBlur as PILGaussianBlur

from PIL.ImageFilter import CONTOUR, DETAIL, EMBOSS
from PIL.ImageFilter import EDGE_ENHANCE, EDGE_ENHANCE_MORE, FIND_EDGES
from PIL.ImageFilter import SMOOTH, SMOOTH_MORE, SHARPEN


class ImagingCoreFilterMixin(object):
    def process(self, img):
        return img.filter(self)


class Contour(CONTOUR, ImagingCoreFilterMixin):
    """ Contour-Enhance Filter """
    pass


class Detail(DETAIL, ImagingCoreFilterMixin):
    """ Detail-Enhance Filter """
    pass


class Emboss(EMBOSS, ImagingCoreFilterMixin):
    """ Emboss-Effect Filter """
    pass


class EdgeEnhance(EDGE_ENHANCE, ImagingCoreFilterMixin):
    """ Edge-Enhance Filter """
    pass


class EdgeEnhanceMore(EDGE_ENHANCE_MORE, ImagingCoreFilterMixin):
    """ Edge-Enhance (With Extreme Predjudice) Filter """
    pass


class FindEdges(FIND_EDGES, ImagingCoreFilterMixin):
    """ Edge-Finder Filter """
    pass


class Smooth(SMOOTH, ImagingCoreFilterMixin):
    """ Image-Smoothing Filter """
    pass


class SmoothMore(SMOOTH_MORE, ImagingCoreFilterMixin):
    """ Image-Smoothing (With Extreme Prejudice) Filter """
    pass


class Sharpen(SHARPEN, ImagingCoreFilterMixin):
    """ Image Sharpener """
    pass


class UnsharpMask(PILUnsharpMask, ImagingCoreFilterMixin):
    """ Unsharp Mask Filter 
        Optionally initialize with params:
            radius (2), percent (150), threshold (3) """
    pass


class SimpleGaussianBlur(PILGaussianBlur, ImagingCoreFilterMixin):
    """ Simple Gaussian Blur Filter 
        Optionally initialize with radius (2) """
    pass


class GaussianBlur(object):
    """ Gaussian Blur Filter 
        Optionally initialize with params:
            n [aka sigma] (3)
            nY (same as n)
    """
    def __init__(self, n=3, nY=None):
        self.n = n
        if nY is not None:
            self.nY = nY
        else:
            self.nY = n
    
    def process(self, img):
        import numpy
        from PIL import Image
        out = kernels.gaussian_blur_filter(
            numpy.array(img),
            sigma=self.n)
        return Image.fromarray(out)


if __name__ == '__main__':
    from PIL import Image
    from instakit.utils import static
    
    image_paths = map(
        lambda image_file: static.path('img', image_file),
            static.listfiles('img'))
    image_inputs = map(
        lambda image_path: Image.open(image_path).convert('RGB'),
            image_paths)
    
    for image_input in image_inputs:
        #image_input.show()
        #GaussianBlur(n=3).process(image_input).show()
        Contour().process(image_input).show()
        Detail().process(image_input).show()
        Emboss().process(image_input).show()
        EdgeEnhance().process(image_input).show()
        EdgeEnhanceMore().process(image_input).show()
        FindEdges().process(image_input).show()
        Smooth().process(image_input).show()
        SmoothMore().process(image_input).show()
        Sharpen().process(image_input).show()
        UnsharpMask().process(image_input).show()
        GaussianBlur(n=3).process(image_input).show()
        SimpleGaussianBlur(radius=3).process(image_input).show()
    
    print(image_paths)
    
