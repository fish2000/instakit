#!/usr/bin/env python
# encoding: utf-8
"""
blur.py

Created by FI$H 2000 on 2012-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""
from __future__ import print_function
from PIL import ImageFilter


class ImagingCoreFilterMixin(object):
    """ A mixin furnishing a `process(…)` method to PIL.ImageFilter classes """
    def process(self, image):
        return image.filter(self)

class Contour(ImageFilter.CONTOUR, ImagingCoreFilterMixin):
    """ Contour-Enhance Filter """
    pass


class Detail(ImageFilter.DETAIL, ImagingCoreFilterMixin):
    """ Detail-Enhance Filter """
    pass


class Emboss(ImageFilter.EMBOSS, ImagingCoreFilterMixin):
    """ Emboss-Effect Filter """
    pass


class FindEdges(ImageFilter.FIND_EDGES, ImagingCoreFilterMixin):
    """ Edge-Finder Filter """
    pass


class EdgeEnhance(ImageFilter.EDGE_ENHANCE, ImagingCoreFilterMixin):
    """ Edge-Enhance Filter """
    pass


class EdgeEnhanceMore(ImageFilter.EDGE_ENHANCE_MORE, ImagingCoreFilterMixin):
    """ Edge-Enhance (With Extreme Predjudice) Filter """
    pass


class Smooth(ImageFilter.SMOOTH, ImagingCoreFilterMixin):
    """ Image-Smoothing Filter """
    pass


class SmoothMore(ImageFilter.SMOOTH_MORE, ImagingCoreFilterMixin):
    """ Image-Smoothing (With Extreme Prejudice) Filter """
    pass


class Sharpen(ImageFilter.SHARPEN, ImagingCoreFilterMixin):
    """ Image Sharpener """
    pass


class UnsharpMask(ImageFilter.UnsharpMask, ImagingCoreFilterMixin):
    """ Unsharp Mask Filter 
        Optionally initialize with params:
            radius (2), percent (150), threshold (3) """
    pass


class SimpleGaussianBlur(ImageFilter.GaussianBlur, ImagingCoreFilterMixin):
    """ Simple Gaussian Blur Filter 
        Optionally initialize with radius (2) """
    pass


class GaussianBlur(object):
    """ Gaussian Blur Filter 
        Optionally initialize with params:
            sigmaX (3)
            sigmaY (3; same as sigmaX)
            sigmaZ (0; same as sigmaX)
    """
    def __init__(self, sigmaX=3, sigmaY=None, sigmaZ=None):
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY or sigmaX
        self.sigmaZ = sigmaZ or sigmaX
    
    def process(self, image):
        from PIL import Image
        from numpy import array
        from instakit.utils import kernels
        return Image.fromarray(kernels.gaussian_blur_filter(
                                input=array(image),
                               sigmaX=self.sigmaX,
                               sigmaY=self.sigmaY,
                               sigmaZ=self.sigmaZ))


if __name__ == '__main__':
    from instakit.utils.static import asset
    from instakit.utils.mode import Mode
    
    image_paths = list(map(
        lambda image_file: asset.path('img', image_file),
            asset.listfiles('img')))
    image_inputs = list(map(
        lambda image_path: Mode.RGB.open(image_path),
            image_paths))
    
    for image_input in image_inputs:
        # image_input.show()
        # Contour().process(image_input).show()
        # Detail().process(image_input).show()
        # Emboss().process(image_input).show()
        # EdgeEnhance().process(image_input).show()
        # EdgeEnhanceMore().process(image_input).show()
        # FindEdges().process(image_input).show()
        # Smooth().process(image_input).show()
        # SmoothMore().process(image_input).show()
        # Sharpen().process(image_input).show()
        # UnsharpMask().process(image_input).show()
        GaussianBlur(sigmaX=3).process(image_input).show()
        # SimpleGaussianBlur(radius=3).process(image_input).show()
    
    print(image_paths)
    
