#!/usr/bin/env python
# encoding: utf-8
"""
stats.py

Created by FI$H 2000 on 2018-12-24.
Copyright (c) 2018 Objects In Space And Time, LLC. All rights reserved.
"""
from __future__ import print_function
from instakit.utils.mode import Mode
from instakit.exporting import Exporter

exporter = Exporter(path=__file__)
export = exporter.decorator()

@export
def pixel_count(image):
    """ Return the number of pixels in the input image. """
    width, height = image.size
    return width * height

@export
def color_count(image):
    """ Return the number of color values in the input image --
        this is the number of pixels times the band count
        of the image.
    """
    width, height = image.size
    return width * height * Mode.of(image).band_count

@export
def histogram_sum(image):
    """ Return the sum of the input images’ histogram values --
        Basically this is an optimized way of doing:
            
            out = 0.0
            histogram = Mode.L.process(image).histogram()
            for value, count in enumerate(histogram):
                out += value * count
            return out
            
        … the one-liner uses the much faster sum(…) in léu
        of looping over the histogram’s enumerated values.
    """
    histogram = Mode.L.process(image).histogram()
    return sum(value * count for value, count in enumerate(histogram))

@export
def histogram_mean(image):
    """ Return the mean of the input images’ histogram values. """
    return float(histogram_sum(image)) / pixel_count(image)

@export
def histogram_entropy_py(image):
    """ Calculate the entropy of an images' histogram. """
    from math import log2, fsum
    histosum = float(color_count(image))
    histonorm = (histocol / histosum for histocol in image.histogram())
    
    return -fsum(p * log2(p) for p in histonorm if p != 0.0)

from PIL import Image

histogram_entropy = hasattr(Image.Image, 'entropy') \
                        and Image.Image.entropy \
                        or histogram_entropy_py

export(histogram_entropy,   name='histogram_entropy')

# Assign the modules’ `__all__` and `__dir__` using the exporter:
__all__, __dir__ = exporter.all_and_dir()