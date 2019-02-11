#!/usr/bin/env python
# encoding: utf-8
"""
stats.py

Created by FI$H 2000 on 2018-12-24.
Copyright (c) 2018 Objects In Space And Time, LLC. All rights reserved.
"""
from __future__ import print_function

from instakit.utils.mode import Mode
from PIL import Image

def pixel_count(image):
    """ Return the number of pixels in the input image. """
    width, height = image.size
    return width * height

def histogram_sum(image):
    """ Return the sum of the input images’ histogram values. """
    histogram = Mode.L.process(image).histogram()
    out = 0.0
    for idx, val in enumerate(histogram):
        out += idx * val
    return out

def histogram_mean(image):
    """ Return the mean of the input images’ histogram values. """
    return histogram_sum(image) / pixel_count(image)

def histogram_entropy_py(image):
    """ Calculate the entropy of an images' histogram.
        Used for “smart cropping” in easy-thumbnails:
            https://git.io/fhqxd
    """
    from math import log2, fsum
    
    histogram = image.histogram()
    histosum = fsum(histogram)
    histonorm = (histocol / histosum for histocol in histogram)
    
    return -fsum(p * log2(p) for p in histonorm if p != 0.0)

histogram_entropy = hasattr(Image.Image, 'entropy') \
                        and Image.Image.entropy \
                        or histogram_entropy_py
