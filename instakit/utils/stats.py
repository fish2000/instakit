#!/usr/bin/env python
# encoding: utf-8
"""
stats.py

Created by FI$H 2000 on 2018-12-24.
Copyright (c) 2018 Objects In Space And Time, LLC. All rights reserved.
"""
from __future__ import print_function
from instakit.utils.mode import Mode

def pixel_count(image):
    """ Return the number of pixels in the input image. """
    width, height = image.size
    return width * height

def pixel_sum(image):
    """ Return the sum of the input images’ histogram values. """
    histogram = Mode.L.process(image).histogram()
    out = 0.0
    for idx, val in enumerate(histogram):
        out += idx * val
    return out

def pixel_mean(image):
    """ Return the mean of the input images’ histogram values. """
    return pixel_sum(image) / pixel_count(image)