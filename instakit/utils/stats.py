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
    width, height = Mode.L.process(image).size
    return width * height

def pixel_sum(image):
    histogram = Mode.L.process(image).histogram()
    out = 0.0
    for idx, val in enumerate(histogram):
        out += idx * val
    return out

def pixel_mean(image):
    image = Mode.L.process(image)
    return pixel_sum(image) / pixel_count(image)