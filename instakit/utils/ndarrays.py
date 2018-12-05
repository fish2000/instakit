#!/usr/bin/env python
# encoding: utf-8
"""
ndarrays.py

Created by FI$H 2000 on 2013-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""

from __future__ import division

import numpy
from PIL import Image

class NDProcessor(object):
    
    def process(self, img):
        return Image.fromarray(
            self.process_ndimage(
                numpy.asarray(img)))
    
    def process_ndimage(self, ndimage):
        """ Override me! """
        return ndimage
    
    @staticmethod
    def compand(ndimage):
        return numpy.uint8(
            numpy.float32(ndimage) * 255.0)
    
    @staticmethod
    def uncompand(ndimage):
        return numpy.float32(ndimage) / 255.0