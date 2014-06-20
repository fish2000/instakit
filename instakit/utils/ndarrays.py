
from __future__ import division

#from PIL import Image

import numpy
import scipy.misc

class NDProcessor(object):
    
    def process(self, img):
        return scipy.misc.toimage(
            self.process_ndimage(
                scipy.misc.fromimage(img)))
    
    def process_ndimage(self, ndimage):
        """ Override me! """
        return ndimage
    
    def compand(self, ndimage):
        return numpy.uint8(
            numpy.float32(ndimage) * 255.0)
    
    def uncompand(self, ndimage):
        return numpy.float32(ndimage) / 255.0