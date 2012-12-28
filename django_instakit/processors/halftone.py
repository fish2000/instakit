#!/usr/bin/env python
# encoding: utf-8
"""
curves.py

Created by FI$H 2000 on 2012-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""
import numpy
from PIL import Image

class Atkinson(object):
    
    threshold = 128.0
    threshold_matrix = int(threshold)*[0] + (256-int(threshold))*[255]
    
    def process(self, img):
        img = img.convert('L')
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                old = img.getpixel((x, y))
                new = self.threshold_matrix[old]
                err = (old - new) >> 3 # divide by 8.
                img.putpixel((x, y), new)
                for nxy in [(x+1, y), (x+2, y), (x-1, y+1), (x, y+1), (x+1, y+1), (x, y+2)]:
                    try:
                        img.putpixel(nxy, img.getpixel(nxy) + err)
                    except IndexError:
                        pass # it happens, evidently.
        return img

class NumAtkinson(object):
    
    threshold = 128.0
    threshold_matrix = int(threshold)*[0] + (256-int(threshold))*[255]
    
    def _error(self, x, y, thresh=None):
        error_val = (int(self.omtx[x, y]) - thresh) >> 3
        try:
            self.omtx[x+1, y] += error_val
        except:
            pass
        try:
            self.omtx[x+2, y] += error_val
        except:
            pass
        try:
            self.omtx[x-1, y+1] += error_val
        except:
            pass
        try:
            self.omtx[x, y+1] += error_val
        except:
            pass
        try:
            self.omtx[x+1, y+1] += error_val
        except:
            pass
        try:
            self.omtx[x, y+2] += error_val
        except:
            pass
    
    def process(self, img):
        img = img.convert('L')
        imtx = numpy.array(img)
        self.omtx = imtx
        for y in xrange(imtx.shape[1]):
            for x in xrange(imtx.shape[0]):
                thresh = self.threshold_matrix[self.omtx[x, y]]
                self._error(x, y, thresh=thresh)
                self.omtx[x, y] = thresh
        return Image.fromarray(self.omtx, mode='L').convert('RGB')


if __name__ == '__main__':
    from django_instakit.utils import static
    
    image_paths = map(
        lambda image_file: static.path('img', image_file),
            static.listfiles('img'))
    image_inputs = map(
        lambda image_path: Image.open(image_path).convert('RGB'),
            image_paths)
    
    for image_input in image_inputs[:2]:
        #image_input.show()
        Atkinson().process(image_input).show()
        NumAtkinson().process(image_input).show()
    
    print image_paths
    