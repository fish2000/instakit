#!/usr/bin/env python
# encoding: utf-8
"""
blur.py

Created by FI$H 2000 on 2012-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""
from instakit.utils import kernels

class GaussianBlur(object):
    """ Gaussian Blur """
    
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
        image_input.show()
        GaussianBlur(n=3).process(image_input).show()
    
    print image_paths
    
