#!/usr/bin/env python
# encoding: utf-8
"""
noise.py

Created by FI$H 2000 on 2014-05-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""
from __future__ import print_function

from instakit.utils.ndarrays import NDProcessor

class Noise(NDProcessor):
    """ Base noise processor (defaults to 'localvar' mode) """
    
    mode = 'localvar'
    
    def process_ndimage(self, ndimage):
        from skimage.util import random_noise
        return self.compand(
               random_noise(ndimage,
                            mode=type(self).mode))


class GaussianNoise(Noise):
    """ Add Gaussian noise """
    mode = 'gaussian'

class PoissonNoise(Noise):
    """ Add Poisson-distributed noise """
    mode = 'poisson'

class GaussianLocalVarianceNoise(Noise):
    """ Add Gaussian noise, with image-dependant local variance """
    pass

class SaltNoise(Noise):
    """ Add 'salt noise' -- replace random pixel values with 1.0f (255) """
    mode = 'salt'

class PepperNoise(Noise):
    """ Add 'pepper noise' -- replace random pixel values with zero """
    mode = 'pepper'

class SaltAndPepperNoise(Noise):
    """ Add 'salt and pepper noise' -- replace random pixel values with 1.0f (255) or zero """
    mode = 's&p'

class SpeckleNoise(Noise):
    """ Add multiplicative noise using out = image + n*image
        (where n is uniform noise with specified mean & variance) """
    mode = 'speckle'


if __name__ == '__main__':
    from PIL import Image
    from instakit.utils import static
    
    image_paths = list(map(
        lambda image_file: static.path('img', image_file),
            static.listfiles('img')))
    image_inputs = list(map(
        lambda image_path: Image.open(image_path).convert('RGB'),
            image_paths))
    
    noises = [
        GaussianNoise, PoissonNoise, GaussianLocalVarianceNoise,
        SaltNoise, PepperNoise, SaltAndPepperNoise, SpeckleNoise
    ]
    
    for idx, image_input in enumerate(image_inputs + image_inputs[:2]):
        image_input.show()
        #Noise().process(image_input).show()
        noises[idx]().process(image_input).show()
    
    print(image_paths)
    
