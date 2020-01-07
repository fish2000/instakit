#!/usr/bin/env python
# encoding: utf-8
"""
noise.py

Created by FI$H 2000 on 2014-05-23.
Copyright (c) 2012-2019 Objects In Space And Time, LLC. All rights reserved.
"""
from __future__ import print_function
from enum import Enum, unique

from instakit.utils.ndarrays import NDProcessor

@unique
class NoiseMode(Enum):
    
    LOCALVAR        = 'localvar'
    GAUSSIAN        = 'gaussian'
    POISSON         = 'poisson'
    SALT            = 'salt'
    PEPPER          = 'pepper'
    SALT_N_PEPPER   = 's&p'
    SPECKLE         = 'speckle'
    
    def to_string(self):
        return str(self.value)
    
    def __str__(self):
        return self.to_string()
    
    def process_nd(self, ndimage, **kwargs):
        from skimage.util import random_noise
        return random_noise(ndimage,
                            mode=self.to_string(),
                          **kwargs)


class Noise(NDProcessor):
    """ Base noise processor
        -- defaults to “localvar” mode; q.v. `GaussianLocalVarianceNoise` sub.
    """
    mode = NoiseMode.LOCALVAR
    
    def process_nd(self, ndimage):
        noisemaker = type(self).mode
        return self.compand(noisemaker.process_nd(ndimage))


class GaussianNoise(Noise):
    """ Add Gaussian noise """
    mode = NoiseMode.GAUSSIAN

class PoissonNoise(Noise):
    """ Add Poisson-distributed noise """
    mode = NoiseMode.POISSON

class GaussianLocalVarianceNoise(Noise):
    """ Add Gaussian noise, with image-dependant local variance """
    pass

class SaltNoise(Noise):
    """ Add “salt noise”
        -- replace random pixel values with 1.0f (255)
    """
    mode = NoiseMode.SALT

class PepperNoise(Noise):
    """ Add “pepper noise”
        -- replace random pixel values with zero
    """
    mode = NoiseMode.PEPPER

class SaltAndPepperNoise(Noise):
    """ Add “salt and pepper noise”
        -- replace random pixel values with either 1.0f (255) or zero
    """
    mode = NoiseMode.SALT_N_PEPPER

class SpeckleNoise(Noise):
    """ Add “speckle noise”
        --- multiplicative noise using `out = image + n * image`
           (where `n` is uniform noise with specified mean + variance)
    """
    mode = NoiseMode.SPECKLE

def test():
    from instakit.utils.static import asset
    from instakit.utils.mode import Mode
    
    image_paths = list(map(
        lambda image_file: asset.path('img', image_file),
            asset.listfiles('img')))
    image_inputs = list(map(
        lambda image_path: Mode.RGB.open(image_path),
            image_paths))
    
    noises = [GaussianNoise,
              PoissonNoise,
              GaussianLocalVarianceNoise,
              SaltNoise,
              PepperNoise,
              SaltAndPepperNoise,
              SpeckleNoise]
    
    for idx, image_input in enumerate(image_inputs + image_inputs[:2]):
        for NoiseProcessor in noises:
            NoiseProcessor().process(image_input).show()
    
    print(image_paths)

if __name__ == '__main__':
    test()