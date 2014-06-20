#!/usr/bin/env python
# encoding: utf-8
"""
noise.py

Created by FI$H 2000 on 2014-05-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""

from instakit.utils.ndarrays import NDProcessor

class Noise(NDProcessor):
    
    mode = 'localvar'
    
    def process_ndimage(self, ndimage):
        from skimage.util import random_noise
        return self.compand(
            random_noise(ndimage,
                mode=self.mode))

class GaussianNoise(Noise):
    mode = 'gaussian'

class PoissonNoise(Noise):
    mode = 'poisson'

class GaussianLocalVarianceNoise(Noise):
    mode = 'localvar'

class SaltNoise(Noise):
    mode = 'salt'

class PepperNoise(Noise):
    mode = 'pepper'

class SaltAndPepperNoise(Noise):
    mode = 's&p'

class SpeckleNoise(Noise):
    mode = 'speckle'




if __name__ == '__main__':
    from PIL import Image
    from instakit.utils import static
    
    image_paths = map(
        lambda image_file: static.path('img', image_file),
            static.listfiles('img'))
    image_inputs = map(
        lambda image_path: Image.open(image_path).convert('RGB'),
            image_paths)
    
    noises = [
        GaussianNoise, PoissonNoise, GaussianLocalVarianceNoise,
        SaltNoise, PepperNoise, SaltAndPepperNoise, SpeckleNoise
    ]
    
    for idx, image_input in enumerate(image_inputs + image_inputs[:2]):
        image_input.show()
        #Noise().process(image_input).show()
        noises[idx]().process(image_input).show()
    
    print image_paths
    
