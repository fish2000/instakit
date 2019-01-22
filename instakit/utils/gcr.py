#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function

from instakit.utils.mode import Mode

RGB = Mode.RGB.value
CMYK = Mode.CMYK.value
cmyk = CMYK.mode

PERCENT_ADMONISHMENT = "Do you not know how percents work??!"

def gcr(image, percentage=20, revert_mode=False):
    ''' basic “Gray Component Replacement” function. Returns a CMYK image* with 
        percentage gray component removed from the CMY channels and put in the
        K channel, ie. for percentage=100, (41, 100, 255, 0) >> (0, 59, 214, 41).
        
    {*} This is the default behavior – to return an image of the same mode as that
        of which was originally provided, pass the value for the (optional) keyword
        argument `revert_mode` as `True`.
    '''
    # from http://stackoverflow.com/questions/10572274/halftone-images-in-python
    
    if percentage is None:
        return revert_mode and image or Mode.CMYK.process(image)
    
    if percentage > 100 or percentage < 1:
        raise ValueError(PERCENT_ADMONISHMENT)
    
    percent = percentage / 100
    
    original_mode = Mode.of(image)
    cmyk_channels = Mode.CMYK.process(image).split() # no-op for images already in CMYK mode
    
    cmyk_image = []
    for channel in cmyk_channels:
        cmyk_image.append(channel.load())
    
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            gray = int(min(cmyk_image[0][x, y],
                           cmyk_image[1][x, y],
                           cmyk_image[2][x, y]) * percent)
            cmyk_image[0][x, y] -= gray
            cmyk_image[1][x, y] -= gray
            cmyk_image[2][x, y] -= gray
            cmyk_image[3][x, y] = gray
    
    out = Mode.CMYK.merge(*cmyk_channels)
    
    if revert_mode:
        return original_mode.process(out)
    return out


class BasicGCR(object):
    
    __doc__ = gcr.__doc__
    
    def __init__(self, percentage=20, revert_mode=False):
        if percentage is None:
            raise ValueError(PERCENT_ADMONISHMENT)
        if percentage > 100 or percentage < 1:
            raise ValueError(PERCENT_ADMONISHMENT)
        self.percentage = percentage
        self.revert_mode = revert_mode
    
    def process(self, image):
        return gcr(image, percentage=self.percentage,
                          revert_mode=self.revert_mode)


if __name__ == '__main__':
    from instakit.utils.static import asset
    
    image_paths = list(map(
        lambda image_file: asset.path('img', image_file),
            asset.listfiles('img')))
    image_inputs = list(map(
        lambda image_path: Mode.RGB.open(image_path),
            image_paths))
    
    for image_input in image_inputs:
        gcred = gcr(image_input)
        assert gcred.mode == CMYK.mode == cmyk
        assert Mode.of(gcred) is Mode.CMYK
        gcred.show()
    
    print(image_paths)
