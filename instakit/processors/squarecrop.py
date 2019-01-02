#!/usr/bin/env python
# encoding: utf-8
"""
squarecrop.py

Created by FI$H 2000 on 2012-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""
from __future__ import print_function

def histogram_entropy(image):
    """ Calculate the entropy of an images' histogram.
        Used for "smart cropping" in easy-thumbnails:
            https://git.io/fhqxd
    """
    from math import log2, fsum
    
    if not hasattr(image, 'histogram'):
        return 0  # Fall back to a constant entropy.
    
    histogram = image.histogram()
    histosum = float(sum(histogram))
    histonorm = (histocol / histosum for histocol in histogram)
    
    return -fsum(p * log2(p) for p in histonorm if p != 0.0)

def compare_entropy(start_slice, end_slice, slice, difference):
    """ Calculate the entropy of two slices (from the start
        and end of an axis), returning a tuple containing
        the amount that should be added to the start
        and removed from the end of the axis.
        
        Based on the eponymous function from easy-thumbnails:
            https://git.io/fhqpT
    """
    start_entropy = histogram_entropy(start_slice)
    end_entropy = histogram_entropy(end_slice)

    if end_entropy and abs(start_entropy / end_entropy - 1) < 0.01:
        # Less than 1% difference, remove from both sides.
        if difference >= slice * 2:
            return slice, slice
        half_slice = slice // 2
        return half_slice, slice - half_slice
    
    if start_entropy > end_entropy:
        return 0, slice
    else:
        return slice, 0

class SquareCrop(object):
    """ Crop an image to an Instagrammy square,
        by whittling away the parts of the image
        with the least entropy.
        
        Based on smart crop implementation from easy-thumbnails:
            https://git.io/fhqxj
    """
    
    def process(self, image):
        source_x, source_y = image.size
        target_width = target_height = min(image.size)
        
        diff_x = int(source_x - min(source_x, target_width))
        diff_y = int(source_y - min(source_y, target_height))
        left = top = 0
        right, bottom = source_x, source_y
        
        while diff_x:
            slice = min(diff_x, max(diff_x // 5, 10))
            start = image.crop((left, 0, left + slice, source_y))
            end = image.crop((right - slice, 0, right, source_y))
            add, remove = compare_entropy(start, end, slice, diff_x)
            left += add
            right -= remove
            diff_x = diff_x - add - remove
        
        while diff_y:
            slice = min(diff_y, max(diff_y // 5, 10))
            start = image.crop((0, top, source_x, top + slice))
            end = image.crop((0, bottom - slice, source_x, bottom))
            add, remove = compare_entropy(start, end, slice, diff_y)
            top += add
            bottom -= remove
            diff_y = diff_y - add - remove
        
        box = (left, top, right, bottom)
        return image.crop(box)


if __name__ == '__main__':
    from instakit.utils import static
    from instakit.utils.mode import Mode
    
    image_paths = list(map(
        lambda image_file: static.path('img', image_file),
            static.listfiles('img')))
    image_inputs = list(map(
        lambda image_path: Mode.RGB.open(image_path),
            image_paths))
    
    for image_input in image_inputs:
        image_input.show()
        SquareCrop().process(image_input).show()
    
    print(image_paths)
    
