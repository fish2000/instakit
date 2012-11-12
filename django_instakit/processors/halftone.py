#!/usr/bin/env python
# encoding: utf-8
"""
halftone.py

Created by FI$H 2000 on 2012-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""


class Atkinson(object):
    
    threshold = 128.0
    
    def process(self, img):
        img = img.convert('L')
        threshold_matrix = int(self.threshold)*[0] + int(self.threshold)*[255]
        
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                
                old = img.getpixel((x, y))
                new = threshold_matrix[old]
                err = (old - new) >> 3 # divide by 8.
                img.putpixel((x, y), new)
                
                for nxy in [(x+1, y), (x+2, y), (x-1, y+1), (x, y+1), (x+1, y+1), (x, y+2)]:
                    try:
                        img.putpixel(nxy, img.getpixel(nxy) + err)
                    except IndexError:
                        pass # it happens, evidently.
        
        return img


if __name__ == '__main__':
    from PIL import Image
    from os.path import join
    from django.contrib.staticfiles.finders import \
        AppDirectoriesFinder
    
    image_files = AppDirectoriesFinder().storages.get(
        'django_instakit').listdir(join(
            'django_instakit', 'img'))[-1]
    image_paths = map(
        lambda image_file: AppDirectoriesFinder().storages.get(
            'django_instakit').path(join(
                'django_instakit', 'img', image_file)), image_files)
    image_inputs = map(
        lambda image_path: Image.open(image_path).convert('RGB'),
            image_paths)
    
    for image_input in image_inputs:
        image_input.show()
        Atkinson().process(image_input).show()
    
    print image_paths
    