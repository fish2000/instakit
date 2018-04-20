#!/usr/bin/env python
# encoding: utf-8
"""
halftone.py

Created by FI$H 2000 on 2012-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""
from __future__ import print_function

from PIL import Image
from PIL import ImageDraw
from PIL import ImageStat

from instakit.utils import pipeline
from instakit.utils.gcr import gcr


try:
    from instakit.processors.ext.halftone import Atkinson

except ImportError:
    
    class Atkinson(object):
        
        def __init__(self, threshold=128.0):
            self.threshold_matrix = int(threshold)*(0,) + (256-int(threshold))*(255,)
        
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


class DotScreen(object):
    
    def __init__(self, sample=1, scale=2, angle=0):
        self.sample = sample
        self.scale = scale
        self.angle = angle
    
    def process(self, img):
        origsize = img.size
        img = img.convert('L').rotate(self.angle, expand=1)
        size = img.size[0]*self.scale, img.size[1]*self.scale
        halftone = Image.new('L', size)
        dotscreen = ImageDraw.Draw(halftone)
        
        for x in range(0, img.size[0], self.sample):
            for y in range(0, img.size[0], self.sample):
                cropbox = img.crop(
                    (x, y, x+self.sample, y+self.sample))
                stat = ImageStat.Stat(cropbox)
                diameter = (stat.mean[0] / 255) ** 0.5
                edge = 0.5 * (1-diameter)
                xpos, ypos = (x+edge)*self.scale, (y+edge)*self.scale
                boxedge = self.sample * diameter * self.scale
                dotscreen.ellipse(
                    (xpos, ypos, xpos+boxedge, ypos+boxedge),
                    fill=255)
        
        halftone = halftone.rotate(-self.angle, expand=1)
        halfwidth, halfheight = halftone.size
        xx = (halfwidth - origsize[0]*self.scale) / 2
        yy = (halfheight - origsize[1]*self.scale) / 2
        return halftone.crop(
            (xx, yy, xx+origsize[0]*self.scale, yy+origsize[1]*self.scale))


class CMYKDotScreen(object):
    
    def __init__(self,
        gcr=20, sample=10, scale=10,
        thetaC=0, thetaM=15, thetaY=30, thetaK=45):
        
        self.gcr = max(min(100, gcr), 0)
        self.overprinter = pipeline.ChannelFork(DotScreen, mode='CMYK')
        self.overprinter.update({
            'C': DotScreen(angle=thetaC, sample=sample, scale=scale),
            'M': DotScreen(angle=thetaM, sample=sample, scale=scale),
            'Y': DotScreen(angle=thetaY, sample=sample, scale=scale),
            'K': DotScreen(angle=thetaK, sample=sample, scale=scale), })
    
    def process(self, img):
        return self.overprinter.process(
            gcr(img, self.gcr))



if __name__ == '__main__':
    from instakit.utils import static
    
    image_paths = map(
        lambda image_file: static.path('img', image_file),
            static.listfiles('img'))
    image_inputs = map(
        lambda image_path: Image.open(image_path).convert('RGB'),
            image_paths)
    
    for image_input in image_inputs:
        #image_input.show()
        Atkinson(threshold=128.0).process(image_input).show()
        #CMYKDotScreen(sample=2, scale=2).process(image_input).show()
    
    print(image_paths)
    
