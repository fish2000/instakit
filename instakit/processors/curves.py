#!/usr/bin/env python
# encoding: utf-8
"""
curves.py

Adapted from this:

    http://www.weemoapps.com/creating-retro-and-analog-image-filters-in-mobile-apps

And also this:

    https://github.com/WeemoApps/filteriser/blob/master/extractCurvesFromACVFile.py

Created by FI$H 2000 on 2012-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""
from __future__ import print_function

import numpy
from struct import unpack
from scipy import interpolate
from PIL import Image

from instakit.utils import static

class Channel(list):
    def __init__(self, name, *args):
        self.name = name
        list.__init__(self, *args)
    
    def asarray(self, dtype=None):
        return numpy.array(self, dtype=dtype)
    
    def lagrange(self):
        xy = self.asarray()
        delegate = interpolate.lagrange(
            xy.T[0], xy.T[1])
        self.delegate = (delegate,)
    
    def interpolate(self, kind='slinear'):
        xy = self.asarray()
        delegate = interpolate.interp1d(
            xy.T[0], xy.T[1], kind=kind)
        self.delegate = (delegate,)
    
    def __call__(self, value):
        if not self.delegate:
            #self.interpolate()
            self.lagrange()
        delegate = self.delegate[0]
        return delegate(value)


class CurveSet(object):
    
    channels = ('composite', 'red', 'green', 'blue')
    
    @classmethod
    def names(cls):
        return [curve_file.rstrip('.acv') \
            for curve_file in static.listfiles('acv') \
            if curve_file.lower().endswith('.acv')]
    
    def __init__(self, name):
        self.curves = []
        self.name = name
        self.count = 0
        object.__init__(self)
        self.read_acv(name)
    
    def channel_name(self, idx):
        try:
            return self.channels[idx]
        except IndexError:
            return "channel%s" % idx
    
    def read_one_curve(self, acv_file, name):
        curve = Channel(name)
        points_in_curve, = unpack("!h", acv_file.read(2))
        for j in range(points_in_curve):
            y, x = unpack("!hh", acv_file.read(4))
            curve.append((x, y))
        #curve.interpolate()
        curve.lagrange()
        return curve
    
    def read_acv(self, name):
        print("Reading curves from %s.acv" % name)
        acv_path = static.path(
            'acv', "%s.acv" % name)
        with open(acv_path, "rb") as acv_file:
            _, self.count = unpack("!hh", acv_file.read(4))
            for i in range(self.count):
                self.curves.append(
                    self.read_one_curve(
                        acv_file, self.channel_name(i)))
    
    def process(self, image):
        if image.mode not in ('RGB', '1', 'L'):
            image.convert('RGB')
        if image.mode is '1':
            image.convert('L')
        if image.mode is 'L':
            return Image.eval(image, self.curves[0])
        # has to be RGB at this point
        image_channels = image.split()
        image_adjusted_channels = []
        for i in range(len(image_channels)):
            image_adjusted_channels.append(
                Image.eval(
                    image_channels[i],
                    lambda v: self.curves[i+1](v)))
        return Image.merge('RGB', image_adjusted_channels)


if __name__ == '__main__':
    curve_sets = [CurveSet(nm) for nm in CurveSet.names()]
    
    image_paths = list(map(
        lambda image_file: static.path('img', image_file),
            static.listfiles('img')))
    image_inputs = list(map(
        lambda image_path: Image.open(image_path).convert('RGB'),
            image_paths))
    
    for image_input in image_inputs[:1]:
        image_input.show()
        for curve_set in curve_sets:
            curve_set.process(image_input).show()
    
    print(curve_sets)
    print(image_paths)
    
