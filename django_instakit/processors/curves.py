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

from os.path import join
from struct import unpack
from scipy import interpolate
from PIL import Image
import numpy

from django.contrib.staticfiles.finders import \
    AppDirectoriesFinder


class Channel(list):
    def __init__(self, name, *args):
        self.name = name
        list.__init__(self, *args)
    
    def asarray(self, dtype=numpy.uint8):
        return numpy.array(self)
    
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
    
    def __init__(self, name):
        self.curves = []
        self.name = name
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
        for j in xrange(points_in_curve):
            y, x = unpack("!hh", acv_file.read(4))
            curve.append((x, y))
        #curve.interpolate()
        curve.lagrange()
        return curve
    
    def read_acv(self, name):
        print "Reading curves from %s.acv" % name
        acv_path = AppDirectoriesFinder().storages.get(
            'django_instakit').path(join(
                'django_instakit', 'acv',
                "%s.acv" % name))
        with open(acv_path, "rb") as acv_file:
            _, self.count = unpack("!hh", acv_file.read(4))
            for i in xrange(self.count):
                self.curves.append(
                    self.read_one_curve(
                        acv_file, self.channel_name(i)))
    
    def process(self, img):
        if img.mode not in ('RGB','1','L'):
            img.convert('RGB')
        if img.mode is '1':
            img.convert('L')
        if img.mode is 'L':
            return Image.eval(img, self.curves[0])
        # has to be RGB at this point
        img_channels = img.split()
        img_adjusted_channels = []
        for i in xrange(len(img_channels)):
            img_adjusted_channels.append(
                Image.eval(
                    img_channels[i],
                    lambda v: self.curves[i+1](v)))
        return Image.merge('RGB', img_adjusted_channels)


if __name__ == '__main__':
    curve_files = AppDirectoriesFinder().storages.get(
        'django_instakit').listdir(join(
            'django_instakit', 'acv'))[-1]
    curve_names = [curve_file.rstrip('.acv') for curve_file in curve_files]
    curve_sets = [CurveSet(name) for name in curve_names if not name.lower() == '.ds_store']

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
        for curve_set in curve_sets:
            curve_set.process(image_input).show()
    
    print curve_sets
    print image_paths
    