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

import numpy, os
from PIL import Image
from enum import Enum, unique
from scipy import interpolate
from struct import unpack

from instakit.utils import static

interpolate_mode_strings = ('linear',
                            'nearest',
                            'zero',
                            'slinear',
                            'quadratic', 'cubic',
                            'previous', 'next',
                            'lagrange')

@unique
class InterpolateMode(Enum):
    
    # These correspond to the “kind” arg
    # from “scipy.interpolate.interp1d(…)”:
    LINEAR = 0
    NEAREST = 1
    ZERO = 2
    SLINEAR = 3
    QUADRATIC = 4
    CUBIC = 5
    PREVIOUS = 6
    NEXT = 7
    
    # This specifies LaGrange interpolation,
    # using “scipy.interpolate.lagrange(…)”:
    LAGRANGE = 8
    
    def to_string(self):
        return interpolate_mode_strings[self.value]
    
    def __str__(self):
        return self.to_string()


class SingleCurve(list):
    
    def __init__(self, name, *args):
        self.name = name
        list.__init__(self, *args)
    
    def asarray(self, dtype=None):
        return numpy.array(self, dtype=dtype)
    
    def interpolate(self, mode=InterpolateMode.LAGRANGE):
        xy = self.asarray()
        if mode == InterpolateMode.LAGRANGE or mode is None:
            delegate = interpolate.lagrange(xy.T[0],
                                            xy.T[1])
        else:
            kind = InterpolateMode(mode).to_string()
            delegate = interpolate.interp1d(xy.T[0],
                                            xy.T[1], kind=kind)
        self.delegate = delegate
        return self
    
    def __call__(self, value):
        if not hasattr(self, 'delegate'):
            self.interpolate()
        delegate = self.delegate
        return delegate(value)


class CurveSet(object):
    
    acv = 'acv'
    dotacv = '.' + acv
    channels = ('composite', 'red', 'green', 'blue')
    valid_modes = ('RGB', '1', 'L')
    
    @classmethod
    def builtin(cls, name):
        print("Reading curves [builtin]: %s%s" % (name, cls.dotacv))
        acv_path = static.path(cls.acv, "%s%s" % (name, cls.dotacv))
        out = cls(acv_path)
        out._is_builtin = True
        return out
    
    @classmethod
    def instakit_names(cls):
        return [curve_file.rstrip(cls.dotacv) \
            for curve_file in static.listfiles(cls.acv) \
            if curve_file.lower().endswith(cls.dotacv)]
    
    @classmethod
    def instakit_curve_sets(cls):
        return [cls.builtin(name) for name in cls.instakit_names()]
    
    @classmethod
    def channel_name(cls, idx):
        try:
            return cls.channels[idx]
        except IndexError:
            return "channel%s" % idx
    
    def __init__(self, path, interpolation_mode=None):
        object.__init__(self)
        self.count = 0
        self.curves = []
        self._is_builtin = False
        self.path = os.path.realpath(path)
        self.name = os.path.basename(path)
        self.interpolation_mode = interpolation_mode
        self.read_acv(self.path,
                      self.interpolation_mode)
    
    @property
    def is_builtin(self):
        return self._is_builtin
    
    @staticmethod
    def read_one_curve(acv_file, name, interpolation_mode):
        curve = SingleCurve(name)
        points_in_curve, = unpack("!h", acv_file.read(2))
        for _ in range(points_in_curve):
            y, x = unpack("!hh", acv_file.read(4))
            curve.append((x, y))
        return curve.interpolate(interpolation_mode)
    
    def read_acv(self, acv_path, interpolation_mode):
        with open(acv_path, "rb") as acv_file:
            _, self.count = unpack("!hh", acv_file.read(4))
            for idx in range(self.count):
                self.curves.append(
                    self.read_one_curve(acv_file,
                                   type(self).channel_name(idx),
                                        interpolation_mode))
    
    def process(self, image):
        mode = image.mode
        if mode not in type(self).valid_modes:
            image = image.convert('RGB')
        if mode == '1':
            image = image.convert('L')
        if mode == 'L':
            return Image.eval(image, self.curves[0])
        # has to be RGB at this point -- but we'll use the
        # mode of the operand image for future-proofiness:
        adjusted_channels = []
        for idx, channel in enumerate(image.split()):
            adjusted_channels.append(
                Image.eval(channel,
                           lambda v: self.curves[idx+1](v)))
        return Image.merge(mode, adjusted_channels)
    
    def __repr__(self):
        cls_name = getattr(type(self), '__qualname__',
                   getattr(type(self), '__name__'))
        address = id(self)
        label = self.is_builtin and '[builtin]' or self.name
        interp = self.interpolation_mode or InterpolateMode.LAGRANGE
        parenthetical = "%s, %d, %s" % (label, self.count, interp)
        return "%s(%s) @ <%s>" % (cls_name, parenthetical, address)


if __name__ == '__main__':
    curve_sets = CurveSet.instakit_curve_sets()
    
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
    
