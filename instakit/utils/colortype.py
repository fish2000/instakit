#!/usr/bin/env python
# encoding: utf-8
"""
utils/colortype.py

Created by FI$H 2000 on 2012-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""

import numpy, imread
#from numpy import vectorize
from os.path import join
from collections import namedtuple, defaultdict

from PIL import Image
from math import floor

#fract = lambda x: x * floor(x)
#mix = lambda vX, vY, n: vX * (1.0-n) + (vY*n)

#@vectorize
def fract(x):
    return x * floor(x)

#@vectorize
def mix(vX, vY, n):
    return vX * (1.0-n) + (vY*n)

def split_abbreviations(s):
    """ If you find this function inscrutable,
        have a look here: https://gist.github.com/4027079 """
    abbreviations = []
    current_token = ''
    for char in s:
        if current_token is '':
            current_token += char
        elif char.islower():
            current_token += char
        else:
            abbreviations.append(str(current_token))
            current_token = ''
            current_token += char
    if current_token is not '':
        abbreviations.append(str(current_token))
    return abbreviations

color_types = defaultdict(lambda: {})

# hash_RGB = lambda rgb: (rgb[0]*256)**2 + (rgb[1]*256) + rgb[2]

def ColorType(name, *args, **kwargs):
    global color_types
    dtype = numpy.dtype(kwargs.pop('dtype', 'uint8')).name
    if name not in color_types[dtype]:
        channels = split_abbreviations(name)
        
        class Color(namedtuple(name, channels)):
            
            def __repr__(self):
                return "%s(dtype=%s, %s)" % (
                    self.__class__.__name__,
                    self.__class__.dtype,
                    ', '.join(['%s=%s' % (i[0], i[1]) \
                        for i in self._asdict().items()]))
            
            def __str__(self):
                return str(repr(self))
            
            def __eq__(self, other):
                if not len(other) == len(self):
                    return False
                return all([self[i] == other[i] for i in xrange(len(self))])
            
            def __hash__(self):
                if len(self) == 3:
                    return int((self[0]*256)**2 + (self[1]*256) + self[2])
                elif len(self) == 2:
                    # budget hack
                    return int((self[1]*256) + self[2])
                elif len(self) == 1:
                    return int(self[0])
                
        Color.__name__ = "%s%s" % (dtype.capitalize(), name)
        Color.dtype = numpy.dtype(dtype)
        color_types[dtype][name] = Color
    return color_types[dtype][name]

