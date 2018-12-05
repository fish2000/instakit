#!/usr/bin/env python
# encoding: utf-8
"""
utils/colortype.py

Created by FI$H 2000 on 2012-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""

import numpy
#from os.path import join
from collections import namedtuple, defaultdict

#from PIL import Image
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
    dtype = numpy.dtype(kwargs.pop('dtype', numpy.uint8))
    if name not in color_types[dtype.name]:
        channels = split_abbreviations(name)
        
        class Color(namedtuple(name, channels)):
            
            def __repr__(self):
                return "%s(dtype=%s, %s)" % (
                    name, self.__class__.dtype.name,
                    ', '.join(['%s=%s' % (i[0], i[1]) \
                        for i in self._asdict().items()]))
            
            def __hex__(self):
                return '0x' + "%x" * len(self) % self
            
            def __int__(self):
                return int(self.__hex__(), 16)
            
            def __long__(self):
                return numpy.long(self.__hex__(), 16)
            
            def __hash__(self):
                return self.__long__()
            
            def __eq__(self, other):
                if not len(other) == len(self):
                    return False
                return all([self[i] == other[i] for i in range(len(self))])
            
            def __unicode__(self):
                return unicode(repr(self))
            
            def composite(self):
                return numpy.dtype([
                    (k, self.__class__.dtype) for k, v in self._asdict().items()])
            
        Color.__name__ = "%s<%s>" % (name, dtype.name)
        Color.dtype = dtype
        color_types[dtype.name][name] = Color
    return color_types[dtype.name][name]

