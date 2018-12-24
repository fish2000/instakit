#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function

from PIL import Image, ImageMode
from enum import Enum, auto

def imode(image):
    return ImageMode.getmode(image.mode)

ImageMode.getmode('RGB') # one call must be made to getmode()
                         # to properly initialize ImageMode._modes:

image_mode_strings = tuple(ImageMode._modes.keys())


class ModeAncestor(Enum):
    """
    Valid ImageMode mode strings:
    ('1',    'L',     'I',     'F',     'P',
     'RGB',  'RGBX',  'RGBA',  'CMYK',  'YCbCr',
     'LAB',  'HSV',   'RGBa',  'LA',    'La',
     'PA',   'I;16',  'I;16L', 'I;16B') """
    
    def _generate_next_value_(name,
                              start,
                              count,
                              last_values):
        return ImageMode.getmode(
               image_mode_strings[count])
    
    @classmethod
    def _missing_(cls, value):
        try:
            return cls(ImageMode.getmode(
                       image_mode_strings[value]))
        except (IndexError, TypeError):
            pass
        return super(ModeAncestor, cls)._missing_(value)
    
    @classmethod
    def is_mode(cls, instance):
        return type(instance) in cls.__mro__


class Mode(ModeAncestor):
    
    """ An enumeration class wrapping ImageMode.ModeDescriptor. """
    
    # N.B. this'll have to be manually updated,
    # whenever PIL.ImageMode gets a change pushed.
    
    MONO    = auto() # formerly ‘1’
    L       = auto()
    I       = auto()
    F       = auto()
    P       = auto()
    
    RGB     = auto()
    RGBX    = auto()
    RGBA    = auto()
    CMYK    = auto()
    YCbCr   = auto()
    
    LAB     = auto()
    HSV     = auto()
    RGBa    = auto()
    LA      = auto()
    La      = auto()
    
    PA      = auto()
    I16     = auto() # formerly ‘I;16’
    I16L    = auto() # formerly ‘I;16L’
    I16B    = auto() # formerly ‘I;16B’
    
    @classmethod
    def of(cls, image):
        for mode in cls:
            if mode.check(image):
                return mode
        raise ValueError("Image has unknown mode %s" % image.mode)
    
    @classmethod
    def for_string(cls, string):
        for mode in cls:
            if mode.to_string() == string:
                return mode
        raise ValueError("for_string(): unknown mode %s" % string)
    
    def to_string(self):
        return str(self.value)
    
    def __str__(self):
        return self.to_string()
    
    def __bytes__(self):
        return bytes(self.to_string(), encoding="UTF-8")
    
    @property
    def bands(self):
        return self.value.bands
    
    @property
    def basemode(self):
        return type(self).for_string(self.value.basemode)
    
    @property
    def basetype(self):
        return type(self).for_string(self.value.basetype)
    
    def check(self, image):
        return imode(image) is self.value
    
    def merge(self, *channels):
        return Image.merge(self.to_string(), channels)
    
    def process(self, image):
        if self.check(image):
            return image
        return image.convert(self.to_string())
    
    def new(self, size, color=0):
        return Image.new(self.to_string(), size, color=color)
    
    def open(self, fileish):
        return self.process(Image.open(fileish))
    
    def frombytes(self, size, data, decoder_name='raw', *args):
        return Image.frombytes(self.to_string(),
                               size, data, decoder_name,
                              *args)

if __name__ == '__main__':
    print(list(Mode))
    print([str(Mode.for_string(str(m))) for m in list(Mode)])
    print([(m.basemode, m.basetype) for m in list(Mode)])
    # print(Mode(10))
    assert Mode(10) == Mode.LAB
