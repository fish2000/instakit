#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function

from pkgutil import extend_path
from abc import ABC, abstractmethod as abstract
from collections import defaultdict
from enum import Enum as EnumBase
from functools import wraps

if '__path__' in locals():
    __path__ = extend_path(__path__, __name__)

__all__ = ('is_in_class',
           'Processor', 'Enum',
           'Container', 'NOOp', 'Fork')

__dir__ = lambda: list(__all__)

def is_in_class(attr, cls):
    """ Test whether or not a class has a named attribute,
        regardless of whether the class uses `__slots__` or
        an internal `__dict__`.
    """
    if hasattr(cls, '__dict__'):
        return attr in cls.__dict__
    elif hasattr(cls, '__slots__'):
        return attr in cls.__slots__
    return False

class Processor(ABC):
    
    """ Base abstract processor class. """
    
    @abstract
    def process(self, image):
        """ Process an image instance, per the processor instance,
            returning the processed image data
        """
        ...
    
    def __call__(self, image):
        return self.process(image)
    
    @classmethod
    def __subclasshook__(cls, subclass):
        if subclass is Processor:
            if any(is_in_class('process', ancestor) for ancestor in subclass.__mro__):
                return True
        return NotImplemented

class Enum(EnumBase):
    
    """ Base abstract processor enum. """
    
    @abstract
    def process(self, image): ...

class Container(Processor):
    
    """ Base abstract processor container. """
    
    @abstract
    def iterate(self):
        """ Return an ordered iterable of sub-processors. """
        ...
    
    @classmethod
    @abstract
    def base_type(cls): ...
    
    @abstract
    def __len__(self): ...
    
    @abstract
    def __contains__(self, value): ...
    
    @abstract
    def __getitem__(self, idx): ...
    
    # Abstract but optional methods:
    
    def __setitem__(self, idx, value):
        raise NotImplementedError()
    
    def get(self, idx, default_value):
        raise NotImplementedError()
    
    def index(self, value):
        raise NotImplementedError()

class NOOp(Processor):
    
    """ A no-op processor. """
    
    def process(self, image):
        return image

class Fork(Container):
    
    """ Base abstract forking processor. """
    
    @classmethod
    def base_type(cls):
        return defaultdict
    
    def __init__(self, default_factory, *args, **kwargs):
        if default_factory in (None, NOOp):
            default_factory = NOOp
        if not callable(default_factory):
            raise AttributeError("Fork() requires a callable default_factory")
        
        self.dict = defaultdict(default_factory, **kwargs)
        super(Fork, self).__init__(*args, **kwargs)
    
    @property
    def default_factory(self):
        return self.dict.default_factory
    
    @wraps(defaultdict.__len__)
    def __len__(self):
        return len(self.dict)
    
    @wraps(defaultdict.__contains__)
    def __contains__(self, value):
        return value in self.dict
    
    @wraps(defaultdict.__getitem__)
    def __getitem__(self, idx):
        return self.dict[idx]
    
    @wraps(defaultdict.__setitem__)
    def __setitem__(self, idx, value):
        if value in (None, NOOp):
            value = NOOp()
        self.dict[idx] = value
    
    @wraps(defaultdict.get)
    def get(self, idx, default_value=None):
        return self.dict.get(idx, default_value)
    
    @abstract
    def split(self, image): ...
    
    @abstract
    def compose(self, *bands): ...

class ThresholdMatrixProcessor(Processor):
    
    """ Abstract base class for a processor using a uint8 threshold matrix """
    # This is used in instakit.processors.halftone
    
    LO_TUP = (0,)
    HI_TUP = (255,)
    
    def __init__(self, threshold = 128.0):
        """ Initialize with a threshold value between 0 and 255 """
        self.threshold_matrix = int(threshold)  * self.LO_TUP + \
                           (256-int(threshold)) * self.HI_TUP

