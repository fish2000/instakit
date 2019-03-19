#!/usr/bin/env python
# encoding: utf-8
"""
                                                                                
                        d8888 888888b.    .d8888b.                              
      o                d88888 888  "88b  d88P  Y88b                    o        
     d8b              d88P888 888  .88P  888    888                   d8b       
    d888b            d88P 888 8888888K.  888        .d8888b          d888b      
"Y888888888P"       d88P  888 888  "Y88b 888        88K          "Y888888888P"  
  "Y88888P"        d88P   888 888    888 888    888 "Y8888b.       "Y88888P"    
  d88P"Y88b       d8888888888 888   d88P Y88b  d88P      X88       d88P"Y88b    
 dP"     "Yb     d88P     888 8888888P"   "Y8888P"   88888P'      dP"     "Yb   
                                                                                
                                                                                
Instakit’s Abstract Base Classes – née ABCs – for processors and data structures

"""
from __future__ import print_function

from pkgutil import extend_path
from abc import ABC, abstractmethod as abstract
from collections import defaultdict
from enum import Enum as EnumBase
from itertools import chain

if '__path__' in locals():
    __path__ = extend_path(__path__, __name__)

__all__ = ('is_in_class', 'subclasshook',
           'Processor', 'Enum', 'NOOp',
           'Container', 'MutableContainer',
           'Fork',
           'ThresholdMatrixProcessor',
           'NDProcessorBase')

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

def subclasshook(cls, subclass):
    """ A subclass hook function for both Processor and Enum """
    if subclass in (Processor, Enum):
        if any(is_in_class('process', ancestor) for ancestor in subclass.__mro__):
            return True
    return NotImplemented

def slots_for(cls):
    """ Get the summation of the __slots__ tuples for a class and its ancestors """
    # q.v. https://stackoverflow.com/a/6720815/298171
    return tuple(chain.from_iterable(
                 getattr(ancestor, '__slots__', tuple()) \
                           for ancestor in cls.__mro__))

def compare_via_attrs(self, other):
    """ Compare two processors:
        1) Return NotImplemented if the types do not exactly match.
        2) For processors using __dict__ mappings for attributes,
           compare them directly.
        3) For processors using __slots__ for attributes,
           iterate through all ancestor slot names using “slots_for(…)”
           and return False if any compare inequal between self and other --
           ultimately returning True.
        4) If the slots/dict situation differs between the two instances,
           raise a TypeError.
    """
    if type(self) is not type(other):
        return NotImplemented
    if hasattr(self, '__dict__') and hasattr(other, '__dict__'):
        return self.__dict__ == other.__dict__
    elif hasattr(self, '__slots__') and hasattr(other, '__slots__'):
        slots = slots_for(type(self))
        for slot in slots:
            if getattr(self, slot) != getattr(other, slot):
                return False
        return True
    raise TypeError("dict/slots mismatch")

class Processor(ABC):
    
    """ Base abstract processor class. """
    __slots__ = tuple()
    
    @abstract
    def process(self, image):
        """ Process an image instance, per the processor instance,
            returning the processed image data.
        """
        ...
    
    def __call__(self, image):
        return self.process(image)
    
    @classmethod
    def __subclasshook__(cls, subclass):
        return subclasshook(cls, subclass)
    
    def __eq__(self, other):
        """ Delegate to “compare_via_attrs(…)” """
        return compare_via_attrs(self, other)

class Enum(EnumBase):
    
    """ Base abstract processor enum. """
    __slots__ = tuple()
    
    @abstract
    def process(self, image):
        """ Process an image instance, per the processor enum instance,
            returning the processed image data.
        """
        ...
    
    @classmethod
    def __subclasshook__(cls, subclass):
        return subclasshook(cls, subclass)

class NOOp(Processor):
    
    """ A no-op processor. """
    __slots__ = tuple()
    
    def process(self, image):
        """ Return the image instance, unchanged """
        return image
    
    def __eq__(self, other):
        """ Simple type-comparison """
        return type(self) is type(other)

class Container(Processor):
    
    """ Base abstract processor container. """
    __slots__ = tuple()
    
    @classmethod
    @abstract
    def base_type(cls):
        """ Return the internal type upon which this instakit.abc.Container
            subclass is based.
        """
        ...
    
    @abstract
    def iterate(self):
        """ Return an ordered iterable of sub-processors. """
        ...
    
    @abstract
    def __len__(self): ...
    
    @abstract
    def __contains__(self, value): ...
    
    @abstract
    def __getitem__(self, idx): ...
    
    def __bool__(self):
        """ A processor container is considered Truthy if it contains values,
            and Falsey if it is empty.
        """
        return len(self) > 0
    
    def __eq__(self, other):
        """ Compare “base_type()” results and item-by-item through “iterate()” """
        if type(self).base_type() is not type(self).base_type():
            return NotImplemented
        for self_item, other_item in zip(self.iterate(),
                                         other.iterate()):
            if self_item != other_item:
                return False
        return True

class Mapping(Container):
    
    __slots__ = tuple()
    
    @abstract
    def get(self, idx, default_value): ...

class Sequence(Container):
    
    __slots__ = tuple()
    
    @abstract
    def index(self, value): ...
    
    @abstract
    def last(self): ...

class MutableContainer(Container):
    
    """ Base abstract processor mutable container. """
    __slots__ = tuple()
    
    @abstract
    def __setitem__(self, idx, value): ...
    
    @abstract
    def __delitem__(self, idx, value): ...

class MutableMapping(MutableContainer):
    
    __slots__ = tuple()
    
    @abstract
    def get(self, idx, default_value): ...
    
    @abstract
    def pop(self, idx, default_value): ...
    
    @abstract
    def update(self, iterable=None, **kwargs): ...

class MutableSequence(MutableContainer):
    
    __slots__ = tuple()
    
    @abstract
    def index(self, value): ...
    
    @abstract
    def last(self): ...
    
    @abstract
    def append(self, value): ...
    
    @abstract
    def extend(self, iterable): ...
    
    @abstract
    def pop(self, idx=-1): ...

class Fork(MutableMapping):
    
    """ Base abstract forking processor. """
    __slots__ = ('dict', '__weakref__')
    
    @classmethod
    def base_type(cls):
        return defaultdict
    
    def __init__(self, default_factory, *args, **kwargs):
        """ The `Fork` ABC implements the same `__init__(¬)` call signature as
            its delegate type, `collections.defaultdict`. A “default_factory”
            callable argument is required to fill in missing values (although
            one can pass None, which will cause a `NOOp` processor to be used).
            
            From the `collections.defaultdict` docstring:
            
           “defaultdict(default_factory[, ...]) --> dict with default factory”
            
           “The default factory is called without arguments to produce
            a new value when a key is not present, in __getitem__ only.
            A defaultdict compares equal to a dict with the same items.
            All remaining arguments are treated the same as if they were
            passed to the dict constructor, including keyword arguments.”
        
        """
        if default_factory in (None, NOOp):
            default_factory = NOOp
        if not callable(default_factory):
            raise AttributeError("Fork() requires a callable default_factory")
        
        self.dict = type(self).base_type()(default_factory, *args, **kwargs)
        super(Fork, self).__init__()
    
    @property
    def default_factory(self):
        """ The default factory for the dictionary. """
        return self.dict.default_factory
    
    @default_factory.setter
    def default_factory(self, value):
        if not callable(value):
            raise AttributeError("Fork.default_factory requires a callable value")
        self.dict.default_factory = value
    
    def __len__(self):
        """ The number of entries in the dictionary.
            See defaultdict.__len__(…) for details.
        """
        return len(self.dict)
    
    def __contains__(self, idx):
        """ True if the dictionary has the specified `idx`, else False.
            See defaultdict.__contains__(…) for details.
        """
        return idx in self.dict
    
    def __getitem__(self, idx):
        """ Get a value from the dictionary, or if no value is present,
            the return value of `default_factory()`.
            See defaultdict.__getitem__(…) for details.
        """
        return self.dict[idx]
    
    def __setitem__(self, idx, value):
        """ Set the value in the dictionary corresponding to the specified
           `idx` to the value passed, or if a value of “None” was passed,
            set the value to `instakit.abc.NOOp()` -- the no-op processor.
        """
        if value in (None, NOOp):
            value = NOOp()
        self.dict[idx] = value
    
    def __delitem__(self, idx):
        """ Delete a value from the dictionary corresponding to the specified
           `idx`, if one is present.
            See defaultdict.__delitem__(…) for details.
        """
        del self.dict[idx]
    
    def get(self, idx, default_value=None):
        """ Get a value from the dictionary, with an optional default
            value to use should a value not be present for this `idx`.
            See defaultdict.get(…) for details.
        """
        return self.dict.get(idx, default_value)
    
    def pop(self, idx, default_value=None):
        """ D.pop(idx[,d]) -> v, remove specified `idx` and return the corresponding value.
            If `idx` is not found, d is returned if given, otherwise `KeyError` is raised.
            See defaultdict.pop(…) for details.
        """
        return self.dict.pop(idx, default_value)
    
    def update(self, iterable=None, **kwargs):
        """ Update the dictionary with new key-value pairs.
            See defaultdict.update(…) for details.
        """
        self.dict.update(iterable or tuple(), **kwargs)
    
    @abstract
    def split(self, image): ...
    
    @abstract
    def compose(self, *bands): ...

class ThresholdMatrixProcessor(Processor):
    
    """ Abstract base class for a processor using a uint8 threshold matrix """
    # This is used in instakit.processors.halftone
    __slots__ = ('threshold_matrix',)
    
    LO_TUP = (0,)
    HI_TUP = (255,)
    
    def __init__(self, threshold = 128.0):
        """ Initialize with a threshold value between 0 and 255 """
        self.threshold_matrix = int(threshold)  * self.LO_TUP + \
                           (256-int(threshold)) * self.HI_TUP

class NDProcessorBase(Processor):
    
    """ An image processor ancestor class that represents PIL image
        data in a `numpy.ndarray`. This is the base abstract class,
        specifying necessary methods for subclasses to override.
        
        Note that “process(…)” has NOT been implemented yet in the
        inheritance chain – a subclass will need to furnish it.
    """
    __slots__ = tuple()
    
    @abstract
    def process_nd(self, ndimage):
        """ Override NDProcessor.process_nd(…) in subclasses
            to provide functionality that acts on image data stored
            in a `numpy.ndarray`.
        """
        ...
    
    @staticmethod
    @abstract
    def compand(ndimage): ...
    
    @staticmethod
    @abstract
    def uncompand(ndimage): ...


def test():
    """ Inline tests for instakit.abc module """
    import os
    if os.environ.get('TM_PYTHON'):
        import sys
        def print_red(text):
            print(text, file=sys.stderr)
    else:
        import colorama, termcolor
        colorama.init()
        def print_red(text):
            print(termcolor.colored(text, color='red'))
    
    import __main__
    print_red(__main__.__doc__)
    
    class SlowAtkinson(ThresholdMatrixProcessor):
        __slots__ = tuple()
        def process(self, image):
            from instakit.utils.mode import Mode
            image = Mode.L.process(image)
            for y in range(image.size[1]):
                for x in range(image.size[0]):
                    old = image.getpixel((x, y))
                    new = self.threshold_matrix[old]
                    err = (old - new) >> 3 # divide by 8.
                    image.putpixel((x, y), new)
                    for nxy in [(x+1, y),
                                (x+2, y),
                                (x-1, y+1),
                                (x, y+1),
                                (x+1, y+1),
                                (x, y+2)]:
                        try:
                            image.putpixel(nxy, int(
                            image.getpixel(nxy) + err))
                        except IndexError:
                            pass
            return image
    
    from pprint import pprint
    slow_atkinson = SlowAtkinson()
    pprint(slow_atkinson)
    print("DICT?", hasattr(slow_atkinson, '__dict__'))
    print("SLOTS?", hasattr(slow_atkinson, '__slots__'))
    pprint(slow_atkinson.__slots__)
    pprint(slow_atkinson.__class__.__base__.__slots__)
    pprint(slots_for(SlowAtkinson))
    print("THRESHOLD_MATRIX:", slow_atkinson.threshold_matrix)
    assert slow_atkinson == SlowAtkinson()
    assert NOOp() == NOOp()

if __name__ == '__main__':
    test()