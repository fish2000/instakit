# encoding: utf-8
from __future__ import print_function

from PIL import ImageOps, ImageChops
from collections import defaultdict, OrderedDict
from copy import copy
from enum import unique
from functools import wraps

from clu.enums import AliasingEnum, alias
from clu.mathematics import Σ
from clu.predicates import tuplize
from clu.typology import string_types

from instakit.abc import Fork, NOOp, Sequence, MutableSequence
from instakit.utils.gcr import BasicGCR
from instakit.utils.mode import Mode
from instakit.processors.adjust import AutoContrast
from instakit.exporting import Exporter

exporter = Exporter(path=__file__)
export = exporter.decorator()

if not hasattr(__builtins__, 'cmp'):
    def cmp(a, b):
        return (a > b) - (a < b)

@export
class Pipe(Sequence):
    
    """ A static linear pipeline of processors to be applied en masse.
        Derived from a `pilkit` class:
            `pilkit.processors.base.ProcessorPipeline`
    """
    __slots__ = tuplize('tuple')
    
    @classmethod
    def base_type(cls):
        return tuple
    
    @wraps(tuple.__init__)
    def __init__(self, *args):
        self.tuple = tuplize(*args)
    
    def iterate(self):
        yield from self.tuple
    
    @wraps(tuple.__len__)
    def __len__(self):
        return len(self.tuple)
    
    @wraps(tuple.__contains__)
    def __contains__(self, value):
        return value in self.tuple
    
    @wraps(tuple.__getitem__)
    def __getitem__(self, idx):
        return self.tuple[idx]
    
    @wraps(tuple.index)
    def index(self, value):
        return self.tuple.index(value)
    
    def last(self):
        if not bool(self):
            raise IndexError("pipe is empty")
        return self.tuple[-1]
    
    def process(self, image):
        for processor in self.iterate():
            image = processor.process(image)
        return image
    
    def __eq__(self, other):
        if not isinstance(other, (type(self), type(self).base_type())):
            return NotImplemented
        return super(Pipe, self).__eq__(other)

@export
class Pipeline(MutableSequence):
    
    """ A mutable linear pipeline of processors to be applied en masse.
        Derived from a `pilkit` class:
            `pilkit.processors.base.ProcessorPipeline`
    """
    __slots__ = tuplize('list')
    
    @classmethod
    def base_type(cls):
        return list
    
    @wraps(list.__init__)
    def __init__(self, *args):
        base_type = type(self).base_type()
        if len(args) == 0:
            self.list = base_type()
        if len(args) == 1:
            target = args[0]
            if type(target) is type(self):
                self.list = copy(target.list)
            elif type(target) is base_type:
                self.list = copy(target)
            elif type(target) in (tuple, set, frozenset):
                self.list = base_type([*target])
            elif type(target) in (dict, defaultdict, OrderedDict):
                self.list = base_type([*sorted(target).values()])
            elif hasattr(target, 'iterate'):
                self.list = base_type([*target.iterate()])
            elif hasattr(target, '__iter__'):
                self.list = base_type([*target])
            else:
                self.list = base_type([target])
        else:
            self.list = base_type([*args])
    
    def iterate(self):
        yield from self.list
    
    @wraps(list.__len__)
    def __len__(self):
        return len(self.list)
    
    @wraps(list.__contains__)
    def __contains__(self, value):
        return value in self.list
    
    @wraps(list.__getitem__)
    def __getitem__(self, idx):
        return self.list[idx]
    
    @wraps(list.__setitem__)
    def __setitem__(self, idx, value):
        if value in (None, NOOp):
            value = NOOp()
        self.list[idx] = value
    
    @wraps(list.__delitem__)
    def __delitem__(self, idx):
        del self.list[idx]
    
    @wraps(list.index)
    def index(self, value):
        return self.list.index(value)
    
    @wraps(list.append)
    def append(self, value):
        self.list.append(value)
    
    @wraps(list.extend)
    def extend(self, iterable):
        self.list.extend(iterable)
    
    def pop(self, idx=-1):
        """ Remove and return item at `idx` (default last).
            Raises IndexError if list is empty or `idx` is out of range.
            See list.pop(…) for details.
        """
        self.list.pop(idx)
    
    def last(self):
        if not bool(self):
            raise IndexError("pipe is empty")
        return self.list[-1]
    
    def process(self, image):
        for processor in self.iterate():
            image = processor.process(image)
        return image
    
    def __eq__(self, other):
        if not isinstance(other, (type(self), type(self).base_type())):
            return NotImplemented
        return super(Pipe, self).__eq__(other)

@export
class BandFork(Fork):
    
    """ BandFork is a processor container -- a processor that applies other
        processors. BandFork acts selectively on the individual bands of
        input image data, either:
        - applying a band-specific processor instance, or
        - applying a default processor factory successively across all bands.
        
        BandFork’s interface is closely aligned with Python’s mutable-mapping
        API‡ -- with which most programmers are no doubt quite familiar:
        
        • Ex. 1: apply Atkinson dithering to each of an RGB images’ bands:
        >>> from instakit.utils.pipeline import BandFork
        >>> from instakit.processors.halftone import Atkinson
        >>> BandFork(Atkinson).process(my_image)
        
        • Ex. 2: apply Atkinson dithering to only the green band:
        >>> from instakit.utils.pipeline import BandFork
        >>> from instakit.processors.halftone import Atkinson
        >>> bfork = BandFork(None)
        >>> bfork['G'] = Atkinson()
        >>> bfork.process(my_image)
        
        BandFork inherits from `instakit.abc.Fork`, which itself is not just
        an Instakit Processor. The Fork ABC implements the required methods
        of an Instakit Processor Container†, through which it furnishes an
        interface to individual bands -- also generally known as channels,
        per the language of the relevant Photoshop UI elements -- of image
        data. 
        
        † q.v. the `instakit.abc` module source code supra.
        ‡ q.v. the `collections.abc` module, and the `MutableMapping`
                    abstract base class within, supra.
    """
    __slots__ = tuplize('mode_t')
    
    def __init__(self, processor_factory, *args, **kwargs):
        """ Initialize a BandFork instance, using the given callable value
            for `processor_factory` and any band-appropriate keyword-arguments,
            e.g. `(R=MyProcessor, G=MyOtherProcessor, B=None)`
        """
        # Call `super(…)`, passing `processor_factory`:
        super(BandFork, self).__init__(processor_factory, *args, **kwargs)
        
        # Reset `self.mode_t` if a new mode was specified --
        # N.B. we can’t use the “self.mode” property during “__init__(…)”:
        self.mode_t = kwargs.pop('mode', Mode.RGB)
    
    @property
    def mode(self):
        return self.mode_t
    
    @mode.setter
    def mode(self, value):
        if value is None:
            return
        if type(value) in string_types:
            value = Mode.for_string(value)
        if Mode.is_mode(value):
            # if value is not self.mode_t:
            self.set_mode_t(value)
        else:
            raise TypeError("invalid mode type: %s (%s)" % (type(value), value))
    
    def set_mode_t(self, value):
        self.mode_t = value # DOUBLE SHADOW!!
    
    @property
    def band_labels(self):
        return self.mode.bands
    
    def iterate(self):
        yield from (self[band_label] for band_label in self.band_labels)
    
    def split(self, image):
        return self.mode.process(image).split()
    
    def compose(self, *bands):
        return self.mode.merge(*bands)
    
    def process(self, image):
        processed = []
        for processor, band in zip(self.iterate(),
                                   self.split(image)):
            processed.append(processor.process(band))
        return self.compose(*processed)

ChannelFork = BandFork

ink_values = (
    (255, 255, 255),    # White
    (0,   250, 250),    # Cyan
    (250, 0,   250),    # Magenta
    (250, 250, 0),      # Yellow
    (0,   0,   0),      # Key (blacK)
    (255, 0,   0),      # Red
    (0,   255, 0),      # Green
    (0,   0,   255),    # Blue
)

class Ink(AliasingEnum):
    
    def rgb(self):
        return ink_values[self.value]
    
    def process(self, image):
        InkType = type(self)
        return ImageOps.colorize(Mode.L.process(image),
                                 InkType(0).rgb(),
                                 InkType(self.value).rgb())

@unique
class CMYKInk(Ink):
    
    WHITE       = 0
    CYAN        = 1
    MAGENTA     = 2
    YELLOW      = 3
    KEY         = 4
    BLACK       = alias(KEY)
    
    @classmethod
    def CMYK(cls):
        return (cls.CYAN, cls.MAGENTA, cls.YELLOW, cls.BLACK)
    
    @classmethod
    def CMY(cls):
        return (cls.CYAN, cls.MAGENTA, cls.YELLOW)

@unique
class RGBInk(Ink):
    
    WHITE       = 0
    RED         = 5
    GREEN       = 6
    BLUE        = 7
    KEY         = 4
    BLACK       = alias(KEY)
    
    @classmethod
    def RGB(cls):
        return (cls.RED, cls.GREEN, cls.BLUE)
    
    @classmethod
    def BGR(cls):
        return (cls.BLUE, cls.GREEN, cls.RED)

@export
class OverprintFork(BandFork):
    
    """ A BandFork subclass that rebuilds its output image using multiply-mode
        to simulate CMYK overprinting effects.
        
        N.B. While this Fork-based processor operates strictly in CMYK mode,
        the composite image it eventually returns will be in RGB mode. This is
        because the CMYK channels are each individually converted to colorized
        representations in order to simulate monotone ink preparations; the
        final compositing operation, in which these colorized channel separation
        images are combined with multiply-mode, is also computed using the RGB
        color model -- q.v. the CMYKInk enum processor supra. and the related
        PIL/Pillow module function `ImageOps.colorize(…)` supra.
    """
    __slots__ = ('contrast', 'basicgcr')
    
    inks = CMYKInk.CMYK()
    
    def __init__(self, processor_factory, gcr=20, *args, **kwargs):
        """ Initialize an OverprintFork instance with the given callable value
            for `processor_factory` and any band-appropriate keyword-arguments,
            e.g. `(C=MyProcessor, M=MyOtherProcessor, Y=MyProcessor, K=None)`
        """
        # Store BasicGCR and AutoContrast processors:
        self.contrast = AutoContrast()
        self.basicgcr = BasicGCR(percentage=gcr)
        
        # Call `super(…)`, passing `processor_factory`:
        super(OverprintFork, self).__init__(processor_factory, *args, mode=Mode.CMYK,
                                                              **kwargs)
        
        # Make each band-processor a Pipeline() ending in
        # the channel-appropriate CMYKInk enum processor:
        self.apply_CMYK_inks()
    
    def apply_CMYK_inks(self):
        """ This method ensures that each bands’ processor is set up
            as a Pipe() or Pipeline() ending in a CMYKInk corresponding
            to the band in question. Calling it multiple times *should*
            be idempotent (but don’t quote me on that)
        """
        for band_label, ink in zip(self.band_labels,
                              type(self).inks):
            processor = self[band_label]
            if processor is None:
                self[band_label] = Pipe(ink)
            elif hasattr(processor, 'append'):
                if processor[-1] is not ink:
                    processor.append(ink)
                    self[band_label] = processor
            elif hasattr(processor, 'last'):
                if processor.last() is not ink:
                    self[band_label] = Pipe(*processor.iterate(), ink)
            else:
                self[band_label] = Pipe(processor, ink)
    
    def set_mode_t(self, value):
        """ Raise an exception if an attempt is made to set the mode to anything
            other than CMYK
        """
        if value is not Mode.CMYK:
            raise AttributeError(
                "OverprintFork only works in %s mode" % Mode.CMYK.to_string())
    
    def update(self, iterable=None, **kwargs):
        """ OverprintFork.update(…) re-applies CMYK ink processors to the
            updated processing dataflow
        """
        super(OverprintFork, self).update(iterable, **kwargs)
        self.apply_CMYK_inks()
    
    def split(self, image):
        """ OverprintFork.split(image) uses imagekit.utils.gcr.BasicGCR(…) to perform
            gray-component replacement in CMYK-mode images; for more information,
            see the imagekit.utils.gcr module
        """
        return self.basicgcr.process(image).split()
    
    def compose(self, *bands):
        """ OverprintFork.compose(…) uses PIL.ImageChops.multiply() to create
            the final composite image output
        """
        return Σ(ImageChops.multiply, bands)

class Grid(Fork):
    pass

class Sequence(Fork):
    pass

ChannelOverprinter = OverprintFork

export(ChannelFork,         name='ChannelFork')
export(ChannelOverprinter,  name='ChannelOverprinter')
export(CMYKInk,             name='CMYKInk',         doc="CMYKInk → Enumeration class furnishing CMYK primitive triple values")
export(RGBInk,              name='RGBInk',          doc="RGBInk → Enumeration class furnishing RGB primitive triple values")

# Assign the modules’ `__all__` and `__dir__` using the exporter:
__all__, __dir__ = exporter.all_and_dir()

def test():
    from pprint import pprint
    from instakit.utils.static import asset
    from instakit.processors.halftone import Atkinson
    
    image_paths = list(map(
        lambda image_file: asset.path('img', image_file),
            asset.listfiles('img')))
    image_inputs = list(map(
        lambda image_path: Mode.RGB.open(image_path),
            image_paths))
    
    for image_input in image_inputs[:2]:
        OverprintFork(Atkinson).process(image_input).show()
        
        print('Creating OverprintFork and BandFork with Atkinson ditherer...')
        overatkins = OverprintFork(Atkinson)
        forkatkins = BandFork(Atkinson)
        
        print('Processing image with BandForked Atkinson in default (RGB) mode...')
        forkatkins.process(image_input).show()
        forkatkins.mode = 'CMYK'
        print('Processing image with BandForked Atkinson in CMYK mode...')
        forkatkins.process(image_input).show()
        forkatkins.mode = 'RGB'
        print('Processing image with BandForked Atkinson in RGB mode...')
        forkatkins.process(image_input).show()
        
        overatkins.mode = 'CMYK'
        print('Processing image with OverprintFork-ized Atkinson in CMYK mode...')
        overatkins.process(image_input).show()
        
        print('Attempting to reset OverprintFork to RGB mode...')
        import traceback, sys
        try:
            overatkins.mode = 'RGB'
            overatkins.process(image_input).show()
        except:
            print(">>>>>>>>>>>>>>>>>>>>> TRACEBACK <<<<<<<<<<<<<<<<<<<<<")
            traceback.print_exc(file=sys.stdout)
            print("<<<<<<<<<<<<<<<<<<<<< KCABECART >>>>>>>>>>>>>>>>>>>>>")
            print('')
    
    bandfork = BandFork(None)
    pprint(bandfork)
    
    print(image_paths)

if __name__ == '__main__':
    test()
