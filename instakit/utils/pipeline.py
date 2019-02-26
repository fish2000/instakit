
from __future__ import print_function

from PIL import ImageOps, ImageChops
from abc import ABC, abstractmethod as abstract
from collections import defaultdict
from enum import Enum as EnumBase, unique
from functools import wraps
# from six import add_metaclass

try:
    from functools import reduce
except ImportError:
    pass

from instakit.utils.mode import Mode
from instakit.utils.misc import string_types

class Processor(ABC):
    
    """ Base abstract processor class. """
    
    @abstract
    def process(self, image): ...
    
    def __call__(self, image):
        return self.process(image)

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

class Pipeline(Container):
    """ A linear pipeline of processors to be applied en masse.
        Derived from an ImageKit class:
        imagekit.processors.base.ProcessorPipeline
    """
    @wraps(list.__init__)
    def __init__(self, *args):
        self.list = list(*args)
    
    @wraps(Container.iterate)
    def iterate(self):
        return iter(self.list)
    
    @wraps(list.__len__)
    def __len__(self):
        return len(self.list)
    
    @wraps(list.__contains__)
    def __contains__(self, value):
        return value in self.list
    
    @wraps(list.__getitem__)
    def __getitem__(self, idx):
        return self.list(idx)
    
    @wraps(list.__setitem__)
    def __setitem__(self, idx, value):
        if value in (None, NOOp):
            value = NOOp()
        self.list[idx] = value
    
    @wraps(list.index)
    def index(self, value):
        return self.list.index(value)
    
    @wraps(list.append)
    def append(self, value):
        self.list.append(value)
    
    @wraps(list.extend)
    def extend(self, iterable):
        self.list.extend(iterable)
    
    def process(self, image):
        for p in self.iterate():
            image = p.process(image)
        return image

class Fork(Container):
    
    """ Base abstract forking processor. """
    
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
    
    def __len__(self):
        return len(self.dict)
    
    def __contains__(self, value):
        return value in self.dict
    
    def __getitem__(self, idx):
        return self.dict[idx]
    
    def __setitem__(self, idx, value):
        if value in (None, NOOp):
            value = NOOp()
        self.dict[idx] = value
    
    def get(self, idx, default_value=None):
        return self.dict.get(idx, default_value)
    
    @abstract
    def split(self, image): ...
    
    @abstract
    def compose(self, *bands): ...


class BandFork(Fork):
    """ A processor wrapper that, for each image channel:
        - applies a band-specific processor, or
        - applies a default processor.
        
        * Ex. 1: apply the Atkinson ditherer to each of an images' bands:
        >>> from instakit.utils.pipeline import BandFork
        >>> from instakit.processors.halftone import Atkinson
        >>> BandFork(Atkinson).process(my_image)
        
        * Ex. 2: apply the Atkinson ditherer to only one band:
        >>> from instakit.utils.pipeline import BandFork
        >>> from instakit.processors.halftone import Atkinson
        >>> bfork = BandFork(None)
        >>> bfork['G'] = Atkinson()
        >>> bfork.process(my_image)
    """
    
    mode_t = Mode.RGB
    
    def __init__(self, default_factory, *args, **kwargs):
        if 'mode' in kwargs:
            new_mode = kwargs.pop('mode')
            if type(new_mode) in string_types:
                new_mode = Mode.for_string(new_mode)
            if type(new_mode) is Mode:
                self.mode_t = new_mode # SHADOW!!
        
        super(BandFork, self).__init__(default_factory, *args, **kwargs)
    
    @property
    def mode(self):
        return self.mode_t
    
    @mode.setter
    def mode(self, value):
        if type(value) in string_types:
            value = Mode.for_string(value)
        if type(value) is Mode:
            self.set_mode_t(value)
        else:
            raise TypeError("invalid mode type: %s (%s)" % (type(value), value))
    
    def set_mode_t(self, value):
        self.mode_t = value
    
    @property
    def band_labels(self):
        return self.mode_t.bands
    
    @wraps(Container.iterate)
    def iterate(self):
        for band in self.band_labels:
            yield self[band]
    
    def split(self, image):
        return self.mode_t.process(image).split()
    
    def compose(self, *bands):
        return self.mode_t.merge(*bands)
    
    def process(self, image):
        processed = []
        for processor, band in zip(self.iterate(),
                                   self.split(image)):
            processed.append(processor.process(band))
        return self.compose(*processed)


class OverprintFork(BandFork):
    pass

class Grid(Fork):
    pass

class Sequence(Fork):
    pass

Pipe = Pipeline

class NOOp(Processor):
    """ A no-op processor. """
    def process(self, image):
        return image

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

class Ink(Enum):
    
    def rgb(self):
        return ink_values[self.value]
    
    def process(self, image):
        InkType = type(self)
        return ImageOps.colorize(Mode.L.process(image),
                                 InkType(0).rgb(),
                                 InkType(self.value).rgb())

@unique
class CMYKInk(Ink):
    
    WHITE = 0
    CYAN = 1
    MAGENTA = 2
    YELLOW = 3
    KEY = 4
    
    @classmethod
    def CMYK(cls):
        return (cls.CYAN, cls.MAGENTA, cls.YELLOW, cls.KEY)
    
    @classmethod
    def CMY(cls):
        return (cls.CYAN, cls.MAGENTA, cls.YELLOW)

@unique
class RGBInk(Ink):
    
    WHITE = 0
    RED = 5
    GREEN = 6
    BLUE = 7
    KEY = 4
    
    @classmethod
    def RGB(cls):
        return (cls.RED, cls.GREEN, cls.BLUE)
    
    @classmethod
    def BGR(cls):
        return (cls.BLUE, cls.GREEN, cls.RED)

class ChannelOverprinter(ChannelFork, Processor):
    """ A ChannelFork subclass that rebuilds its output image using
        multiply-mode to simulate CMYK overprinting effects.
    """
    default_mode = 'CMYK'
    
    def _set_mode(self, mode_string):
        if mode_string != self.default_mode: # CMYK
            raise AttributeError(
                "ChannelOverprinter can operate in %s mode only" %
                    self.default_mode) # CMYK
    
    def compose(self, *channels):
        return reduce(ImageChops.multiply, channels)
    
    def process(self, image):
        # Compile the standard CMYK ink values as a dict of processing ops,
        # keyed with the letter of their channel name (e.g. C, M, Y, and K):
        inks = zip(self.default_mode, # CMYK
                  [CMYKInk(ink_label) for ink_label in CMYKInk.CMYK()])
        
        # Manually two-phase allocate/initialize a ChannelFork “clone” instance
        # to run the ChannelOverprinter CMYK composition processing operations:
        # clone = super(ChannelOverprinter, self).__new__(self.default_factory,
        #                                                 mode=self.channels.mode)
        # clone.__init__(self.default_factory,
        #                mode=self.channels.mode)
        clone = super(ChannelOverprinter, self).__new__(type(self).__mro__[1],
                                                             self.default_factory)
        clone.__init__(self.default_factory)
        
        # Create a pipeline for each of the overprinters’ CMYK channel operations,
        # and install the pipeline in the newly created “clone” ChannelFork:
        for channel_name, ink in inks:
            # For each channel, first run the prescribed operations;
            # and afterward, colorize the output (as per a duotone image) using
            # the CMYK ink as the colorization value:
            clone[channel_name] = Pipeline([self[channel_name], ink])
        
        # Delegate processing to the “clone” instance:
        return clone.process(image)


if __name__ == '__main__':
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
        ChannelOverprinter(Atkinson).process(image_input).show()
        
        print('Creating ChannelOverprinter and ChannelFork with Atkinson ditherer...')
        overatkins = ChannelOverprinter(Atkinson)
        forkatkins = BandFork(Atkinson)
        
        print('Processing image with ChannelForked Atkinson in default (RGB) mode...')
        forkatkins.process(image_input).show()
        forkatkins.mode = 'CMYK'
        print('Processing image with ChannelForked Atkinson in CMYK mode...')
        forkatkins.process(image_input).show()
        forkatkins.mode = 'RGB'
        print('Processing image with ChannelForked Atkinson in RGB mode...')
        forkatkins.process(image_input).show()
        
        overatkins.mode = 'CMYK'
        print('Processing image with ChannelOverprinter-ized Atkinson in CMYK mode...')
        overatkins.process(image_input).show()
        
        print('Attempting to reset ChannelOverprinter to RGB mode...')
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
    
