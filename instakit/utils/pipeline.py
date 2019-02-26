# encoding: utf-8
from __future__ import print_function

from PIL import ImageOps, ImageChops
from enum import unique
from functools import wraps

try:
    from functools import reduce
except ImportError:
    pass

from instakit.utils.gcr import gcrcore
from instakit.utils.mode import Mode
from instakit.utils.misc import string_types
from instakit.abc import Enum, Container, NOOp, Fork

class Pipeline(Container):
    
    """ A linear pipeline of processors to be applied en masse.
        Derived from an ImageKit class:
        imagekit.processors.base.ProcessorPipeline
    """
    
    @classmethod
    def base_type(cls):
        return list
    
    @wraps(list.__init__)
    def __init__(self, *args):
        self.list = list(*args)
    
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
    
    def last(self):
        return self.list[-1]
    
    def process(self, image):
        for p in self.iterate():
            image = p.process(image)
        return image

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
        """ Initialize a BandFork instance, using the given callable value
            for `default_factory` and any band-appropriate keyword-arguments,
            e.g. `(R=MyProcessor, G=MyOtherProcessor, B=None)`
        """
        # Reset mode if a new mode was specified:
        if 'mode' in kwargs:
            self.mode = kwargs.pop('mode')
        
        # Call super(…):
        super(BandFork, self).__init__(default_factory, *args, **kwargs)
    
    @property
    def mode(self):
        return self.mode_t
    
    @mode.setter
    def mode(self, value):
        if type(value) in string_types:
            value = Mode.for_string(value)
        if type(value) is Mode:
            if value is not self.mode_t:
                self.set_mode_t(value)
        else:
            raise TypeError("invalid mode type: %s (%s)" % (type(value), value))
    
    def set_mode_t(self, value):
        self.mode_t = value # SHADOW!!
    
    @property
    def band_labels(self):
        return self.mode_t.bands
    
    def iterate(self):
        for band_label in self.band_labels:
            yield self[band_label]
    
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

Pipe = Pipeline
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

class OverprintFork(BandFork):
    
    """ A ChannelFork subclass that rebuilds its output image using
        multiply-mode to simulate CMYK overprinting effects.
    """
    
    mode_t = Mode.CMYK
    
    def __init__(self, default_factory, gcr=20, *args, **kwargs):
        """ Initialize an OverprintFork instance with the given callable value
            for `default_factory` and any band-appropriate keyword-arguments,
            e.g. `(C=MyProcessor, M=MyOtherProcessor, Y=MyProcessor, K=None)`
        """
        # Store GCR percentage:
        self.gcr = gcr
        
        # Call super():
        super(OverprintFork, self).__init__(default_factory, *args, **kwargs)
        
        # Make each band-processor a Pipeline() ending in
        # the channel-appropriate CMYKInk enum processor:
        self.apply_CMYK_inks()
    
    def apply_CMYK_inks(self):
        """ This method ensures that each bands’ processor is set up
            as a Pipeline() ending in a CMYKInk corresponding to the
            band in question. Calling it multiple times *should* be
            idempotent (but don’t quote me on that)
        """
        modestring = type(self).mode_t.to_string()
        CMYKLabels = CMYKInk.CMYK()
        for band in self.band_labels:
            processor = self[band]
            idx = modestring.index(band)
            ink = CMYKInk(CMYKLabels[idx])
            if hasattr(processor, 'append'):
                if processor[-1] is not ink:
                    processor.append(ink)
                    self[band] = processor
            else:
                self[band] = Pipeline([processor, ink])
    
    def set_mode_t(self, value):
        """ Raise an exception if an attempt is made to set the mode to anything
            other than CMYK
        """
        if value is not type(self).mode_t:
            raise AttributeError(
                "OverprintFork only works in %s mode" % type(self).mode_t.to_string())
    
    def split(self, image):
        """ OverprintFork.split(image) uses imagekit.utils.gcr.gcrcore(…) to perform
            gray-component replacement in CMYK-mode images; for more information,
            see the imagekit.utils.gcr module
        """
        bands = super(OverprintFork, self).split(image)
        gcrcore(image, self.gcr, bands)
        return bands
    
    def compose(self, *bands):
        """ OverprintFork.compose(…) uses PIL.ImageChops.multiply() to create
            the final composite image output
        """
        return reduce(ImageChops.multiply, bands)

class Grid(Fork):
    pass

class Sequence(Fork):
    pass

ChannelOverprinter = OverprintFork

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
    
