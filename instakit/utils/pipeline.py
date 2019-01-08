
from __future__ import print_function

from collections import defaultdict
from PIL import ImageOps, ImageChops
from enum import Enum, unique

try:
    from functools import reduce
except ImportError:
    pass

from instakit.utils.mode import Mode

class Pipe(list):
    """ A linear pipeline of processors to be applied en masse.
        Derived from an ImageKit class:
        imagekit.processors.base.ProcessorPipeline
    """
    def process(self, image):
        for p in self:
            image = p.process(image)
        return image

class NOOp(object):
    """ A no-op processor. """
    def process(self, image):
        return image

class ChannelFork(defaultdict):
    """ A processor wrapper that, for each image channel:
        - applies a channel-specific processor, or
        - applies a default processor.
        
        * Ex. 1: apply the Atkinson ditherer to each of an images' channels:
        >>> from instakit.utils.pipeline import ChannelFork
        >>> from instakit.processors.halftone import Atkinson
        >>> ChannelFork(Atkinson).process(my_image)
        
        * Ex. 2: apply the Atkinson ditherer to only one channel:
        >>> from instakit.utils.pipeline import ChannelFork
        >>> from instakit.processors.halftone import Atkinson
        >>> cfork = ChannelFork(None)
        >>> cfork['G'] = Atkinson()
        >>> cfork.process(my_image)
    """
    
    default_mode = 'RGB'
    
    def __init__(self, default_factory, *args, **kwargs):
        if default_factory is None:
            default_factory = NOOp
        if not callable(default_factory):
            raise AttributeError(
                "ChannelFork() requires a callable default_factory")
        
        self.channels = Mode.for_string(
                        kwargs.pop('mode', self.default_mode))
        
        super(ChannelFork, self).__init__(default_factory, *args, **kwargs)
    
    def __setitem__(self, idx, value):
        if value in (None, NOOp):
            value = NOOp()
        super(ChannelFork, self).__setitem__(idx, value)
    
    @property
    def mode(self):
        return self.channels.to_string()
    
    @mode.setter
    def mode(self, mode_string):
        self._set_mode(mode_string)
    
    def _set_mode(self, mode_string):
        self.channels = Mode.for_string(mode_string)
    
    def compose(self, *channels):
        return self.channels.merge(*channels)
    
    def process(self, image):
        if not self.channels.check(image):
            image = self.channels.process(image)
        
        processed_channels = []
        for idx, channel in enumerate(image.split()):
            processed_channels.append(
                self[self.channels.bands[idx]].process(channel))
        
        return self.compose(*processed_channels)

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

class ChannelOverprinter(ChannelFork):
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
            clone[channel_name] = Pipe([self[channel_name], ink])
        
        # Delegate processing to the “clone” instance:
        return clone.process(image)


if __name__ == '__main__':
    from instakit.utils.static import asset
    from instakit.processors.halftone import Atkinson
    
    image_paths = list(map(
        lambda image_file: asset.path('img', image_file),
            asset.listfiles('img')))
    image_inputs = list(map(
        lambda image_path: Mode.RGB.open(image_path),
            image_paths))
    
    for image_input in image_inputs[:2]:
        #ChannelOverprinter(Atkinson).process(image_input).show()
        
        print('Creating ChannelOverprinter and ChannelFork with Atkinson ditherer...')
        overatkins = ChannelOverprinter(Atkinson)
        forkatkins = ChannelFork(Atkinson)
        
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
    
    print(image_paths)
    
