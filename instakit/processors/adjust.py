
from PIL import ImageOps, ImageChops
from PIL.ImageEnhance import Color as _Color
from PIL.ImageEnhance import Brightness as _Brightness
from PIL.ImageEnhance import Contrast as _Contrast
from PIL.ImageEnhance import Sharpness as _Sharpness

from instakit.abc import Processor

class Adjustment(Processor):
    
    __slots__ = ('value',)
    
    def __init__(self, value=1.0):
        self.value = value
    
    def adjust(self, image):
        return image
    
    def process(self, image):
        if self.value != 1.0:
            return self.adjust(image)
        return image

class Color(Adjustment):
    """ Globally tweak the image color """
    __slots__ = tuple()
    def adjust(self, image):
        return _Color(image).enhance(self.value)

class Brightness(Adjustment):
    """ Adjust the image brightness """
    __slots__ = tuple()
    def adjust(self, image):
        return _Brightness(image).enhance(self.value)

class Contrast(Adjustment):
    """ Adjust the image contrast """
    __slots__ = tuple()
    def adjust(self, image):
        return _Contrast(image).enhance(self.value)

class Sharpness(Adjustment):
    """ Adjust the sharpness of the image """
    __slots__ = tuple()
    def adjust(self, image):
        return _Sharpness(image).enhance(self.value)

class Invert(Processor):
    __slots__ = tuple()
    def process(self, image):
        return ImageChops.invert(image)

class Equalize(Processor):
    """ Apply a non-linear mapping to the image, via histogram """
    __slots__ = ('mask',)
    
    def __init__(self, mask=None):
        self.mask = hasattr(mask, 'copy') and mask.copy() or mask
    
    def process(self, image):
        return ImageOps.equalize(image, mask=self.mask)

class AutoContrast(Processor):
    """ Normalize contrast throughout the image, via histogram """
    __slots__ = ('cutoff', 'ignore')
    
    def __init__(self, cutoff=0, ignore=None):
        self.cutoff = cutoff
        self.ignore = ignore
    
    def process(self, image):
        return ImageOps.autocontrast(image, cutoff=self.cutoff,
                                            ignore=self.ignore)

class Solarize(Processor):
    """ Invert all pixel values above an 8-bit threshold """
    __slots__ = ('threshold',)
    
    def __init__(self, threshold=128):
        self.threshold = min(max(1, threshold), 255)
    
    def process(self, image):
        return ImageOps.solarize(image, threshold=self.threshold)

class Posterize(Processor):
    """ Reduce the number of bits (1 to 8) per channel """
    __slots__ = ('bits',)
    
    def __init__(self, bits=4):
        self.bits = min(max(1, bits), 8)
    
    def process(self, image):
        return ImageOps.posterize(image, bits=self.bits)
    