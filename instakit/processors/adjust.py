# encoding: utf-8
from __future__ import print_function

from PIL import ImageOps, ImageChops, ImageEnhance as enhancers
from clu.abstract import Slotted
from clu.predicates import tuplize
from instakit.abc import abstract, ABC, Processor
from instakit.exporting import Exporter

exporter = Exporter(path=__file__)
export = exporter.decorator()

@export
class EnhanceNop(ABC, metaclass=Slotted):
    
    __slots__ = tuplize('image')
    
    def __init__(self, image=None):
        self.image = image
    
    def adjust(self, *args, **kwargs):
        return self.image

@export
class Adjustment(Processor):
    
    """ Base type for image adjustment processors """
    __slots__ = tuplize('value')
    
    def __init__(self, value=1.0):
        """ Initialize the adjustment with a float value """
        self.value = value
    
    @abstract
    def adjust(self, image):
        """ Adjust the image, using the float value with which
            the adjustment was first initialized
        """
        ...
    
    def process(self, image):
        return (self.value == 1.0) and image or self.adjust(image)

@export
class Color(Adjustment):
    
    """ Globally tweak the image color """
    
    def adjust(self, image):
        return enhancers.Color(image).enhance(self.value)

@export
class Brightness(Adjustment):
    
    """ Adjust the image brightness """
    
    def adjust(self, image):
        return enhancers.Brightness(image).enhance(self.value)

@export
class Contrast(Adjustment):
    
    """ Adjust the image contrast """
    
    def adjust(self, image):
        return enhancers.Contrast(image).enhance(self.value)

@export
class Sharpness(Adjustment):
    
    """ Adjust the sharpness of the image """
    
    def adjust(self, image):
        return enhancers.Sharpness(image).enhance(self.value)

@export
class BrightnessContrast(Adjustment):
    
    """ Adjust the image brightness and contrast simultaneously """
    
    def adjust(self, image):
        for Enhancement in (enhancers.Brightness, enhancers.Contrast):
            image = Enhancement(image).enhance(self.value)
        return image

@export
class BrightnessSharpness(Adjustment):
    
    """ Adjust the image brightness and sharpness simultaneously """
    
    def adjust(self, image):
        for Enhancement in (enhancers.Brightness, enhancers.Sharpness):
            image = Enhancement(image).enhance(self.value)
        return image

@export
class ContrastSharpness(Adjustment):
    
    """ Adjust the image contrast and sharpness simultaneously """
    
    def adjust(self, image):
        for Enhancement in (enhancers.Contrast, enhancers.Sharpness):
            image = Enhancement(image).enhance(self.value)
        return image

@export
class Invert(Processor):
    
    """ Perform a simple inversion of the image values """
    
    def process(self, image):
        return ImageChops.invert(image)

@export
class Equalize(Processor):
    
    """ Apply a non-linear mapping to the image, via histogram """
    __slots__ = tuplize('mask')
    
    def __init__(self, mask=None):
        self.mask = hasattr(mask, 'copy') and mask.copy() or mask
    
    def process(self, image):
        return ImageOps.equalize(image, mask=self.mask)

@export
class AutoContrast(Processor):
    
    """ Normalize contrast throughout the image, via histogram """
    __slots__ = tuplize('cutoff', 'ignore')
    
    def __init__(self, cutoff=0, ignore=None):
        self.cutoff, self.ignore = cutoff, ignore
    
    def process(self, image):
        return ImageOps.autocontrast(image, cutoff=self.cutoff,
                                            ignore=self.ignore)

@export
class Solarize(Processor):
    
    """ Invert all pixel values above an 8-bit threshold """
    __slots__ = tuplize('threshold')
    
    def __init__(self, threshold=128):
        self.threshold = min(max(1, threshold), 255)
    
    def process(self, image):
        return ImageOps.solarize(image, threshold=self.threshold)

@export
class Posterize(Processor):
    
    """ Reduce the number of bits (1 to 8) per channel """
    __slots__ = tuplize('bits')
    
    def __init__(self, bits=4):
        self.bits = min(max(1, bits), 8)
    
    def process(self, image):
        return ImageOps.posterize(image, bits=self.bits)

# Assign the modules’ `__all__` and `__dir__` using the exporter:
__all__, __dir__ = exporter.all_and_dir()

def test():
    # from clu.predicates import haspyattr
    from clu.predicates import isslotted
    
    G = globals()
    
    for typename in __all__:
        if typename != "Adjustment":
            assert G[typename]
            assert isslotted(G[typename]())
            # assert not isdictish(G[typename]())
            # assert not haspyattr(G[typename](), 'dict')

if __name__ == '__main__':
    test()