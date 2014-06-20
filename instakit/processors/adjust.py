
from PIL import ImageOps, ImageChops
from PIL.ImageEnhance import Color as _Color
from PIL.ImageEnhance import Brightness as _Brightness
from PIL.ImageEnhance import Contrast as _Contrast
from PIL.ImageEnhance import Sharpness as _Sharpness


class Adjustment(object):
    def __init__(self, value=1.0):
        self.value = value
    def adjust(self, img):
        return img
    def process(self, img):
        if self.value != 1.0:
            return self.adjust(img)
        return img

class Color(Adjustment):
    """ Globally tweak the image color """
    def adjust(self, img):
        return _Color(img).enhance(self.value)

class Brightness(Adjustment):
    """ Adjust the image brightness """
    def adjust(self, img):
        return _Brightness(img).enhance(self.value)

class Contrast(Adjustment):
    """ Adjust the image contrast """
    def adjust(self, img):
        return _Contrast(img).enhance(self.value)

class Sharpness(Adjustment):
    """ Adjust the sharpness of the image """
    def adjust(self, img):
        return _Sharpness(img).enhance(self.value)

class Invert(object):
    def process(self, img):
        return ImageChops.invert(img)

class Equalize(object):
    """ Apply a non-linear mapping to the image, via histogram """
    def __init__(self, mask=None):
        self.mask = hasattr(mask, 'copy') and \
            mask.copy() or mask
    def process(self, img):
        return ImageOps.equalize(img,
            mask=self.mask)

class AutoContrast(object):
    """ Normalize contrast throughout the image, via histogram """
    def __init__(self, cutoff=0, ignore=None):
        self.cutoff = cutoff
        self.ignore = ignore
    def process(self, img):
        return ImageOps.autocontrast(img,
            cutoff=self.cutoff,
            ignore=self.ignore)

class Solarize(object):
    """ Invert all pixel values above an 8-bit threshold """
    def __init__(self, threshold=128):
        self.threshold = min(max(1, threshold), 255)
    def process(self, img):
        return ImageOps.solarize(img,
            threshold=self.threshold)

class Posterize(object):
    """ Reduce the number of bits (1 to 8) per channel """
    def __init__(self, bits=4):
        self.bits = min(max(1, bits), 8)
    def process(self, img):
        return ImageOps.posterize(img,
            bits=self.bits)
    