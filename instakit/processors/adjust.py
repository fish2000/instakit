
from PIL import ImageOps
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
    def adjust(self, img):
        return _Color(img).enhance(self.value)

class Brightness(Adjustment):
    def adjust(self, img):
        return _Brightness(img).enhance(self.value)

class Contrast(Adjustment):
    def adjust(self, img):
        return _Contrast(img).enhance(self.value)

class Sharpness(Adjustment):
    def adjust(self, img):
        return _Sharpness(img).enhance(self.value)


class Equalize(object):
    def __init__(self, mask=None):
        self.mask = hasattr(mask, 'copy') and \
            mask.copy() or mask
    def process(self, img):
        return ImageOps.equalize(img,
            mask=self.mask)


class AutoContrast(object):
    def __init__(self, cutoff=0, ignore=None):
        self.cutoff = cutoff
        self.ignore = ignore
    def process(self, img):
        return ImageOps.autocontrast(img,
            cutoff=self.cutoff,
            ignore=self.ignore)
