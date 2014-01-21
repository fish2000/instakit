#!/usr/bin/env python
# encoding: utf-8
"""
lutmap.py

Created by FI$H 2000 on 2012-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""

import numpy, imread
from os.path import join
from collections import defaultdict

from PIL import Image
#from math import floor

from instakit.utils.colortype import ColorType
from instakit.utils import static

class RGBTable(defaultdict):
    RGB = ColorType('RGB', dtype=numpy.uint8)
    identity = numpy.zeros(
        shape=(512, 512, 3),
        dtype=RGB.dtype)
    
    for bx in xrange(0, 8):
        for by in xrange(0, 8):
            for r in xrange(0, 64):
                for g in xrange(0, 64):
                    identity[
                        int(g + by * 64),
                        int(r + bx * 64)] = RGB(
                            int(r * 255.0 / 63.0 + 0.5),
                            int(g * 255.0 / 63.0 + 0.5),
                            int((bx + by * 8.0) * 255.0 / 63.0 + 0.5))
    
    def __init__(self):
        super(RGBTable, self).__init__(default_factory=None)
        self.data = self.identity
    
    def __getitem__(self, color):
        return defaultdict.__getitem__(self, self._idx(color))
    
    def __missing__(self, idx):
        self[idx] = value = self.lookup(self._rgb(idx))
        return value
    
    def _idx(self, color):
        return int('%02x%02x%02x' % color, 16)
    
    def _rgb(self, idx):
        RGB = self.RGB
        return RGB(*reversed(
            [(idx >> (8*i)) & 255 for i in range(3)]))
    
    def lookup(self, color):
        return self.color_at(*self._xy(color))
    
    def _xy(self, color):
        try:
            return zip(*numpy.where(
                numpy.all(
                    RGBTable.identity == color,
                    axis=-1)))[0]
        except IndexError:
            return []
    
    def color_at(self, x, y, data=None):
        if data is None:
            data = self.data
        return tuple(data[x, y])
    
    def float_color_at(self, x, y, data=None):
        if data is None:
            data = self.identity
        return (channel/255.0 for channel in self.color_at(x, y, data=data))


class LUT(RGBTable):
    
    def __init__(self, name='identity'):
        RGBTable.__init__(self)
        self.name = name
        self.data = self._read_png_matrix(self.name)
    
    @classmethod
    def _read_png_matrix(cls, name):
        print "Reading LUT image: %s" % name
        return imread.imread(
            static.path(join('lut', '%s.png' % name)))


def main():

    RGB = ColorType('RGB')
    RGB24 = ColorType('RGB', dtype=numpy.uint8)
    YCrCb = ColorType('YCrCb', dtype=numpy.uint16)
    
    print RGB(2, 3, 4)
    print RGB24
    print YCrCb(8, 88, 808)
    
    identity = LUT()
    amatorka = LUT('amatorka')
    
    print ""
    print identity[RGB(146,146,36)]
    print identity[RGB(22,33,44)]
    print identity[RGB(255, 25, 25)]
    
    print ""
    print "YO DOGG"
    print amatorka[RGB(146,146,36)]
    print identity[RGB(22,33,44)]
    print identity[RGB(255, 25, 25)]
    

def blurthday():
    
    from imread import imread
    from pprint import pprint
    imfuckingshowalready = lambda mx: Image.fromarray(mx).show()
    
    identity = LUT()
    amatorka = LUT('amatorka')
    #miss_etikate = LUT('miss_etikate')
    #soft_elegance_1 = LUT('soft_elegance_1')
    #soft_elegance_2 = LUT('soft_elegance_2')
    
    im1 = imread(static.path(join('img', '06-DSCN4771.JPG')))
    im2 = imread(static.path(join(
        'img', '430023_3625646599363_1219964362_3676052_834528487_n.jpg')))
    
    pprint(identity)
    pprint(amatorka)
    
    im9 = amatorka.transform(im1)
    pprint(im9)
    imfuckingshowalready(im9)
    print im1
    print im2


def old_maid():
    pass
    #global __multipons__
    #from pprint import pprint
    #pprint(__multipons__)

def old_main():
    
    #imfuckingshowalready = lambda mx: Image.fromarray(mx).show()

    old_identity = static.path(join('lut', 'identity.png'))

    im_old_identity = imread.imread(old_identity)
    im_identity = numpy.zeros_like(im_old_identity)

    for bx in xrange(0, 8):
        for by in xrange(0, 8):
            for r in xrange(0, 64):
                for g in xrange(0, 64):
                    im_identity[
                        int(g + by * 64),
                        int(r + bx * 64)] = numpy.array((
                            int(r * 255.0 / 63.0 + 0.5),
                            int(g * 255.0 / 63.0 + 0.5),
                                int((bx + by * 8.0) * 255.0 / 63.0 + 0.5)),
                                dtype=numpy.uint8)
    
    print "THE OLD: %s, %s, %s" % (
        im_old_identity.size, im_old_identity.shape,
        str(im_old_identity.dtype))
    #print im_old_identity
    print ""
    
    print "THE NEW: %s, %s, %s" % (
        im_identity.size, im_identity.shape,
        str(im_identity.dtype))
    #print im_identity
    print ""
    
    
    
    print "THE END: %s" % bool(im_old_identity.shape == im_identity.shape)
    #print im_old_identity == im_identity
    
    #imfuckingshowalready(im_identity)
    #imfuckingshowalready(im_old_identity)
    
    pil_im_old_identity = Image.fromarray(im_old_identity)
    pil_im_old_identity.save('/tmp/im_old_identity.jpg',
        format="JPEG")
    
    pil_im_identity = Image.fromarray(im_identity)
    pil_im_identity.save('/tmp/im_identity.jpg',
        format="JPEG")




if __name__ == '__main__':
    main()



