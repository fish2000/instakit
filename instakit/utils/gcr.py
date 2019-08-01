#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function
from math import fabs, pow as mpow

from instakit.utils.mode import Mode
from instakit.abc import Processor
from instakit.exporting import Exporter

exporter = Exporter(path=__file__)
export = exporter.decorator()

PERCENT_ADMONISHMENT = "Do you not know how percents work??!"

@export
def gcr(image, percentage=20, revert_mode=False):
    ''' basic “Gray Component Replacement” function. Returns a CMYK image* with 
        percentage gray component removed from the CMY channels and put in the
        K channel, e.g. for percentage=100, (41, 100, 255, 0) >> (0, 59, 214, 41).
        
    {*} This is the default behavior – to return an image of the same mode as that
        of which was originally provided, pass the value for the (optional) keyword
        argument `revert_mode` as `True`.
    '''
    # from http://stackoverflow.com/questions/10572274/halftone-images-in-python
    
    if percentage is None:
        return revert_mode and image or Mode.CMYK.process(image)
    
    if percentage > 100 or percentage < 1:
        raise ValueError(PERCENT_ADMONISHMENT)
    
    percent = percentage / 100
    
    original_mode = Mode.of(image)
    cmyk_channels = Mode.CMYK.process(image).split()
    width, height = image.size
    
    cmyk_image = []
    for channel in cmyk_channels:
        cmyk_image.append(channel.load())
    
    for x in range(width):
        for y in range(height):
            gray = int(min(cmyk_image[0][x, y],
                           cmyk_image[1][x, y],
                           cmyk_image[2][x, y]) * percent)
            cmyk_image[0][x, y] -= gray
            cmyk_image[1][x, y] -= gray
            cmyk_image[2][x, y] -= gray
            cmyk_image[3][x, y] = gray
    
    recomposed = Mode.CMYK.merge(*cmyk_channels)
    
    if revert_mode:
        return original_mode.process(recomposed)
    return recomposed

@export
class BasicGCR(Processor):
    
    __slots__ = ('percentage', 'revert_mode')
    __doc__ = gcr.__doc__
    
    def __init__(self, percentage=20, revert_mode=False):
        if percentage is None:
            raise ValueError(PERCENT_ADMONISHMENT)
        if percentage > 100 or percentage < 1:
            raise ValueError(PERCENT_ADMONISHMENT)
        self.percentage = percentage
        self.revert_mode = revert_mode
    
    def process(self, image):
        return gcr(image, percentage=self.percentage,
                          revert_mode=self.revert_mode)

@export
def hex2rgb(h):
    """ Convert a hex string or number to an RGB triple """
    # q.v. https://git.io/fh9E2
    if isinstance(h, str):
        return hex2rgb(int(h[1:] if h.startswith('#') else h, 16))
    return (h >> 16) & 0xff, (h >> 8) & 0xff, h & 0xff

@export
def compand(v):
    """ Compand a linearized value to an sRGB byte value """
    # q.v. http://www.brucelindbloom.com/index.html?Math.html
    V = (v <= 0.0031308) and (v * 12.92) or fabs((1.055 * mpow(v, 1 / 2.4)) - 0.055)
    return int(V * 255.0)

@export
def uncompand(A):
    """ Uncompand an sRGB byte value to a linearized value """
    # q.v. http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    V = A / 255.0
    return (V <= 0.04045) and (V / 12.92) or mpow(((V + 0.055) / 1.055), 2.4)

@export
def ucr(image, revert_mode=False):
    ''' basic “Under-Color Removal” function. Returns a CMYK image* in which regions
        containing overlapping C, M, and Y ink are replaced with K (“Key”, née “BlacK”)
        ink. Images are first converted to RGB and linearized in order to perform the UCR
        operation in linear space. E.g.:
        
        0xFFDE17 >  rgb(255, 222, 23)
                 >  RGB(1.0, 0.7304607400903537, 0.008568125618069307)
                 >  CMY(0.0, 0.26953925990964633, 0.9914318743819307)
        
        rgb() > RGB() > CMY() > CMYK(41, 100, 255, 0) >> cmyk(0, 59, 214, 41).
        
    {*} This is the default behavior – to return an image of the same mode as that
        of which was originally provided, pass the value for the (optional) keyword
        argument `revert_mode` as `True`.
    '''
    # Adapted from http://www.easyrgb.com/en/math.php#text13
    # N.B. this is not, out of the gate, particularly well-optimized
    
    original_mode = Mode.of(image)
    width, height = image.size
    cmyk_target = Mode.CMYK.new(image.size, color=0)
    rgb_channels = Mode.RGB.process(image).split()
    cmyk_channels = cmyk_target.split()
    
    rgb_image = []
    for channel in rgb_channels:
        rgb_image.append(channel.load())
    
    cmyk_image = []
    for channel in cmyk_channels:
        cmyk_image.append(channel.load())
    
    for x in range(width):
        for y in range(height):
            # Get the rgb byte values:
            rgb = (rgb_image[0][x, y],
                   rgb_image[1][x, y],
                   rgb_image[2][x, y])
            
            # Uncompand rgb bytes to linearized RGB:
            RGB = (uncompand(v) for v in rgb)
            
            # Convert linear RGB to linear CMY:
            (C, M, Y) = (1.0 - V for V in RGB)
            
            # Perform simple UCR with the most combined
            # overlapping C/M/Y ink values:
            K = min(C, M, Y, 1.0)
            
            if K == 1:
                C = M = Y = 0
            else:
                denominator = (1 - K)
                C = (C - K) / denominator
                M = (M - K) / denominator
                Y = (Y - K) / denominator
            
            # Recompand linear CMYK to cmyk byte values for Pillow:
            cmyk_image[0][x, y] = compand(C)
            cmyk_image[1][x, y] = compand(M)
            cmyk_image[2][x, y] = compand(Y)
            cmyk_image[3][x, y] = compand(K)
    
    recomposed = Mode.CMYK.merge(*cmyk_channels)
    
    if revert_mode:
        return original_mode.process(recomposed)
    return recomposed

@export
class BasicUCR(Processor):
    
    __slots__ = ('revert_mode',)
    __doc__ = ucr.__doc__
    
    def __init__(self, revert_mode=False):
        self.revert_mode = revert_mode
    
    def process(self, image):
        return ucr(image, revert_mode=self.revert_mode)

@export
class DemoUCR(object):
    
    """ Demonstrate each phase of the UCR color-conversion process """
    
    def __init__(self, hex_triple):
        """ Initialize with a hex-encoded RGB triple, either as a string
            or as a hexadecimal integer, e.g.:
                
                >>> onedemo = DemoUCR(0xD8DCAB)
                >>> another = DemoUCR('#6A9391')
        """
        self.hex_triple = hex_triple
    
    def calculate(self):
        self.rgb = self.get_rgb()
        self.RGB = self.get_RGB()
        self.CMY = self.get_CMY()
        self.CMYK = self.get_CMYK()
        self.cmyk = self.get_cmyk()
    
    def get_rgb(self):
        """ Return the rgb byte-value (0-255) 3-tuple corresponding to the
            initial hex value
        """
        return hex2rgb(self.hex_triple)
    
    def get_RGB(self):
        """ Return the linearized (uncompanded) RGB 3-tuple version of the
            rgb 3-byte value tuple
        """
        return tuple(uncompand(v) for v in self.rgb)
    
    def get_CMY(self):
        """ Return the linearized (uncompanded) CMY color-model analog of the
            linear RGB value 3-tuple
        """
        (C, M, Y) = (1.0 - V for V in self.RGB)
        return (C, M, Y)
    
    def get_CMYK(self):
        """ Return the UCR’ed -- the under-color removed -- linearized CMYK analog
            of the linear CMY color-model value tuple
        """
        K = min(*self.CMY, 1.0)
        (C, M, Y) = self.CMY
        
        if K == 1:
            C = M = Y = 0
        else:
            denominator = (1 - K)
            C = (C - K) / denominator
            M = (M - K) / denominator
            Y = (Y - K) / denominator
        return (C, M, Y, K)
    
    def get_cmyk(self):
        """ Return the companded cmyk byte-value (0-255) 4-tuple Pillow-friendly
            CMYK analog of the initial value
        """
        return tuple(compand(V) for V in self.CMYK)
    
    def stringify_demo_values(self):
        """ Return each of the values, step-by-step in the conversion process,
            admirably formatted in a handsome manner suitable for printing
        """
        self.calculate()
        return """
            
    %(hex_triple)s > rgb%(rgb)s
            > RGB%(RGB)s
            > CMY%(CMY)s
            > CMYK%(CMYK)s
            > cmyk%(cmyk)s
        
        """ % self.__dict__
    
    def __str__(self):
        """ Stringify yo self """
        return self.stringify_demo_values()

# Assign the modules’ `__all__` and `__dir__` using the exporter:
__all__, __dir__ = exporter.all_and_dir()

def test():
    from instakit.utils.static import asset
    from itertools import chain
    from os.path import relpath
    from pprint import pprint
    
    start = "/usr/local/Cellar/python/3.7.2_2/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages"
    
    image_paths = list(map(lambda image_file: asset.path('img', image_file), asset.listfiles('img')))
    image_inputs = list(map(lambda image_path: Mode.RGB.open(image_path), image_paths))
    # print("len:", len(list(image_paths)), len(list(image_inputs)), len(list(callables)))
    
    print('\t<<<<<<<<<<<<<<<------------------------------------------------------>>>>>>>>>>>>>>>')
    print()
    
    functions = (gcr, ucr)
    processors = (BasicGCR(), BasicUCR(), Mode.CMYK)
    callables = chain((processor.process for processor in processors), functions)
    
    image_components = zip(image_paths, image_inputs, callables)
    
    for path, image, process_functor in image_components:
        print("«TESTING: %s»" % relpath(path, start=start))
        print()
        tup = image.size + (image.mode,)
        print("¬ Input: %sx%s %s" % tup)
        print("¬ Calling functor on image…")
        result = process_functor(image)
        tup = result.size + (result.mode,)
        print("¬ Output: %sx%s %s" % tup)
        print("¬ Displaying…")
        print()
        result.show()
    
    print("«¡SUCCESS!»")
    print()
    
    print("«TESTING: MANUAL CALLABLES»")
    # print()
    
    if len(image_inputs):
        image = image_inputs.pop()
        
        # Test GCR function:
        gcred = gcr(image)
        assert gcred.mode == Mode.CMYK.value.mode
        assert Mode.of(gcred) is Mode.CMYK
        # gcred.show()
        
        # close image:
        image.close()
    
    if len(image_inputs):
        image = image_inputs.pop()
        
        # Test UCR function:
        ucred = ucr(image)
        assert ucred.mode == Mode.CMYK.value.mode
        assert Mode.of(ucred) is Mode.CMYK
        # ucred.show()
        
        # close image:
        image.close()
    
    if len(image_inputs):
        image = image_inputs.pop()
        
        # Test GCR processor:
        gcr_processor = BasicGCR()
        gcred = gcr_processor.process(image)
        assert gcred.mode == Mode.CMYK.value.mode
        assert Mode.of(gcred) is Mode.CMYK
        # gcred.show()
        
        # close image:
        image.close()
    
    if len(image_inputs):
        image = image_inputs.pop()
        
        # Test UCR processor:
        ucr_processor = BasicUCR()
        ucred = ucr_processor.process(image)
        assert ucred.mode == Mode.CMYK.value.mode
        assert Mode.of(ucred) is Mode.CMYK
        # ucred.show()
        
        # close image:
        image.close()
    
    print("«¡SUCCESS!»")
    print()
    
    print('\t<<<<<<<<<<<<<<<------------------------------------------------------>>>>>>>>>>>>>>>')
    print()
    
    pprint(list(relpath(path, start=start) for path in image_paths))
    print()
    
    print("«TESTING: DemoUCR ALGORITHM-STAGE TRACE PRINTER»")
    print()
    
    print(DemoUCR("#BB2F53"))
    print()
    
    print(DemoUCR(0x6F2039))
    print()


if __name__ == '__main__':
    test()