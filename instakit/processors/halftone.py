#!/usr/bin/env python
# encoding: utf-8
"""
halftone.py

Created by FI$H 2000 on 2012-08-23.
Copyright (c) 2012 Objects In Space And Time, LLC. All rights reserved.
"""
from __future__ import print_function

from PIL import ImageDraw

from instakit.utils import pipeline, gcr
from instakit.utils.mode import Mode
from instakit.utils.stats import histogram_mean
from instakit.abc import Processor, ThresholdProcessor

class SlowAtkinson(ThresholdProcessor):
    
    """ It’s not a joke, this processor is slow as fuck;
        if at all possible, use the cythonized version instead
        (q.v. instakit.processors.ext.Atkinson) and never ever
        use this one if at all possible – unless, like, you’re
        being paid by the hour or somesuch. Up to you dogg.
    """
    __slots__ = tuple()
    
    def process(self, image):
        """ The process call returns a monochrome ('L'-mode) image """
        image = Mode.L.process(image)
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                old = image.getpixel((x, y))
                new = self.threshold_matrix[old]
                err = (old - new) >> 3 # divide by 8.
                image.putpixel((x, y), new)
                for nxy in [(x+1, y),
                            (x+2, y),
                            (x-1, y+1),
                            (x, y+1),
                            (x+1, y+1),
                            (x, y+2)]:
                    try:
                        image.putpixel(nxy, int(
                        image.getpixel(nxy) + err))
                    except IndexError:
                        pass # it happens, evidently.
        return image

class SlowFloydSteinberg(ThresholdProcessor):
    
    """ A similarly super-slow reference implementation of Floyd-Steinberg.
        Adapted from an RGB version here: https://github.com/trimailov/qwer
    """
    __slots__ = tuple()
    
    # Precalculate fractional error multipliers:
    SEVEN_FRAC = 7/16
    THREE_FRAC = 3/16
    CINCO_FRAC = 5/16
    ALONE_FRAC = 1/16
    
    def process(self, image):
        """ The process call returns a monochrome ('L'-mode) image """
        # N.B. We store local references to the fractional error multipliers
        # to avoid the Python internal-dict-stuff member-lookup overhead:
        image = Mode.L.process(image)
        SEVEN_FRAC = type(self).SEVEN_FRAC
        THREE_FRAC = type(self).THREE_FRAC
        CINCO_FRAC = type(self).CINCO_FRAC
        ALONE_FRAC = type(self).ALONE_FRAC
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                old = image.getpixel((x, y))
                new = self.threshold_matrix[old]
                image.putpixel((x, y), new)
                err = old - new
                for nxy in [((x+1, y),      SEVEN_FRAC),
                            ((x-1, y+1),    THREE_FRAC),
                            ((x, y+1),      CINCO_FRAC),
                            ((x+1, y+1),    ALONE_FRAC)]:
                    try:
                        image.putpixel(nxy[0], int(
                        image.getpixel(nxy[0]) + err * nxy[1]))
                    except IndexError:
                        pass # it happens, evidently.
        return image

# Register the stub as a instakit.abc.Processor “virtual subclass”:
@Processor.register
class Problematic(object):
    def __init__(self):
        raise TypeError("Fast-math version couldn't be imported")

try:
    # My man, fast Bill Atkinson
    from instakit.processors.ext.halftone import Atkinson as FastAtkinson
except ImportError:
    Atkinson = SlowAtkinson
    FastAtkinson = Problematic
else:
    # Register the Cythonized processor with the ABC:
    Atkinson = Processor.register(FastAtkinson)

try:
    # THE FLOYDSTER
    from instakit.processors.ext.halftone import FloydSteinberg as FastFloydSteinberg
except ImportError:
    FloydSteinberg = SlowFloydSteinberg
    FastFloydSteinberg = Problematic
else:
    # Register the Cythonized processor with the ABC:
    FloydSteinberg = Processor.register(FastFloydSteinberg)

class CMYKAtkinson(Processor):
    
    """ Create a full-color CMYK Atkinson-dithered halftone, with gray-component
        replacement (GCR) at a specified percentage level
    """
    __slots__ = ('gcr', 'overprinter')
    
    def __init__(self, gcr=20):
        self.gcr = max(min(100, gcr), 0)
        self.overprinter = pipeline.BandFork(Atkinson, mode='CMYK')
    
    def process(self, image):
        return pipeline.Pipe(gcr.BasicGCR(self.gcr),
                                          self.overprinter).process(image)

class CMYKFloydsterBill(Processor):
    
    """ Create a full-color CMYK Atkinson-dithered halftone, with gray-component
        replacement (GCR) and OH SHIT SON WHAT IS THAT ON THE CYAN CHANNEL DOGG
    """
    __slots__ = ('gcr', 'overprinter')
    
    def __init__(self, gcr=20):
        self.gcr = max(min(100, gcr), 0)
        self.overprinter = pipeline.BandFork(Atkinson, mode='CMYK')
        self.overprinter.update({ 'C' : SlowFloydSteinberg() })
    
    def process(self, image):
        return pipeline.Pipe(gcr.BasicGCR(self.gcr),
                                          self.overprinter).process(image)

class DotScreen(Processor):
    
    """ This processor creates a monochrome dot-screen halftone pattern
        from an image. While this may be useful on its own, it is far
        more useful when used across all channels of a CMYK image in
        a BandFork or OverprintFork processor operation (q.v. sources
        of `instakit.utils.pipeline.BandFork` et al. supra.) serially,
        with either a gray-component replacement (GCR) or an under-color
        replacement (UCR) function.
        
        Regarding the latter two operations, instakit only has a basic
        GCR implementation currently, at the time of writing – q.v. the
        `instakit.utils.gcr` module sub.
        
        Adapted originally from this sample code:
            https://stackoverflow.com/a/10575940/298171
    """
    __slots__ = ('sample', 'scale', 'angle')
    
    def __init__(self, sample=1, scale=2, angle=0):
        self.sample = sample
        self.scale = scale
        self.angle = angle
    
    def process(self, image):
        orig_width, orig_height = image.size
        image = Mode.L.process(image).rotate(self.angle, expand=1)
        width, height = image.size
        halftone = Mode.L.new((width * self.scale,
                              height * self.scale))
        dotscreen = ImageDraw.Draw(halftone)
        
        SAMPLE = self.sample
        SCALE = self.scale
        ANGLE = self.angle
        
        for y in range(0, height, SAMPLE):
            for x in range(0, width, SAMPLE):
                cropbox = image.crop((x,          y,
                                      x + SAMPLE, y + SAMPLE))
                diameter = (histogram_mean(cropbox) / 255) ** 0.5
                edge = 0.5 * (1 - diameter)
                xpos, ypos = (x + edge) * SCALE, (y + edge) * SCALE
                boxedge = SAMPLE * diameter * SCALE
                dotscreen.ellipse((xpos,           ypos,
                                   xpos + boxedge, ypos + boxedge),
                                   fill=255)
        
        halftone = halftone.rotate(-ANGLE, expand=1)
        tone_width, tone_height = halftone.size
        xx = (tone_width  - orig_width  * SCALE) / 2
        yy = (tone_height - orig_height * SCALE) / 2
        return halftone.crop((xx,                      yy,
                              xx + orig_width * SCALE, yy + orig_height * SCALE))

class CMYKDotScreen(Processor):
    
    """ Create a full-color CMYK dot-screen halftone, with gray-component
        replacement (GCR), individual rotation angles for each channel’s
        dot-screen, and resampling value controls.
    """
    __slots__ = ('overprinter', 'sample', 'scale')
    
    def __init__(self,      gcr=20,
                 sample=10, scale=10,
                  thetaC=0, thetaM=15, thetaY=30, thetaK=45):
        """ Initialize an internal instakit.utils.pipeline.OverprintFork() """
        self.sample = sample
        self.scale = scale
        self.overprinter = pipeline.OverprintFork(None, gcr=gcr)
        self.overprinter['C'] = DotScreen(angle=thetaC, sample=sample, scale=scale)
        self.overprinter['M'] = DotScreen(angle=thetaM, sample=sample, scale=scale)
        self.overprinter['Y'] = DotScreen(angle=thetaY, sample=sample, scale=scale)
        self.overprinter['K'] = DotScreen(angle=thetaK, sample=sample, scale=scale)
        self.overprinter.apply_CMYK_inks()
    
    @property
    def gcr_percentage(self):
        return self.overprinter.basicgcr.percentage
    
    def angle(self, band_label):
        if band_label not in self.overprinter.band_labels:
            raise ValueError('invalid band label')
        return self.overprinter[band_label].angle
    
    @property
    def thetaC(self):
        """ Return the C-band halftone screen’s rotation """
        return self.angle('C')
    
    @property
    def thetaM(self):
        """ Return the M-band halftone screen’s rotation """
        return self.angle('M')
    
    @property
    def thetaY(self):
        """ Return the Y-band halftone screen’s rotation """
        return self.angle('Y')
    
    @property
    def thetaK(self):
        """ Return the K-band halftone screen’s rotation """
        return self.angle('K')
    
    def process(self, image):
        return self.overprinter.process(image)

def test():
    from instakit.utils.static import asset
    
    image_paths = list(map(
        lambda image_file: asset.path('img', image_file),
            asset.listfiles('img')))
    image_inputs = list(map(
        lambda image_path: Mode.RGB.open(image_path),
            image_paths))
    
    for image_input in image_inputs:
        image_input.show()
        
        # Atkinson(threshold=128.0).process(image_input).show()
        # FloydSteinberg(threshold=128.0).process(image_input).show()
        # SlowFloydSteinberg(threshold=128.0).process(image_input).show()
        
        # CMYKAtkinson().process(image_input).show()
        # CMYKFloydsterBill().process(image_input).show()
        CMYKDotScreen(sample=10, scale=4).process(image_input).show()
    
    print(image_paths)

if __name__ == '__main__':
    test()