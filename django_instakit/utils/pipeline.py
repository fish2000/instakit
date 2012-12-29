
from collections import defaultdict
from PIL import Image

class Pipe(list):
    """ derived from an ImageKit class:
        imagekit.processors.base.ProcessorPipeline """
    def process(self, img):
        for p in self:
            img = p.process(img)
        return img

class ChannelFork(defaultdict):
    
    default_mode = 'RGB'
    
    def __init__(self, default_factory, *args, **kwargs):
        if not callable(default_factory):
            raise AttributeError("ChannelFork() requires a callable default_factory.")
        
        from PIL import ImageMode
        self.channels = ImageMode.getmode(
            kwargs.pop('mode', self.default_mode))
        
        super(ChannelFork, self).__init__(default_factory, *args, **kwargs)
    
    def compose(self, *channels):
        return Image.merge(
            self.channels.mode,
            channels)
    
    def process(self, img):
        if img.mode != self.channels.mode:
            img = img.convert(self.channels.mode)
        
        processed_channels = []
        for idx, channel in enumerate(img.split()):
            processed_channels.append(
                self[self.channels.bands[idx]].process(channel))
        
        return self.compose(*processed_channels)

class CMYKInk(object):
    
    WHITE =     (255,   255,    255)
    CYAN =      (0,     250,    250)
    MAGENTA =   (250,   0,      250)
    YELLOW =    (250,   250,    0)
    KEY =       (0,     0,      0)
    CMYK =      (CYAN, MAGENTA, YELLOW, KEY)
    
    def __init__(self, ink_value=None):
        if ink_value is None:
            ink_value = self.KEY
        self.ink_value = ink_value
    
    def process(self, img):
        from PIL import ImageOps
        return ImageOps.colorize(
            img.convert('L'),
            self.WHITE,
            self.ink_value)


class ChannelOverprinter(ChannelFork):
    
    default_mode = 'CMYK'
    
    def compose(self, *channels):
        from PIL import ImageChops
        return reduce(ImageChops.multiply, channels)
    
    def process(self, img):
        inks = zip(self.default_mode,
            [CMYKInk(ink_label) \
                for ink_label in CMYKInk.CMYK])
        for channel_name, ink in inks:
            self[channel_name] = Pipe([
                self[channel_name], ink])
        return super(ChannelOverprinter, self).process(img)




if __name__ == '__main__':
    from django_instakit.utils import static
    from django_instakit.processors.halftone import Atkinson
    
    image_paths = map(
        lambda image_file: static.path('img', image_file),
            static.listfiles('img'))
    image_inputs = map(
        lambda image_path: Image.open(image_path).convert('RGB'),
            image_paths)
    
    for image_input in image_inputs[:2]:
        #image_input.show()
        #Atkinson().process(image_input).show()
        #NumAtkinson().process(image_input).show()
        ChannelOverprinter(Atkinson).process(image_input).show()
    
    print image_paths
    