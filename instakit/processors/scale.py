
from instakit.processors.ext.scale import HQx

class HQ2x(HQx):
    def __init__(self):
        super(HQx, self).__init__(factor=2)

class HQ3x(HQx):
    def __init__(self):
        super(HQx, self).__init__(factor=3)

class HQ4x(HQx):
    def __init__(self):
        super(HQx, self).__init__(factor=4)

if __name__ == '__main__':
    from PIL import Image
    from instakit.utils import static
    
    image_paths = map(
        lambda image_file: static.path('img', image_file),
            static.listfiles('img'))
    image_inputs = map(
        lambda image_path: Image.open(image_path).convert('RGB'),
            image_paths)
    
    for image_input in image_inputs:
        #image_input.show()
        HQ2x().process(image_input).show()
    
    print image_paths
    
