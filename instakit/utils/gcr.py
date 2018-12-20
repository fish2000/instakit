
from PIL import Image, ImageMode

RGB = ImageMode.getmode('RGB')
CMYK = ImageMode.getmode('CMYK')
cmyk = CMYK.mode

def gcr(image, percentage=20, revert_mode=False):
    ''' basic "Gray Component Replacement" function. Returns a CMYK image* with 
        percentage gray component removed from the CMY channels and put in the
        K channel, ie. for percentage=100, (41, 100, 255, 0) >> (0, 59, 214, 41).
        
    {*} This is the default behavior – to return an image of the same mode as that
        of which was originally provided, pass the value for the (optional) keyword
        argument `revert_mode` as `True`.
    '''
    # from http://stackoverflow.com/questions/10572274/halftone-images-in-python
    
    if percentage is None:
        return revert_mode and image or image.convert(cmyk)
    
    if percentage > 100 or percentage < 1:
        raise ValueError("Do you not know how percents work??!")
    
    percent = percentage / 100
    
    original_mode = ImageMode.getmode(image.mode)
    cmyk_channels = original_mode == CMYK and image.split() or image.convert(cmyk).split()
    
    cmyk_image = []
    for channel in cmyk_channels:
        cmyk_image.append(channel.load())
    
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            gray = int(min(cmyk_image[0][x, y],
                           cmyk_image[1][x, y],
                           cmyk_image[2][x, y]) * percent)
            cmyk_image[0][x, y] -= gray
            cmyk_image[1][x, y] -= gray
            cmyk_image[2][x, y] -= gray
            cmyk_image[3][x, y] = gray
    
    out = Image.merge(cmyk, cmyk_channels)
    
    if revert_mode:
        return out.convert(original_mode.mode)
    return out


if __name__ == '__main__':
    from instakit.utils import static
    
    image_paths = list(map(
        lambda image_file: static.path('img', image_file),
            static.listfiles('img')))
    image_inputs = list(map(
        lambda image_path: Image.open(image_path).convert(RGB.mode),
            image_paths))
    
    for image_input in image_inputs:
        gcred = gcr(image_input)
        assert gcred.mode == CMYK.mode == cmyk
        gcred.show()
    
    print(image_paths)
