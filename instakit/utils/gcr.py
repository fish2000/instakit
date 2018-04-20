
from PIL import Image

def gcr(image, percentage, revert_mode=False):
    ''' basic "Gray Component Replacement" function. Returns a CMYK image* with 
        percentage gray component removed from the CMY channels and put in the
        K channel, ie. for percentage=100, (41, 100, 255, 0) >> (0, 59, 214, 41).
        
    {*} This is the default behavior – to return an image of the same mode as that
        of which was originally provided, pass the value for the (optional) keyword
        argument `revert_mode` as `True`.
    '''
    # from http://stackoverflow.com/questions/10572274/halftone-images-in-python
    
    if not percentage:
        return revert_mode and image or image.convert('CMYK')
    
    original_mode = image.mode
    cmyk_image = image.mode is 'CMYK' and image.split() or image.convert('CMYK').split()
    
    cmyk = []
    for idx in range(4):
        cmyk.append(cmyk_image[idx].load())
    
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            gray = min(cmyk[0][x, y],
                       cmyk[1][x, y],
                       cmyk[2][x, y]) * percentage / 100
            for idx in range(3):
                cmyk[idx][x, y] -= gray
            cmyk[3][x, y] = gray
    
    out = Image.merge('CMYK', cmyk_image)
    if revert_mode:
        out = out.convert(original_mode)
    
    return out
