
from PIL import Image

def gcr(im, percentage):
    ''' basic "Gray Component Replacement" function. Returns a CMYK image with 
        percentage gray component removed from the CMY channels and put in the
        K channel, ie. for percentage=100, (41, 100, 255, 0) >> (0, 59, 214, 41)
    '''
    # from http://stackoverflow.com/questions/10572274/halftone-images-in-python
    
    cmyk_im = im.convert('CMYK')
    if not percentage:
        return cmyk_im
    
    cmyk_im = cmyk_im.split()
    cmyk = []
    
    for i in xrange(4):
        cmyk.append(cmyk_im[i].load())
    
    for x in xrange(im.size[0]):
        for y in xrange(im.size[1]):
            gray = min(cmyk[0][x,y], cmyk[1][x,y], cmyk[2][x,y]) * percentage / 100
            for i in xrange(3):
                cmyk[i][x,y] = cmyk[i][x,y] - gray
            cmyk[3][x,y] = gray
    
    return Image.merge('CMYK', cmyk_im)
