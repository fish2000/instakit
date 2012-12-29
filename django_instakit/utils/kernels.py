
def gaussian(size, sizeY=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    from scipy import mgrid, exp
    size = int(size)    
    if not sizeY:
        sizeY = size
    else:
        sizeY = int(sizeY)               
    x, y = mgrid[-size:size+1, -sizeY:sizeY+1]
    g = exp(-(x**2/float(size)+y**2/float(sizeY)))
    return (g / g.sum()).flatten()

def gaussian_blur(img, n=3, nY=None):
    from scipy import ndimage
    return ndimage.convolve(img,
        gaussian(n, sizeY=nY),
        mode='reflect')

def gaussian_blur_kernel(img, kernel):
    from scipy import ndimage
    return ndimage.convolve(img,
        kernel, mode='reflect')

def gaussian_blur_filter(img, sigma=3):
    from scipy import ndimage
    return ndimage.filters.gaussian_filter(
        img, sigma=(sigma, sigma, 0),
        order=0, mode='reflect')