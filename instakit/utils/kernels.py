
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

def gaussian_blur_kernel(ndim, kernel):
    from scipy.ndimage import convolve
    return convolve(ndim,
        kernel, mode='reflect')

def gaussian_blur(ndim, sigma=3, nY=None):
    return gaussian_blur_kernel(ndim,
        gaussian(sigma, sizeY=nY))

def gaussian_blur_filter(ndim, sigma=3):
    from scipy.ndimage.filters import gaussian_filter
    return gaussian_filter(ndim,
        sigma=(sigma, sigma, 0),
        order=0,
        mode='reflect')