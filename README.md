INSTAKIT
========

Instakit<sup><b>†</b></sup> is a suite of image processors and filters, for processing PIL images.

InstaKit processors use the same API as [PILKit](https://github.com/matthewwithanm/pilkit)'s,
so they can be used with anything that supports those, including [ImageKit](https://github.com/matthewwithanm/django-imagekit).
Or you can just use them by themselves to process images using Python.

![one](http://i.imgur.com/pQ6Vw.jpg)

Image Processors and Utilities On Offer:
-----------------------------------------------------

* `instakit.processors.adjust`

    * `Color(0.0 – 1.0)`
    * `Brightness(0.0 – 1.0)`
    * `Contrast(0.0 – 1.0)`
    * `Sharpness(0.0 – 1.0)`
    * `Invert()`
    * `Equalize([mask])`
    * `AutoContrast([cutoff{uint8_t}, [ignore{uint8_t}]])`
    * `Solarize([threshold{uint8_t}])`
    * `Posterize([bits{2**n}])`

* `instakit.processors.blur`

    * `Contour()`
    * `Detail()`
    * `Emboss()`
    * `FindEdges()`
    * `EdgeEnhance()`
    * `EdgeEnhanceMore()`
    * `Smooth()`
    * `SmoothMore()`
    * `Sharpen()`
    * `UnsharpMask([radius=2, [percent=150, [threshold=3]]])`
    * `SimpleGaussianBlur([radius=2])`
    * `GaussianBlur([sigmaX=3, [sigmaY=3, [sigmaZ=3]]])`
    
* `instakit.processors.curves`

    * `InterpolationMode`
        * `LINEAR`, `NEAREST`, `ZERO`, `SLINEAR`, `QUADRATIC`, `CUBIC`, `PREVIOUS`, `NEXT`, `LAGRANGE`
    * `CurveSet(<FILE.ACV>, [InterpolationMode.LAGRANGE])`

* `instakit.processors.halftone`

    * `Atkinson([threshold{uint8_t}])`
    * `FloydSteinberg([threshold{uint8_t}])`
    * `SlowAtkinson([threshold{uint8_t}])`
    * `SlowFloydSteinberg([threshold{uint8_t}])`
    * `CMYKAtkinson([gcr=20{%}])`
    * `CMYKFloydsterBill([gcr=20{%}])`
    * `DotScreen([sample=1, [scale=2, [angle=0{°}]]])`
    * `CMYKDotScreen([gcr=20{%}, [sample=10, [scale=10, [thetaC=0{°}, [thetaM=15{°}, [theta=30{°}, [thetaK=45{°}]]]]]]])`

* `instakit.processors.noise`

    * `GaussianNoise()`
    * `PoissonNoise()`
    * `GaussianLocalVarianceNoise()`
    * `SaltNoise()`
    * `PepperNoise()`
    * `SaltAndPepperNoise()`
    * `SpeckleNoise()`

* `instakit.processors.squarecrop`

    * `histogram_entropy(image)`
    * `SquareCrop()` … smart!

* `instakit.utils.ext.api` (Cythonized)

    * `hsluv_to_rgb(…)`
    * `rgb_to_hsluv(…)`
    * `hpluv_to_rgb(…)`
    * `rgb_to_hpluv(…)`

* `instakit.utils.gcr`

    * `gcr(image, [percentage=20{%}, [revert_mode=False]])`
    * `BasicGCR([percentage=20{%}, [revert_mode=False]])`

* `instakit.utils.kernels`
* `instakit.utils.lutmap`
* `instakit.utils.mode`

    * `Mode`
        * `MONO`, `L`, `I`, `F`, `P`, `RGB`, `RGBX`, `RGBA`, `CMYK`, `YCbCr`, `LAB`, `HSV`, `RGBa`, `LA`, `La`, `PA`, `I16`, `I16L`, `I16B`
        * Many useful `PIL.Image` delegate methods (q.v. source)

* `instakit.utils.ndarrays`
* `instakit.utils.pipeline`

    * `Pipe`, `Ink`, `NOOp`
    * `CMYKInk` and `RGBInk`
    * `ChannelFork` and `ChannelOverprinter`

* `instakit.utils.static`
* `instakit.utils.stats`


![two](http://i.imgur.com/ln1Eq.jpg)

![three](http://i.imgur.com/MBuC5.jpg)

As of this first draft there are [Instagrammy](http://www.instagram.com/) image-curve adjusters and a few other geegaws.

Instakit is made available to you and the public at large under the [MIT license](http://opensource.org/licenses/MIT) -- see LICENSE.md for the full text.

 † née “django-instakit” – All dependencies and traces of Django have since been excised, with thanks to [matthewwithanm](https://github.com/matthewwithanm).
