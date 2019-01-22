Instakit: Filters and Tools; BYO Facebook Buyout
=====================================

Image processors and filters - inspired by Instagram, built on top of the
PIL/Pillow, SciPy and scikit-image packages, accelerated with Cython, and
ready to use with PILKit and the django-imagekit framework.

Included are filters for Atkinson and Floyd-Steinberg dithering, dot-pitch
halftoning (with GCR and per-channel pipeline processors), classes exposing
image-processing pipeline data as NumPy ND-arrays, Gaussian kernel functions,
processors for applying channel-based LUT curves to images from Photoshop
.acv files, imagekit-ready processors furnishing streamlined access to a wide
schmorgasbord of Pillow's many image adjustment algorithms (e.g. noise, blur,
and sharpen functions, histogram-based operations like Brightness/Contrast,
among others), an implementation of the entropy-based smart-crop algorithm
many will recognize from the easy-thumbnails Django app - and much more.