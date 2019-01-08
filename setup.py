#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    INSTAKIT -- Instagrammy PIL-based processors and tools
#
#    Copyright © 2012-2025 Alexander Bohn
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy 
#    of this software and associated documentation files (the "Software"), to deal 
#    in the Software without restriction, including without limitation the rights 
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
#    copies of the Software, and to permit persons to whom the Software is 
#    furnished to do so, subject to the following conditions:
#    
#    The above copyright notice and this permission notice shall be included in all 
#    copies or substantial portions of the Software.
#    
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
#    SOFTWARE.
#

from __future__ import print_function
from io import open

import sys
import os
import os.path

# HOST PYTHON VERSION
python_version = float("%s%s%s" % (sys.version_info.major,
                                   os.extsep,
                                   sys.version_info.minor))

try:
    from setuptools import setup, Extension
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, Extension
from Cython.Build import cythonize
from distutils.sysconfig import get_python_inc

# CYTHON & C-API EXTENSION MODULES
def cython_module(*args, **kwargs):
    sources = []
    sources.extend(kwargs.pop('sources', []))
    include_dirs = []
    include_dirs.extend(kwargs.pop('include_dirs', []))
    ext_package = os.path.extsep.join(args)
    ext_pth = os.path.sep.join(args) + os.extsep + "pyx"
    sources.insert(0, ext_pth)
    language = kwargs.pop('language', 'c')
    extra_compile_args = ['-Wno-unused-function',
                          '-Wno-unneeded-internal-declaration',
                          '-O3',
                          '-fstrict-aliasing',
                          '-funroll-loops',
                          '-mtune=native']
    if language.lower() == 'c++' or \
       language.lower() == 'mm':
        extra_compile_args.extend(['-std=c++17',
                                   '-stdlib=libc++',
                                   '-Wno-sign-compare',
                                   '-Wno-unused-private-field'])
    return Extension(ext_package, sources,
        language=language,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args)

def cython_comparator(name, **kwargs):
    return cython_module('instakit', 'comparators', 'ext', name, **kwargs)

def cython_processor(name, **kwargs):
    return cython_module('instakit', 'processors', 'ext', name, **kwargs)

def cython_utility(name, **kwargs):
    return cython_module('instakit', 'utils', 'ext', name, **kwargs)

# PROJECT VERSION & METADATA
__version__ = "<undefined>"
try:
    exec(compile(
        open(os.path.join(
             os.path.dirname(__file__),
            '__version__.py')).read(),
            '__version__.py', 'exec'))
except:
    __version__ = '0.6.1'

name = 'instakit'

keywords = " ".join(('instakit',
                     'django',
                     'imagekit',
                     'image processing',
                     'halftone',
                     'dithering',
                     'Cython',
                     'Photoshop',
                     'PIL',
                     'Pillow',
                     'NumPy',
                     'SciPy',
                     'scikit-image',
                     'acv',
                     'curves'))

# PROJECT DESCRIPTION
description = 'Image processing tools based on PIL/Pillow and scikit-image'

longer_description = """Instakit: Filters and Tools; BYO Facebook Buyout
========================================================================

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
many will recognize from the easy-thumbnails Django app - and much more."""

long_description_etc = """
Experienced users may also make use of the many utilities shipping with
instakit: LUT maps, color structs, pipeline processing primitives and 
ink-based separation simulation tools, Enums and wrapper APIs to simplify
PIL's rougher edges – like (say) image modes and compositing - plus other
related miscellany for the enterprising programmer. """

# LICENSE
license = open(os.path.join(
               os.path.dirname(__file__),
               'LICENSE.txt')).read()

# REQUIRED DEPENDENCIES
install_requires = [
    'Cython>=0.29.0',
    'Pillow>=3.0.0',
    'numpy>=1.7.0',
    'scipy>=1.1.0',
    'scikit-image>=0.10.0']

if python_version < 3.4:
    install_requires.append('enum34>=1.1.0')

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'License :: OSI Approved :: MIT License',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: OS Independent',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]

try:
    import numpy
except ImportError:
    class FakeNumpy(object):
        def get_include(self):
            return os.path.curdir
    numpy = FakeNumpy()

# SETUPTOOLS: FIND SUBORDINATE PACKAGES
try:
    from setuptools import find_packages

except ImportError:
    def is_package(path):
        return (os.path.isdir(path) and \
                os.path.isfile(
                os.path.join(path, '__init__.py')))
    
    def find_packages(path, base=""):
        """ Find all packages in path; see also:
            http://wiki.python.org/moin/Distutils/Cookbook/AutoPackageDiscovery
        """
        packages = {}
        for item in os.listdir(path):
            pth = os.path.join(path, item)
            if is_package(pth):
                if base:
                    module_name = "%(base)s.%(item)s" % vars()
                else:
                    module_name = item
                packages[module_name] = pth
                packages.update(
                    find_packages(pth, module_name))
        return packages

# SETUPTOOLS: CLEAN BUILD ARTIFACTS
if 'sdist' in sys.argv:
    import subprocess
    finder = "/usr/bin/find %s \( -iname \*.pyc -or -name .DS_Store \) -delete"
    theplace = os.getcwd()
    if theplace not in (os.path.sep, os.path.curdir):
        print("+ Deleting crapola from %s..." % theplace)
        print("$ %s" % finder % theplace)
        output = subprocess.getoutput(finder % theplace)
        print(output)

instakit_base_path = os.path.join(
                     os.path.abspath(
                     os.path.dirname(__file__)), 'instakit')

hsluv_source = os.path.join(os.path.relpath(instakit_base_path,
                      start=os.path.dirname(__file__)), 'utils',
                                                        'ext',
                                                        'hsluv.c')

butteraugli_source = os.path.join(os.path.relpath(instakit_base_path,
                            start=os.path.dirname(__file__)), 'comparators',
                                                              'ext',
                                                              'butteraugli.cc')

include_dirs = [
    os.path.curdir,
    numpy.get_include(),
    get_python_inc(plat_specific=1)]

setup(
    name=name,
    version=__version__,
    author="Alexander Böhn",
    author_email='fish2000@gmail.com',
    
    description=description,
    long_description=longer_description,
    long_description_content_type="text/markdown",
    
    keywords=keywords,
    url='https://github.com/fish2000/instakit',
    download_url='https://github.com/fish2000/instakit/zipball/master',
    classifiers=classifiers,
    license=license, platforms=['any'],
    
    packages=find_packages(),
    package_data={ '' : ['*.*'] },
    include_package_data=True,
    zip_safe=False,
    
    install_requires=install_requires,
    include_dirs=include_dirs,
    
    ext_modules=cythonize([
        cython_comparator("buttereye",  sources=[butteraugli_source],
                                        language="c++"),
        cython_processor("halftone",    include_dirs=include_dirs,
                                        language="c"),
        cython_utility("api",           sources=[hsluv_source],
                                        language="c")
        ], compiler_directives=dict(language_level=3,
                                    infer_types=True,
                                    embedsignature=True)),
)
