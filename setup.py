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
import sys
import os
import os.path

try:
    from setuptools import setup, Extension
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, Extension
from Cython.Build import cythonize

def cython_module(*args, **kwargs):
    sources = []
    sources.extend(kwargs.pop('sources', []))
    include_dirs = []
    include_dirs.extend(kwargs.pop('include_dirs', []))
    ext_package = os.path.extsep.join(args)
    ext_pth = os.path.sep.join(args) + ".pyx"
    sources.insert(0, ext_pth)
    language = kwargs.pop('language', 'c')
    return Extension(ext_package, sources,
        language=language,
        include_dirs=include_dirs,
        extra_compile_args=['-Wno-unused-function',
                            '-Wno-unneeded-internal-declaration',
                            '-O3',
                            '-fstrict-aliasing',
                            '-funroll-loops',
                            '-mtune=native'])

def cython_processor(name, **kwargs):
    return cython_module('instakit', 'processors', 'ext', name, **kwargs)

def cython_utility(name, **kwargs):
    return cython_module('instakit', 'utils', 'ext', name, **kwargs)

# VERSION & METADATA
__version__ = "<undefined>"
try:
    exec(compile(
        open(os.path.join(
             os.path.dirname(__file__),
            '__version__.py')).read(),
            '__version__.py', 'exec'))
except:
    __version__ = '0.4.4'

from Cython.Distutils import build_ext
from distutils.sysconfig import get_python_inc

name = 'instakit'
description = 'Image processors and filters.'
keywords = 'python django imagekit image processing filters'

long_description = """
Image processors and filters, inspired by Instagram, and ready for use
with django-imagekit.

Included are filters for Atkinson-dither halftoning,  dot-pitch halftoning
(with GCR and per-channel pipeline processors), classes for Numpy-based
image processors, Gaussian kernel functions, processors for applying curves
to images from Photoshop .acv files, imagekit-ready processors exposing
Pillow’s many image adjustment algorithms (e.g. noise and blur functions,
histogram-based adjustments like Brightness/Contrast, and many others),
and an implementation of the entropy-based smart-crop algorithm that many
know from the easy-thumbnails Django app.

Experienced users may also make use of the many utilities shipping with
instakit: LUT maps, color structs, pipeline processing primitives and 
ink-based separation simulation tools, and other related miscellany.
"""

classifiers = [
    'Development Status :: 5 - Production/Stable']

try:
    import numpy
except ImportError:
    class FakeNumpy(object):
        def get_include(self):
            return os.path.curdir
    numpy = FakeNumpy()

try:
    from setuptools import find_packages

except ImportError:
    def is_package(path):
        return (os.path.isdir(path) and \
                os.path.isfile(
                os.path.join(path, '__init__.py')))
    
    def find_packages(path, base=""):
        """ Find all packages in path
            See also http://wiki.python.org/moin/Distutils/Cookbook/AutoPackageDiscovery
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

include_dirs = [
    os.path.curdir,
    numpy.get_include(),
    get_python_inc(plat_specific=1)]

setup(
    name=name,
    version=__version__,
    description=description,
    long_description=long_description,
    keywords=keywords, platforms=['any'],
    
    author=u"Alexander Bohn", author_email='fish2000@gmail.com',
    
    license='MIT',
    url='http://github.com/fish2000/%s' % name,
    download_url='http://github.com/fish2000/%s/zipball/master' % name,
    
    packages=find_packages(),
    package_data={ '' : ['*%s*' % os.path.extsep] },
    include_package_data=True,
    zip_safe=False,
    
    install_requires=[
        'Cython',
        'Pillow',
        'numpy',
        'scipy',
        'scikit-image'],
    
    ext_modules=cythonize([
        cython_processor("halftone", include_dirs=include_dirs),
        cython_utility("api", sources=[hsluv_source])
        ], compiler_directives=dict(language_level=3,
                                    infer_types=True,
                                    embedsignature=True)),
    
    cmdclass=dict(build_ext=build_ext),
    include_dirs=include_dirs,
    classifiers=classifiers+[
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: OS Independent',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
