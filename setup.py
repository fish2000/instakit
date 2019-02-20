#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    INSTAKIT -- Instagrammy image-processors and tools, based on Pillow and SciPy
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
''' Image processors for django-imagekit - based on Pillow, SciPy, and scikit-image '''

from __future__ import print_function
import os, sys, sysconfig

from psutil import cpu_count
from setuptools import setup
from Cython.Build import cythonize

# HOST PYTHON VERSION
PYTHON_VERSION = float("%s%s%s" % (sys.version_info.major, os.extsep,
                                   sys.version_info.minor))

# CONSTANTS
PROJECT_NAME = 'instakit'
AUTHOR_NAME = 'Alexander Böhn'
AUTHOR_USER = 'fish2000'

GITHUB = 'github.com'
GMAIL = 'gmail.com'

AUTHOR_EMAIL = '%s@%s' % (AUTHOR_USER, GMAIL)
PROJECT_GH_URL = 'https://%s/%s/%s' % (GITHUB,
                                       AUTHOR_USER,
                                       PROJECT_NAME)
PROJECT_DL_URL = '%s/zipball/master' % PROJECT_GH_URL

KEYWORDS = ('django',
            'imagekit', PROJECT_NAME,
                        AUTHOR_USER,
            'image processing',
            'image analysis',
            'image comparison',
            'halftone', 'dithering',
            'Photoshop', 'acv', 'curves',
            'PIL', 'Pillow',
            'Cython',
            'NumPy', 'SciPy', 'scikit-image')

CPPLANGS = ('c++', 'cxx', 'cpp', 'cc', 'mm')

# PROJECT DIRECTORY
CWD = os.path.dirname(__file__)
BASE_PATH = os.path.join(
            os.path.abspath(CWD), PROJECT_NAME)

def project_content(filename):
    import io
    filepath = os.path.join(CWD, filename)
    if not os.path.isfile(filepath):
        raise IOError("""File %s doesn't exist""" % filepath)
    out = ''
    with io.open(filepath, 'r') as handle:
        out += handle.read()
    if not out:
        raise ValueError("""File %s couldn't be read""" % filename)
    return out

# CYTHON & C-API EXTENSION MODULES
def cython_module(*args, **kwargs):
    from Cython.Distutils import Extension
    sources = []
    sources.extend(kwargs.pop('sources', []))
    include_dirs = []
    include_dirs.extend(kwargs.pop('include_dirs', []))
    ext_package = os.path.extsep.join(args)
    ext_pth = os.path.sep.join(args) + os.extsep + "pyx"
    sources.insert(0, ext_pth)
    language = kwargs.pop('language', 'c').lower()
    extra_compile_args = ['-Wno-unused-function',
                          '-Wno-unneeded-internal-declaration',
                          '-O3',
                          '-fstrict-aliasing',
                          '-funroll-loops',
                          '-mtune=native']
    if language in CPPLANGS:
        extra_compile_args.extend(['-std=c++17',
                                   '-stdlib=libc++',
                                   '-Wno-sign-compare',
                                   '-Wno-unused-private-field'])
    return Extension(ext_package, sources,
        language=language,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args)

def cython_comparator(name, **kwargs):
    return cython_module(PROJECT_NAME, 'comparators', 'ext', name, **kwargs)

def cython_processor(name, **kwargs):
    return cython_module(PROJECT_NAME, 'processors', 'ext', name, **kwargs)

def cython_utility(name, **kwargs):
    return cython_module(PROJECT_NAME, 'utils', 'ext', name, **kwargs)

def additional_source(*args):
    return os.path.join(
           os.path.relpath(BASE_PATH, start=CWD), *args)

# PROJECT VERSION & METADATA
__version__ = "<undefined>"
try:
    exec(compile(
        open(os.path.join(CWD,
            '__version__.py')).read(),
            '__version__.py', 'exec'))
except:
    __version__ = '0.6.8'

# PROJECT DESCRIPTION
LONG_DESCRIPTION = project_content('ABOUT.md')

# SOFTWARE LICENSE
LICENSE = project_content('LICENSE.txt')

# REQUIRED INSTALLATION DEPENDENCIES
INSTALL_REQUIRES = [
    'Cython>=0.29.0',
    'Pillow>=3.0.0',
    'numpy>=1.7.0',
    'scipy>=1.1.0',
    'scikit-image>=0.12.0']

if PYTHON_VERSION < 3.4:
    INSTALL_REQUIRES.append('enum34>=1.1.0')

# PYPI PROJECT CLASSIFIERS
CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'License :: OSI Approved :: MIT License',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: OS Independent',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7']

# NUMPY: C-API INCLUDE DIRECTORY
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
                    module_name = "%(base)s.%(item)s" % dict(base=base, item=item)
                else:
                    module_name = item
                packages[module_name] = pth
                packages.update(
                    find_packages(pth, module_name))
        return packages

# SETUPTOOLS: CLEAN BUILD ARTIFACTS
if 'sdist' in sys.argv:
    import subprocess
    finder = "/usr/bin/find %s \( -iname \*.pyc -or -iname .ds_store \) -print -delete"
    theplace = os.getcwd()
    if theplace not in (os.path.sep, os.path.curdir):
        print("+ Deleting crapola from %s..." % theplace)
        print("$ %s" % finder % theplace)
        output = subprocess.getoutput(finder % theplace)
        print(output)

# SOURCES & INCLUDE DIRECTORIES
hsluv_source = additional_source('utils', 'ext', 'hsluv.c')
augli_source = additional_source('comparators', 'ext', 'butteraugli.cc')
include_dirs = [numpy.get_include(),
                sysconfig.get_path('include')]

# THE CALL TO `setup(…)`
setup(
    name=PROJECT_NAME,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    
    version=__version__,
    description=__doc__,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    
    keywords=" ".join(KEYWORDS),
    url=PROJECT_GH_URL, download_url=PROJECT_DL_URL,
    license=LICENSE, platforms=['any'],
    classifiers=CLASSIFIERS,
    
    packages=find_packages(),
    package_data={ '' : ['*.*'] },
    include_package_data=True,
    zip_safe=False,
    
    install_requires=INSTALL_REQUIRES,
    include_dirs=include_dirs,
    
    ext_modules=cythonize([
        cython_comparator("buttereye",  sources=[augli_source],
                                        language="c++"),
        cython_processor("halftone",    include_dirs=include_dirs,
                                        language="c"),
        cython_utility("api",           sources=[hsluv_source],
                                        language="c")
        ], nthreads=cpu_count(),
           compiler_directives=dict(language_level=3,
                                    infer_types=True,
                                    embedsignature=True)),
)
