
from __future__ import division, print_function
import sys, os

try:
    import setuptools
except:
    print('''
setuptools not found.

On linux, the package is often called python-setuptools''')
    sys.exit(1)

from distutils.sysconfig import get_python_inc

try:
    import numpy
except ImportError:
    class FakeNumpy(object):
        def get_include(self):
            return "."
    numpy = FakeNumpy()


__version__ = "<undefined>"
exec(compile(open('PyImgC_Version.py').read(),
             'PyImgC_Version.py', 'exec'))

long_description = open('README.md').read()

undef_macros = []
define_macros = []

DEBUG = os.environ.get('DEBUG', '1')

define_macros.append(
     ('PY_ARRAY_UNIQUE_SYMBOL', 'PyImgC_PyArray_API_Symbol'))
define_macros.append(
     ('MAGICKCORE_QUANTUM_DEPTH', '16'))
define_macros.append(
     ('MAGICKCORE_HDRI_ENABLE', '1'))

if DEBUG:
    #undef_macros = ['NDEBUG']
    if DEBUG == '2':
        define_macros.append(
            ('IMGC_DEBUG', DEBUG))
        define_macros.append(
            ('_GLIBCXX_DEBUG', '1'))

include_dirs = [
    numpy.get_include(),
    get_python_inc(plat_specific=1),
    os.getcwd()]
library_dirs = []


for pth in (
    '/usr/local/include',
    '/usr/X11/include',
    '/usr/local/opt/imagemagick/include/ImageMagick-6'):
    if os.path.isdir(pth):
        include_dirs.append(pth)

for pth in (
    '/usr/local/lib',
    '/usr/X11/lib',
    '/usr/local/opt/imagemagick/lib'):
    if os.path.isdir(pth):
        library_dirs.append(pth)

extensions = {
    '_PyImgC': ["PyImgC/pyimgc.cpp"],
    '_structcode': ["PyImgC/structcode.cpp"],
}

libraries = []

ext_modules = [
    setuptools.Extension(key,
        libraries=libraries,
        library_dirs=library_dirs,
        include_dirs=include_dirs,
        sources=sources,
        undef_macros=undef_macros,
        define_macros=define_macros,
        extra_compile_args=[
            '-std=c++11',
            '-Wno-error=unused-command-line-argument-hard-error-in-future',
            '-Wno-unused-function',
            '-Wno-deprecated-writable-strings',
            '-Qunused-arguments',
        ]) for key, sources in extensions.items()]

packages = setuptools.find_packages()
#package_dir = { 'imread.tests': 'imread/tests' }
#package_data = { 'imread.tests': ['data/*.*', 'data/pvrsamples/*'] }
package_dir = dict()
package_data = dict()

classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Multimedia',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Topic :: Software Development :: Libraries',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: C++',
    'License :: OSI Approved :: MIT License']

setuptools.setup(name='imread',
    version=__version__,
    description='PyImgC: CImg bridge library',
    long_description=long_description,
    author='Alexander Bohn',
    author_email='fish2000@gmail.com',
    license='MIT',
    platforms=['Any'],
    classifiers=classifiers,
    url='http://github.com/fish2000/instakit',
    packages=packages,
    ext_modules=ext_modules,
    package_dir=package_dir,
    package_data=package_data,
    test_suite='nose.collector')