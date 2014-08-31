
from __future__ import division, print_function
import sys, os

# SETUPTOOLS
try:
    import setuptools
except:
    print('''
setuptools not found.

On linux, the package is often called python-setuptools''')
    sys.exit(1)

# GOSUB: basicaly `backticks` (cribbed from plotdevice)
def gosub(cmd, on_err=True):
    """Run a shell command and return the output"""
    from subprocess import Popen, PIPE
    shell = isinstance(cmd, basestring)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=shell)
    out, err = proc.communicate()
    ret = proc.returncode
    if on_err:
        msg = '%s:\n' % on_err if isinstance(on_err, basestring) else ''
        assert ret==0, msg + (err or out)
    return out, err, ret


# PYTHON & NUMPY INCLUDES
from distutils.sysconfig import get_python_inc
from distutils.spawn import find_executable as which
try:
    import numpy
except ImportError:
    class FakeNumpy(object):
        def get_include(self):
            return "."
    numpy = FakeNumpy()

# VERSION & METADATA
__version__ = "<undefined>"
exec(compile(open('PyImgC_Version.py').read(),
             'PyImgC_Version.py', 'exec'))

long_description = open('README.md').read()


# COMPILATION
DEBUG = os.environ.get('DEBUG', '1')

# LIBS: ENABLED BY DEFAULT
USE_PNG = os.environ.get('USE_PNG', '16')
USE_TIFF = os.environ.get('USE_TIFF', '1')
USE_MAGICKPP = os.environ.get('USE_MAGICKPP', '1')
USE_FFTW3 = os.environ.get('USE_FFTW3', '1')
USE_OPENEXR = os.environ.get('USE_OPENEXR', '1')

# LIBS: disabled
USE_OPENCV = os.environ.get('USE_OPENCV', '0') # libtbb won't link

# 'other, misc'
USE_MINC2 = os.environ.get('USE_MINC2', '0')
USE_FFMPEG = os.environ.get('USE_FFMPEG', '0') # won't even work
USE_LAPACK = os.environ.get('USE_LAPACK', '0') # HOW U MAEK LINKED

undef_macros = []
define_macros = []
define_macros.append(
     ('PY_ARRAY_UNIQUE_SYMBOL', 'PyImgC_PyArray_API_Symbol'))

if DEBUG:
    undef_macros = ['NDEBUG']
    if int(DEBUG) > 2:
        define_macros.append(
            ('IMGC_DEBUG', DEBUG))
        define_macros.append(
            ('_GLIBCXX_DEBUG', '1'))

print(""" ********************* DEBUG: %s ********************* """ % DEBUG)

include_dirs = [
    numpy.get_include(),
    get_python_inc(plat_specific=1),
    os.getcwd()]

library_dirs = []

for pth in (
    '/usr/local/include',
    '/usr/X11/include'):
    if os.path.isdir(pth):
        include_dirs.append(pth)

for pth in (
    '/usr/local/lib',
    '/usr/X11/lib'):
    if os.path.isdir(pth):
        library_dirs.append(pth)

extensions = {
    '_PyImgC': ["PyImgC/pyimgc.cpp"],
    '_structcode': ["PyImgC/structcode.cpp"],
}

# the basics
libraries = ['png', 'jpeg', 'z', 'm', 'pthread', 'c++']

# the addenda
def parse_config_flags(config, config_flags=None):
    if config_flags is None: # need something in there
        config_flags = ['']
    for config_flag in config_flags:
        out, err, ret = gosub(' '.join([config, config_flag]))
        if len(out):
            for flag in out.split():
                if flag.startswith('-L'): # link path
                    if os.path.exists(flag[2:]) and flag[2:] not in library_dirs:
                        library_dirs.append(flag[2:])
                    continue
                if flag.startswith('-l'): # library link name
                    if flag[2:] not in libraries:
                        libraries.append(flag[2:])
                    continue
                if flag.startswith('-D'): # preprocessor define
                    macro = flag[2:].split('=')
                    if macro[0] not in dict(define_macros).keys():
                        if len(macro) < 2:
                            macro.append('1')
                        define_macros.append(tuple(macro))
                    continue
                if flag.startswith('-I'):
                    if os.path.exists(flag[2:]) and flag[2:] not in include_dirs:
                        include_dirs.append(flag[2:])
                    continue

# if we're using it, ask it how to fucking work it
if int(USE_TIFF):
    parse_config_flags(
        which('pkg-config'),
        ('libtiff-4 --libs', 'libtiff-4 --cflags'))
    define_macros.append(
        ('cimg_use_tiff', '1'))

if int(USE_PNG):
    libpng_pkg = 'libpng'
    if USE_PNG.strip().endswith('6'):
        libpng_pkg += '16' # use 1.6
    elif USE_PNG.strip().endswith('5'):
        libpng_pkg += '15' # use 1.5
    parse_config_flags(
        which('pkg-config'), (
            '%s --libs' % libpng_pkg,
            '%s --cflags' % libpng_pkg))
    define_macros.append(
        ('cimg_use_png', '1'))

if int(USE_MAGICKPP):
    # Linking to ImageMagick++ calls for a bunch of libraries and paths,
    # all with crazy names that change depending on compile options
    parse_config_flags(
        which('Magick++-config'),
        ('--ldflags', '--cppflags'))
    define_macros.append(
        ('cimg_use_magick', '1'))

if int(USE_MINC2):
    # I have no idea what this library does (off by default)
    parse_config_flags(
        which('pkg-config'),
        ('minc2 --libs', 'minc2 --cflags'))
    define_macros.append(
        ('cimg_use_minc2', '1'))

if int(USE_FFTW3):
    # FFTW3 has been config'd for three pkgs:
    # fftw3 orig, fftwl (long? like long integers?),
    # and fftw3f (floats? fuckery? fiber-rich?) --
    # hence this deceptively non-repetitive flag list:
    parse_config_flags(
        which('pkg-config'), (
        'fftw3f --libs-only-l',
        'fftw3l --libs-only-l',
        'fftw3 --libs', 'fftw3 --cflags'))
    define_macros.append(
        ('cimg_use_fftw3', '1'))

if int(USE_OPENEXR):
    # Linking OpenEXR pulls in ilmBase, which includes its own
    # math and threading libraries... WATCH OUT!!
    parse_config_flags(
        which('pkg-config'),
        ('OpenEXR --libs', 'OpenEXR --cflags'))
    define_macros.append(
        ('cimg_use_openexr', '1'))

if int(USE_OPENCV):
    # Linking OpenCV gets you lots more including TBB and IPL,
    # and also maybe ffmpeg, I think somehow
    parse_config_flags(
        which('pkg-config'),
        ('opencv --libs', 'opencv --cflags'))
    out, err, ret = gosub('brew --prefix tbb')
    if out:
        library_dirs.append(os.path.join(out.strip(), 'lib'))
        include_dirs.append(os.path.join(out.strip(), 'include'))
    define_macros.append(
        ('cimg_use_opencv', '1'))

ext_modules = [
    setuptools.Extension(key,
        libraries=map(
            lambda lib: lib.endswith('.dylib') and lib.split('.')[0] or lib,
                libraries),
        library_dirs=library_dirs,
        include_dirs=include_dirs,
        sources=sources,
        undef_macros=undef_macros,
        define_macros=define_macros,
        language='c++',
        extra_compile_args=[
            '-O2',
            '-std=c++11',
            '-stdlib=libc++',
            '-Wno-error=unused-command-line-argument-hard-error-in-future',
            '-Wno-unused-function',
            '-Wno-deprecated-register', # CImg w/OpenEXR throws these
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
