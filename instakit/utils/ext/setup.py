from distutils.sysconfig import get_python_inc
from distutils.core import setup, Extension

try:
    import numpy
except ImportError:
    class FakeNumpy(object):
        def get_include(self):
            return "."
    numpy = FakeNumpy()

setup(
    name="PyImgC",
    version="0.1.0",
    ext_modules=[
        Extension("PyImgC",
            ["PyImgC/pyimgc.cpp"])],
    include_dirs=[
        numpy.get_include(),
        get_python_inc(plat_specific=1)],
    )