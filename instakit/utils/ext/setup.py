from distutils.core import setup, Extension
setup(
    name="PyImgC",
    version="0.1.0",
    ext_modules=[
        Extension("PyImgC",
            ["PyImgC/pyimgc.cpp"]),
        ])