#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    INSTAKIT -- Instagrammy PIL-based processors and tools
#
#    Copyright © 2012 Alexander Bohn
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

name = 'instakit'
version = '0.1.6'
description = 'Image processors and filters.'
keywords = 'python django imagekit image processing filters'

classifiers = [
    'Development Status :: 5 - Production/Stable']

try:
    from setuptools import setup, find_packages

except ImportError:
    from distutils.core import setup

    def is_package(path):
        return (os.path.isdir(path) and \
            os.path.isfile(
                os.path.join(
                    path, '__init__.py')))
    
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
                    find_packages(
                        pth, module_name))
        return packages

if 'sdist' in sys.argv and 'upload' in sys.argv:
    import commands
    finder = "/usr/bin/find %s \( -name \*.pyc -or -name .DS_Store \) -delete"
    theplace = os.getcwd()
    if theplace not in (".", "/"):
        print("+ Deleting crapola from %s..." % theplace)
        print("$ %s" % finder % theplace)
        commands.getstatusoutput(finder % theplace)
        print("")

setup(
    name=name, version=version, description=description,
    keywords=keywords, platforms=['any'],
    
    author=u"Alexander Bohn", author_email='fish2000@gmail.com',
    
    license='MIT',
    url='http://github.com/fish2000/%s/' % name,
    download_url='http://github.com/fish2000/%s/zipball/master' % name,
    
    packages=find_packages(),
    package_data={'': ['*.*']},
    include_package_data=True,
    zip_safe=False,
    
    install_requires=[
        'numpy',
        'scipy',
        'imread',
        'Pillow'],
    
    
    classifiers=classifiers+[
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: OS Independent',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
)
