# -*- coding: utf-8 -*-
from __future__ import print_function

import os
from instakit.utils import misc

projectdir = os.path.join(os.path.dirname(__file__), '..', '..')

asset = misc.Namespace()
asset.root = os.path.join(projectdir, 'instakit')
asset.data = os.path.join(asset.root, 'data')
asset.listfiles = lambda *p: os.listdir(os.path.join(asset.data, *p))
asset.path = lambda *p: os.path.abspath(os.path.join(asset.data, *p))

# root = os.path.join(projectdir, 'instakit')
# data = os.path.join(root, 'data')
# listfiles = lambda *pth: os.listdir(os.path.join(data, *pth))
# path = lambda *pth: os.path.abspath(os.path.join(data, *pth))

root = asset.root
data = asset.data
listfiles = asset.listfiles
path = asset.path

tests = misc.Namespace()
tests.root = os.path.join(projectdir, 'tests')
tests.data = os.path.join(tests.root, 'data')
tests.listfiles = lambda *p: os.listdir(os.path.join(tests.data, *p))
tests.path = lambda *p: os.path.abspath(os.path.join(tests.data, *p))

def test():
    assert os.path.isdir(projectdir)
    
    assert os.path.isdir(asset.root)
    assert os.path.isdir(asset.data)
    assert len(asset.listfiles('acv')) > 0
    assert len(asset.listfiles('icc')) > 0
    assert len(asset.listfiles('img')) > 0
    assert len(asset.listfiles('lut')) > 0
    
    assert os.path.isdir(root)
    assert len(listfiles('acv')) > 0
    assert len(listfiles('icc')) > 0
    assert len(listfiles('img')) > 0
    assert len(listfiles('lut')) > 0
    
    assert os.path.isdir(tests.root)
    assert os.path.isdir(tests.data)
    assert len(os.listdir(tests.data)) > 0

if __name__ == '__main__':
    test()