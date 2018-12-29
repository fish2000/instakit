# -*- coding: utf-8 -*-
from __future__ import print_function

import os
from instakit.utils import misc

root = os.path.join(os.path.dirname(__file__), '..', 'data')
listfiles = lambda *pth: os.listdir(os.path.join(root, *pth))
path = lambda *pth: os.path.abspath(os.path.join(root, *pth))

tests = misc.SimpleNamespace()
tests.root = os.path.join(os.path.dirname(__file__), '..', '..', 'tests')
tests.data = os.path.join(tests.root, 'data')
tests.listfiles = lambda *pth: os.listdir(os.path.join(tests.data, *pth))
tests.path = lambda *pth: os.path.abspath(os.path.join(tests.data, *pth))

def test():
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