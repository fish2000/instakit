#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os

from instakit.exporting import Exporter

exporter = Exporter(path=__file__)
export = exporter.decorator()

projectdir = os.path.join(os.path.dirname(__file__), '..', '..')
namespaces = set()

@export
def static_namespace(name):
    """ Configure and return a clu.typespace.namespace.Namespace instance,
        festooning it with shortcuts allowing for accesing static files
        within subdirectories of the Instakit project package tree.
    """
    from clu.typespace.namespace import SimpleNamespace as Namespace
    ns = Namespace()
    ns.name = str(name)
    ns.root = os.path.join(projectdir, ns.name)
    ns.data = os.path.join(ns.root, 'data')
    ns.relative = lambda p: os.path.relpath(p, start=ns.root)
    ns.listfiles = lambda *p: os.listdir(os.path.join(ns.data, *p))
    ns.path = lambda *p: os.path.abspath(os.path.join(ns.data, *p))
    namespaces.add(ns)
    return ns

asset = static_namespace('instakit')
tests = static_namespace('tests')

export(projectdir,      name='projectdir')
export(namespaces,      name='namespaces')
export(asset,           name='asset',       doc="asset → static namespace relative to the Instakit package assets")
export(tests,           name='tests',       doc="tests → static namespace relative to the Instakit testing assets")

# Assign the modules’ `__all__` and `__dir__` using the exporter:
__all__, __dir__ = exporter.all_and_dir()

def test():
    assert os.path.isdir(projectdir)
    assert len(namespaces) == 2
    
    assert os.path.isdir(asset.root)
    assert os.path.isdir(asset.data)
    assert len(asset.listfiles('acv')) > 0
    assert len(asset.listfiles('icc')) > 0
    assert len(asset.listfiles('img')) > 0
    assert len(asset.listfiles('lut')) > 0
    
    assert os.path.isdir(tests.root)
    assert os.path.isdir(tests.data)
    assert len(os.listdir(tests.data)) > 0

if __name__ == '__main__':
    test()
