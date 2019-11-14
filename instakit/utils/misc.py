#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function

from clu.fs.misc import suffix_searcher, u8encode, u8bytes, u8str
from clu.predicates import wrap_value, none_function, tuplize, uniquify, listify
from clu.repr import stringify
from clu.typespace.namespace import SimpleNamespace, Namespace
from clu.typology import (string_types, bytes_types as byte_types)
from instakit.exporting import Exporter

exporter = Exporter(path=__file__)
export = exporter.decorator()

try:
    import six
except ImportError:
    class FakeSix(object):
        @property
        def string_types(self):
            return tuple()
    six = FakeSix()

export(wrap_value,              name='wrap_value')
export(none_function,           name='none_function')
export(tuplize,                 name='tuplize')
export(uniquify,                name='uniquify')
export(listify,                 name='listify')

export(SimpleNamespace)
export(Namespace)

export(stringify,               name='stringify')
export(string_types)
export(byte_types)

export(suffix_searcher)
export(u8encode)
export(u8bytes)
export(u8str)

# Assign the modules’ `__all__` and `__dir__` using the exporter:
__all__, __dir__ = exporter.all_and_dir()

