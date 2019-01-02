#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function

class Namespace(object):
    
    """ Implementation courtesy this SO answer:
        • https://stackoverflow.com/a/37161391/298171
    """
    __slots__ = ('__dict__', '__weakref__')
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

# Backwards-compatibility:
SimpleNamespace = Namespace