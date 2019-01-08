#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function

import collections.abc


class SimpleNamespace(object):
    
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


class Namespace(SimpleNamespace, collections.abc.MutableMapping,
                                 collections.abc.Sized,
                                 collections.abc.Iterable,
                                 collections.abc.Container):
    
    """ A less-simple namespace -- one implementing several useful
        interface APIs from `collections.abc`.
    """
    __slots__ = tuple()
    
    def __bool__(self):
        return bool(self.__dict__)
    
    def __len__(self):
        return len(self.__dict__)
    
    def __contains__(self, key):
        return key in self.__dict__
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, val):
        self.__dict__[key] = val
    
    def __delitem__(self, key):
        del self.__dict__[key]
    
    def get(self, key, default_value=None):
        return self.__dict__.get(key, default_value)
    
    def pop(self, key, default_value=None):
        return self.__dict__.pop(key, default_value)
    
    def update(self, iterable=None, **kwargs):
        self.__dict__.update(iterable or tuple(), **kwargs)
    
    def __iter__(self):
        return iter(self.__dict__)
    
    def __copy__(self):
        return type(self)(**self.__dict__)
    
    def __eq__(self, other):
        return self.__dict__ == getattr(other, '__dict__', {})
