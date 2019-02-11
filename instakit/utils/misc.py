#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function

import collections.abc
from functools import wraps

try:
    import six
except ImportError:
    class FakeSix(object):
        @property
        def string_types(self):
            return tuple()
    six = FakeSix()

UTF8_ENCODING = 'UTF-8'

def tuplize(*items):
    """ Return a new tuple containing all non-`None` arguments """
    return tuple(item for item in items if item is not None)

def uniquify(*items):
    """ Return a tuple with a unique set of all non-`None` arguments """
    return tuple(frozenset(item for item in items if item is not None))

def listify(*items):
    """ Return a new list containing all non-`None` arguments """
    return list(item for item in items if item is not None)


class SimpleNamespace(object):
    
    """ Implementation courtesy this SO answer:
        • https://stackoverflow.com/a/37161391/298171
    """
    __slots__ = tuplize('__dict__', '__weakref__')
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class Namespace(SimpleNamespace, collections.abc.MutableMapping,
                                 collections.abc.Hashable):
    
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
    
    def __hash__(self):
        return hash(tuple(self.__dict__.keys()) +
                    tuple(self.__dict__.values()))
    
    def __iter__(self):
        return iter(self.__dict__)
    
    def __copy__(self):
        return type(self)(**self.__dict__)
    
    def __eq__(self, other):
        return self.__dict__ == getattr(other, '__dict__', {})

def wrap_value(value):
    return lambda *args, **kwargs: value

none_function = wrap_value(None)

string_types = uniquify(type(''),
                        type(b''),
                        type(r''),
                       *six.string_types)

class Memoizer(dict):
    
    """ Very simple memoizer (only works with positional args) """
    
    def __init__(self, function):
        super(Memoizer, self).__init__()
        self.original = function
    
    def __missing__(self, key):
        function = self.original
        self[key] = out = function(*key)
        return out
    
    @property
    def original(self):
        return self.original_function
    
    @original.setter
    def original(self, value):
        if not value:
            self.original_function = none_function
        elif not callable(value):
            self.original_function = wrap_value(value)
        else:
            self.original_function = value
    
    @property
    def __wrapped__(self):
        return self.original_function
    
    def __call__(self, function=None):
        if function is None:
            function = self.original
        else:
            self.original = function
        @wraps(function)
        def memoized(*args):
            return self[tuplize(*args)]
        memoized.__wrapped__ = function
        memoized.__instance__ = self
        return memoized


def memoize(function):
    memoinstance = Memoizer(function)
    @wraps(function)
    def memoized(*args):
        return memoinstance[tuplize(*args)]
    memoized.__wrapped__ = function
    memoized.__instance__ = memoinstance
    return memoized

def stringify(instance, fields):
    """ Stringify an object instance, using an iterable field list to
        extract and render its values, and printing them along with the 
        typename of the instance and its memory address -- yielding a
        repr-style string of the format:
        
            TypeName(fieldname="val", otherfieldname="otherval") @ 0x0FE
        
        The `stringify(…)` function is of use in `__str__()` and `__repr__()`
        definitions, E.G. something like:
        
            def __repr__(self):
                return stringify(self, type(self).__slots__)
        
    """
    field_dict = {}
    for field in fields:
        field_value = getattr(instance, field, "")
        field_value = callable(field_value) and field_value() or field_value
        if field_value:
            field_dict.update({ u8str(field) : field_value })
    field_dict_items = []
    for k, v in field_dict.items():
        field_dict_items.append('''%s="%s"''' % (k, v))
    typename = type(instance).__name__
    field_dict_string = ", ".join(field_dict_items)
    hex_id = hex(id(instance))
    return "%s(%s) @ %s" % (typename, field_dict_string, hex_id)

def suffix_searcher(suffix):
    """ Return a boolean function that will search for the given
        file suffix in strings with which it is called, returning
        True when they are found and False when they aren’t.
        
        Useful in filter(…) calls and comprehensions, e.g.:
        
        >>> plists = filter(suffix_searcher('plist'), os.listdir())
        >>> mmsuffix = suffix_searcher('mm')
        >>> objcpp = (f for f in os.listdir() where mmsuffix(f))
    """
    import re, os
    if len(suffix) < 1:
        return lambda searching_for: True
    regex_str = r""
    if suffix.startswith(os.extsep):
        regex_str += r"\%s$" % suffix
    else:
        regex_str += r"\%s%s$" % (os.extsep, suffix)
    searcher = re.compile(regex_str, re.IGNORECASE).search
    return lambda searching_for: bool(searcher(searching_for))

def u8encode(source):
    """ Encode a source as bytes using the UTF-8 codec """
    return bytes(source, encoding=UTF8_ENCODING)

def u8bytes(source):
    """ Encode a source as bytes using the UTF-8 codec, guaranteeing
        a proper return value without raising an error
    """
    if type(source) is bytes:
        return source
    elif type(source) is bytearray:
        return bytes(source)
    elif type(source) is str:
        return u8encode(source)
    elif isinstance(source, string_types):
        return u8encode(source)
    elif isinstance(source, (int, float)):
        return u8encode(str(source))
    elif type(source) is bool:
        return source and b'True' or b'False'
    elif source is None:
        return b'None'
    return bytes(source)

def u8str(source):
    """ Encode a source as a Python string, guaranteeing a proper return
        value without raising an error
    """
    return type(source) is str and source \
                        or u8bytes(source).decode(UTF8_ENCODING)
