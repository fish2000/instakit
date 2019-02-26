#!/usr/bin/env python
# -*- coding: utf-8 -*-
#    
#    INSTAKIT -- Instagrammy image-processors and tools, based on Pillow and SciPy
#    
#    Copyright © 2012-2025 Alexander Böhn
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

import re

from collections import OrderedDict, namedtuple as NamedTuple
from functools import wraps
from pkgutil import extend_path

if '__path__' in locals():
    __path__ = extend_path(__path__, __name__)

if not hasattr(__builtins__, 'cmp'):
    def cmp(a, b):
        return (a > b) - (a < b)

# Embedded project metadata:
__version__ = "‽.‽.‽"
__title__ = 'instakit'
__author__ = 'Alexander Böhn'
__maintainer__ = 'Alexander Böhn'
__license__ = 'MIT'
__copyright__ = '© 2012-2025 Alexander Böhn'

# get the project version tag without importing:
try:
    exec(compile(open('__version__.py')).read(),
                      '__version__.py', 'exec')
except:
    __version__ = '0.6.10'

# module exports:
__all__ = ('__version__', 'version',
           '__title__', '__author__', '__maintainer__',
           '__license__',
           '__copyright__',
           '__path__', 'FIELDS', 'VersionInfo')

__dir__ = lambda: list(__all__)

FIELDS = ('major', 'minor', 'patch',
          'pre',   'build')

# The `namedtuple` ancestor,
# from which our VersionInfo struct inherits:
VersionAncestor = NamedTuple('VersionAncestor',
                              FIELDS,
                              defaults=('', 0))

# sets, for various comparisons and checks:
fields = frozenset(FIELDS)
string_types = { type(lit) for lit in ('', u'', r'') }
byte_types = { bytes, bytearray } - string_types # On py2, bytes == str
dict_types = { dict, OrderedDict }
comparable = dict_types | { VersionAncestor }

# utility conversion functions:
def intify(arg):
    if arg is None:
        return None
    return int(arg)

def strify(arg):
    if arg is None:
        return None
    if type(arg) in string_types:
        return arg
    if type(arg) in byte_types:
        return arg.decode('UTF-8')
    return str(arg)

def dictify(arg):
    if arg is None:
        return None
    if hasattr(arg, '_asdict'):
        return arg._asdict()
    if hasattr(arg, 'to_dict'):
        return arg.to_dict()
    if type(arg) in dict_types:
        return arg
    return dict(arg)

# compare version information by dicts:
def compare_keys(dict1, dict2):
    """ Blatantly based on code from “semver”: https://git.io/fhb98 """
    
    for key in ('major', 'minor', 'patch'):
        result = cmp(dict1.get(key), dict2.get(key))
        if result:
            return result
    
    pre1, pre2 = dict1.get('pre'), dict2.get('pre')
    if pre1 is None and pre2 is None:
        return 0
    if pre1 is None:
        pre1 = '<unknown>'
    elif pre2 is None:
        pre2 = '<unknown>'
    preresult = cmp(pre1, pre2)
    
    if not preresult:
        return 0
    if not pre1:
        return 1
    elif not pre2:
        return -1
    return preresult

# comparison-operator method decorator:
def comparator(operator):
    """ Wrap a VersionInfo binary op method in a typechecker """
    @wraps(operator)
    def wrapper(self, other):
        if not isinstance(other, tuple(comparable)):
            return NotImplemented
        return operator(self, other)
    return wrapper

# the VersionInfo class:
class VersionInfo(VersionAncestor):
    
    """ NamedTuple-descendant class allowing for convenient
        and reasonably sane manipulation of semantic-version
        (née “semver”) string-triple numberings, or whatever
        the fuck is the technical term for them, erm. Yes!
    """
    
    SEPARATORS = '..-+'
    UNKNOWN = '‽'
    NULL_VERSION = "%s.%s.%s" % ((UNKNOWN,) * 3)
    REG = re.compile('(?P<major>[\d‽]+)\.'     \
                     '(?P<minor>[\d‽]+)'       \
                     '(?:\.(?P<patch>[\d‽]+)'  \
                     '(?:\-(?P<pre>[\w‽]+)'    \
                     '(?:\+(?P<build>[\d‽]+))?)?)?')
    
    @classmethod
    def from_string(cls, version_string):
        """ Instantiate a VersionInfo with a semver string """
        result = cls.REG.search(version_string)
        if result:
            return cls.from_dict(result.groupdict())
        return cls.from_dict({ field : cls.UNKNOWN for field in FIELDS })
    
    @classmethod
    def from_dict(cls, version_dict):
        """ Instantiate a VersionInfo with a dict of related values
            (q.v. FIELD string names supra.)
        """
        assert 'major' in version_dict
        assert 'minor' in version_dict
        assert frozenset(version_dict.keys()).issubset(fields)
        return cls(**version_dict)
    
    def to_string(self):
        """ Return the VersionInfo data as a semver string """
        if not bool(self):
            return type(self).NULL_VERSION
        SEPARATORS = type(self).SEPARATORS
        out = "%i%s%i" % (self.major or 0, SEPARATORS[0],
                          self.minor or 0)
        if self.patch is not None:
            out += "%s%i" % (SEPARATORS[1], self.patch)
            if self.pre:
                out += "%s%s" % (SEPARATORS[2], self.pre)
                if self.build:
                    out += "%s%i" % (SEPARATORS[3], self.build)
        return out
    
    def to_dict(self):
        out = OrderedDict()
        for field in FIELDS:
            if getattr(self, field, None) is not None:
                out[field] = getattr(self, field)
        return out
    
    def to_tuple(self):
        return (self.major, self.minor, self.patch,
                self.pre, self.build)
    
    def __new__(cls, from_value=None, major='‽', minor='‽',
                                      patch='‽', pre='‽',
                                      build=0):
        """ Instantiate a VersionInfo, populating its fields per args """
        if from_value is not None:
            if type(from_value) in string_types:
                return cls.from_string(from_value)
            elif type(from_value) in byte_types:
                return cls.from_string(from_value.decode('UTF-8'))
            elif type(from_value) in dict_types:
                return cls.from_dict(from_value)
            elif type(from_value) is cls:
                return cls.from_dict(from_value.to_dict())
        if cls.UNKNOWN in str(major):
            major = None
        if cls.UNKNOWN in str(minor):
            minor = None
        if cls.UNKNOWN in str(patch):
            patch = None
        if cls.UNKNOWN in str(pre):
            pre = None
        if cls.UNKNOWN in str(build):
            build = 0
        instance = super(VersionInfo, cls).__new__(cls, intify(major),
                                                        intify(minor),
                                                        intify(patch),
                                                        strify(pre),
                                                               build)
        return instance
    
    def __str__(self):
        """ Stringify the VersionInfo (q.v. “to_string(…)” supra.) """
        return self.to_string()
    
    def __bytes__(self):
        """ Bytes-ify the VersionInfo (q.v. “to_string(…)” supra.) """
        return bytes(self.to_string())
    
    def __hash__(self):
        """ Hash the VersionInfo, using its tuplized value """
        return hash(self.to_tuple())
    
    def __bool__(self):
        """ An instance of VersionInfo is considered Falsey if its “major”,
           “minor”, and “patch” fields are all set to None; otherwise it’s
            a Truthy value in boolean contexts
        """
        return not (self.major is None and \
                    self.minor is None and \
                    self.patch is None)
    
    # Comparison methods also lifted from “semver”: https://git.io/fhb9i
    
    @comparator
    def __eq__(self, other):
        return compare_keys(self._asdict(), dictify(other)) == 0
    
    @comparator
    def __ne__(self, other):
        return compare_keys(self._asdict(), dictify(other)) != 0
    
    @comparator
    def __lt__(self, other):
        return compare_keys(self._asdict(), dictify(other)) < 0
    
    @comparator
    def __le__(self, other):
        return compare_keys(self._asdict(), dictify(other)) <= 0
    
    @comparator
    def __gt__(self, other):
        return compare_keys(self._asdict(), dictify(other)) > 0
    
    @comparator
    def __ge__(self, other):
        return compare_keys(self._asdict(), dictify(other)) >= 0

# the InstaKit project version:
version = VersionInfo(__version__)

# inline tests:
def test():
    # print(VersionInfo.REG)
    print("VersionInfo Instance:", repr(version))
    print("Semantic Version:", version)
    
    assert version  < VersionInfo("9.0.0")
    assert version == VersionInfo(version)
    assert version == VersionInfo(__version__)
    assert version <= VersionInfo(__version__)
    assert version >= VersionInfo(__version__)
    assert version  > VersionInfo(b'0.1.0')
    assert version != VersionInfo(b'0.1.0')
    
    assert bool(version)
    assert not bool(VersionInfo('‽.‽.‽'))

if __name__ == '__main__':
    test()