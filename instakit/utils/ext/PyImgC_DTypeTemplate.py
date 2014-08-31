#!/usr/bin/env python

import xerox

# //const npy_intp typecode = NPY_%(npytype)s;

structs = u'''
struct CImage_NPY_%(npytype)s : public CImage_Type<%(ctype)s> {
    const char structcode[%(structcodelen)s] = { '%(structcode)s', NILCODE };
    const unsigned int structcode_length = %(structcodelen)s;
    const bool native = %(native)s;
    const bool complex = %(complicated)s;
    CImage_NPY_%(npytype)s() {}
    CImage_Functor<NPY_%(npytype)s, %(ctype)s> reg();
};
'''

functors = u'''
template <>
struct CImage_Functor<NPY_%(npytype)s, %(ctype)s> : public CImage_FunctorType {
    CImage_Functor<NPY_%(npytype)s, %(ctype)s>(int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_%(npytype)s, %(ctype)s>));
    }
};
'''

_registration = u'''
CImage_Functor<NPY_%(npytype)s, %(ctype)s> CImage_NPY_%(npytype)s::reg(NPY_%(npytype)s);'''
registration = u""

# TRAILING TUPLE: (native, complex)

types = [
    ('BOOL', 'bool', ('?',), (True, False)),
    ('BYTE', 'char', ('b',), (True, False)),
    ('HALF', 'npy_half', ('e',), (False, False)),
    
    ('SHORT', 'short', ('h',), (True, False)),
    ('INT', 'int', ('i',), (True, False)),
    ('LONG', 'long', ('l',), (True, False)),
    ('LONGLONG', 'long long', ('q',), (True, False)),
    ('UBYTE', 'unsigned char', ('B',), (True, False)),
    ('USHORT', 'unsigned short', ('H',), (True, False)),
    ('UINT', 'unsigned int', ('I',), (True, False)),
    ('ULONG', 'unsigned long', ('L',), (True, False)),
    ('ULONGLONG', 'unsigned long long', ('Q',), (True, False)),

    ('CFLOAT', 'std::complex<float>', ('f',), (False, True)),
    ('CDOUBLE', 'std::complex<double>', ('d',), (False, True)),
    ('FLOAT', 'float', ('f',), (False, False)),
    ('DOUBLE', 'double', ('d',), (False, False)),
    ('CLONGDOUBLE', 'std::complex<long double>', ('g',), (True, False)),
    ('LONGDOUBLE', 'std::complex<long double>', ('g',), (True, True)),
]


def render_template(template, types):
    out = u""
    for typedef in types:
        npytype, ctype, structcode, flagtuple = typedef
        native, complicated = flagtuple
        out += template % dict(
            npytype=npytype, ctype=ctype,
            structcode=u"', '".join(structcode),
            structcodelen=len(structcode)+1,
            native=str(native).lower(),
            complicated=str(complicated).lower())
    return out

out = \
    render_template(structs, types) \
  + render_template(functors, types) \
  + render_template(registration, types)

xerox.copy(out)
print out
