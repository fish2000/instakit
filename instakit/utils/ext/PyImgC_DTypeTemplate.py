#!/usr/bin/env python

import xerox

template = u'''
typedef struct CImage_NPY_%(npytype)s : public CImage_Base {
    const char structcode[%(structcodelen)s] = { '%(structcode)s', NILCODE };
    const unsigned int structcode_length = %(structcodelen)s;
    const npy_intp typecode = NPY_%(npytype)s;
    const bool native = %(native)s;
    const bool complex = %(complicated)s;

    CImage_NPY_%(npytype)s() {}

    virtual CImg<%(ctype)s> as_pybuffer(Py_buffer *pybuffer, bool is_shared=true) {
        return from_pybuffer<%(ctype)s>(pybuffer, is_shared); }

    virtual CImg<%(ctype)s> as_pybuffer_with_dims(Py_buffer *pybuffer,
        int sW, int sH, int channels=3,
        bool is_shared=true) {
        return from_pybuffer<%(ctype)s>(pybuffer, sW, sH, channels, is_shared); }

    virtual CImg<%(ctype)s> as_pyarray(PyArrayObject *pyarray, bool is_shared=true) {
        return from_pyarray<%(ctype)s>(pyarray, is_shared); }

    virtual CImg<%(ctype)s> as_datasource(PyObject *datasource, bool is_shared=true) {
        return from_pyobject<%(ctype)s>(datasource, is_shared); }

    virtual CImg<%(ctype)s> as_datasource_with_dims(Py_buffer *pybuffer,
        int sW, int sH, int channels=3,
        bool is_shared=true) {
        return from_pybuffer<%(ctype)s>(pybuffer, sW, sH, channels, is_shared); }

    virtual inline bool operator()(const char sc) {
        for (int idx = 0; structcode[idx] != NILCODE; ++idx) {
            if (structcode[idx] == sc) { return true; }
        }
        return false; }
    virtual inline bool operator[](const npy_intp tc) { return tc == typecode; }

} CImage_NPY_%(npytype)s;

REGISTER_TYPESTRUCT(CImage_NPY_%(npytype)s, NPY_%(npytype)s)

'''

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

    ('INT16', 'short', ('h',), (False, False)),
    ('INT32', 'int', ('i', 'l'), (False, False)),
    #('INT32', 'long', ('l',), (False, False)),
    ('INT64', 'long long', ('q',), (False, False)),
    ('UINT16', 'unsigned short', ('H',), (False, False)),
    ('UINT32', 'unsigned int', ('I', 'L'), (False, False)),
    #('UINT32', 'unsigned long', ('L',), (False, False)),
    ('UINT64', 'unsigned long long', ('Q',), (False, False)),

    ('CFLOAT', 'std::complex<float>', ('f',), (False, True)),
    ('CDOUBLE', 'std::complex<double>', ('d',), (False, True)),
    ('FLOAT', 'float', ('f',), (False, False)),
    ('DOUBLE', 'double', ('d',), (False, False)),
    ('CLONGDOUBLE', 'std::complex<long double>', ('g',), (True, False)),
    ('LONGDOUBLE', 'std::complex<long double>', ('g',), (True, True)),
]

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
xerox.copy(out)
