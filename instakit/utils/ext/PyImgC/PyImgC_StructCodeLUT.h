#ifndef PyImgC_STRUCTCODELUT_H
#define PyImgC_STRUCTCODELUT_H

#include <map>

#include <Python.h>
#include <numpy/ndarrayobject.h>
using namespace std;

namespace structcode {

    template <typename T>
    struct txbase { typedef T type; };

    struct tx_npy_bool : txbase<npy_bool> {
        static txbase *create() { return new tx_npy_bool; }
    };

    struct tx_npy_byte : txbase<npy_byte> {
        static txbase *create() { return new tx_npy_byte; }
    };

    struct tx_npy_ubyte : txbase<npy_ubyte> {
        static txbase *create() { return new tx_npy_ubyte; }
    };

    struct tx_npy_short : txbase<npy_short> {
        static txbase *create() { return new tx_npy_short; }
    };

    struct tx_npy_int16 : txbase<npy_int16> {
        static txbase *create() { return new tx_npy_int16; }
    };

    map<int, txbase<void *> * (*)()> typemap;

    typemap[NPY_BOOL] = &tx_npy_bool::create;
    typemap[NPY_BYTE] = &tx_npy_byte::create;
    typemap[NPY_UBYTE] = &tx_npy_ubyte::create;
    typemap[NPY_SHORT] = &tx_npy_short::create;
    typemap[NPY_INT16] = &tx_npy_int16::create;

    txbase<void *> map_type(npy_intp npy_typenum) {
        return typemap[npy_typenum]()::type;
    }


    inline int struct_to_typecode(char letter, int native=1, int complex=0) {
        switch (letter)
        {
        case '?': return NPY_BOOL;
        case 'b': return NPY_BYTE;
        case 'B': return NPY_UBYTE;
        case 'h': return native ? NPY_SHORT : NPY_INT16;
        case 'H': return native ? NPY_USHORT : NPY_UINT16;
        case 'i': return native ? NPY_INT : NPY_INT32;
        case 'I': return native ? NPY_UINT : NPY_UINT32;
        case 'l': return native ? NPY_LONG : NPY_INT32;
        case 'L': return native ? NPY_ULONG : NPY_UINT32;
        case 'q': return native ? NPY_LONGLONG : NPY_INT64;
        case 'Q': return native ? NPY_ULONGLONG : NPY_UINT64;
        case 'e': return NPY_HALF;
        case 'f': return complex ? NPY_CFLOAT : NPY_FLOAT;
        case 'd': return complex ? NPY_CDOUBLE : NPY_DOUBLE;
        case 'g': return native ? (complex ? NPY_CLONGDOUBLE : NPY_LONGDOUBLE) : -1;
        default:
            /* Other unhandled cases */
            return -1;
        }
        return -1;
    }

}



#endif