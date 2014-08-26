#ifndef PyImgC_TYPECODE_H
#define PyImgC_TYPECODE_H

#ifndef IMGC_DEBUG
#define IMGC_DEBUG 0
#endif

#include "../pyimgc.h"
#include <numpy/ndarraytypes.h>

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <map>
using namespace std;

typedef integral_constant<NPY_TYPES, NPY_BOOL> ENUM_NPY_BOOL;
typedef integral_constant<NPY_TYPES, NPY_BYTE> ENUM_NPY_BYTE;
typedef integral_constant<NPY_TYPES, NPY_HALF> ENUM_NPY_HALF;
typedef integral_constant<NPY_TYPES, NPY_SHORT> ENUM_NPY_SHORT;
typedef integral_constant<NPY_TYPES, NPY_INT> ENUM_NPY_INT;
typedef integral_constant<NPY_TYPES, NPY_LONG> ENUM_NPY_LONG;
typedef integral_constant<NPY_TYPES, NPY_LONGLONG> ENUM_NPY_LONGLONG;
typedef integral_constant<NPY_TYPES, NPY_UBYTE> ENUM_NPY_UBYTE;
typedef integral_constant<NPY_TYPES, NPY_USHORT> ENUM_NPY_USHORT;
typedef integral_constant<NPY_TYPES, NPY_UINT> ENUM_NPY_UINT;
typedef integral_constant<NPY_TYPES, NPY_ULONG> ENUM_NPY_ULONG;
typedef integral_constant<NPY_TYPES, NPY_ULONGLONG> ENUM_NPY_ULONGLONG;
typedef integral_constant<NPY_TYPES, NPY_CFLOAT> ENUM_NPY_CFLOAT;
typedef integral_constant<NPY_TYPES, NPY_CDOUBLE> ENUM_NPY_CDOUBLE;
typedef integral_constant<NPY_TYPES, NPY_FLOAT> ENUM_NPY_FLOAT;
typedef integral_constant<NPY_TYPES, NPY_DOUBLE> ENUM_NPY_DOUBLE;
typedef integral_constant<NPY_TYPES, NPY_CLONGDOUBLE> ENUM_NPY_CLONGDOUBLE;
typedef integral_constant<NPY_TYPES, NPY_LONGDOUBLE> ENUM_NPY_LONGDOUBLE;

struct typecodemaps {
    
    static map<int, NPY_TYPES> init_type_enum() {
        map<int, NPY_TYPES> _type_enum = {
            { NPY_BOOL, ENUM_NPY_BOOL::value }, 
            { NPY_BYTE, ENUM_NPY_BYTE::value }, 
            { NPY_HALF, ENUM_NPY_HALF::value }, 
            { NPY_SHORT, ENUM_NPY_SHORT::value }, 
            { NPY_INT, ENUM_NPY_INT::value }, 
            { NPY_LONG, ENUM_NPY_LONG::value }, 
            { NPY_LONGLONG, ENUM_NPY_LONGLONG::value }, 
            { NPY_UBYTE, ENUM_NPY_UBYTE::value }, 
            { NPY_USHORT, ENUM_NPY_USHORT::value }, 
            { NPY_UINT, ENUM_NPY_UINT::value }, 
            { NPY_ULONG, ENUM_NPY_ULONG::value }, 
            { NPY_ULONGLONG, ENUM_NPY_ULONGLONG::value }, 
            { NPY_CFLOAT, ENUM_NPY_CFLOAT::value }, 
            { NPY_CDOUBLE, ENUM_NPY_CDOUBLE::value }, 
            { NPY_FLOAT, ENUM_NPY_FLOAT::value }, 
            { NPY_DOUBLE, ENUM_NPY_DOUBLE::value }, 
            { NPY_CLONGDOUBLE, ENUM_NPY_CLONGDOUBLE::value }, 
            { NPY_LONGDOUBLE, ENUM_NPY_LONGDOUBLE::value }
        };
        return _type_enum;
    }
    
    static const map<int, NPY_TYPES> type_enum;
};

const map<int, NPY_TYPES> typecodemaps::type_enum = typecodemaps::init_type_enum();


#endif