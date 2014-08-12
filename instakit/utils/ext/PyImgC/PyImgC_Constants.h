#ifndef PyImgC_CONSTANTS_H
#define PyImgC_CONSTANTS_H
/// INSERT PYTHON C-API STUFF HERE

/// lil' bit pythonic
#ifndef False
#define False 0
#endif
#ifndef True
#define True 1
#endif
#ifndef None
#define None NULL
#endif

template <typename T>
inline char typecode();

unordered_map<(const char *), typename> typecodes;

#define DECLARE_TYPECODE(type, constant) \
    template <> inline \
    char typecode<type>() { return constant; } \
    \
    template <> inline \
    char typecode<const type>() { return constant; } \
    \
    template <> inline \
    char typecode<volatile type>() { return constant; } \
    \
    template <> inline \
    char typecode<volatile const type>() { return constant; } \
    \
    typecodes.emplace(constant, type);

DECLARE_TYPECODE(float, "d")
DECLARE_TYPECODE(double, "g")
DECLARE_TYPECODE(bool, "?")
#ifdef _Bool
DECLARE_TYPECODE(_Bool, "?")
#endif

DECLARE_TYPECODE(char, "b")
DECLARE_TYPECODE(short, "h")
DECLARE_TYPECODE(int, "i")
DECLARE_TYPECODE(long, "l")
DECLARE_TYPECODE(long long, "q")

DECLARE_TYPECODE(unsigned char, "B")
DECLARE_TYPECODE(unsigned short, "H")
DECLARE_TYPECODE(unsigned int, "I")
DECLARE_TYPECODE(unsigned long, "L")
DECLARE_TYPECODE(unsigned long long, "Q")

DECLARE_TYPECODE((char *), "s")
DECLARE_TYPECODE((char *), "p")
DECLARE_TYPECODE((void *), "P")

/*
DECLARE_TYPECODE(std::complex<float>, NPY_CFLOAT)
DECLARE_TYPECODE(std::complex<double>, NPY_CDOUBLE)
*/

int image_typecode() const {
    if (typeid(T) == typeid(unsigned char))  return IPL_DEPTH_8U;
    if (typeid(T) == typeid(char))           return IPL_DEPTH_8S;
    if (typeid(T) == typeid(unsigned short)) return IPL_DEPTH_16U;
    if (typeid(T) == typeid(short))          return IPL_DEPTH_16S;
    if (typeid(T) == typeid(int))            return IPL_DEPTH_32S;
    if (typeid(T) == typeid(float))          return IPL_DEPTH_32F;
    if (typeid(T) == typeid(double))         return IPL_DEPTH_64F;
    return False;
}



/// END OF PYTHON C-API STUFF'S LIMINAL HASH-IFNDEF
#endif