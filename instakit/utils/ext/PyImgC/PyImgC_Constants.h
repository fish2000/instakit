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
inline char struct_typecode();

#define DECLARE_STRUCT_TYPECODE(type, constant) \
    template <> inline \
    char struct_typecode<type>() { return constant; } \
    \
    template <> inline \
    char struct_typecode<const type>() { return constant; } \
    \
    template <> inline \
    char struct_typecode<volatile type>() { return constant; } \
    \
    template <> inline \
    char struct_typecode<volatile const type>() { return constant; }

DECLARE_STRUCT_TYPECODE(bool, "?")
DECLARE_STRUCT_TYPECODE(float, "d")
DECLARE_STRUCT_TYPECODE(double, "g")

DECLARE_STRUCT_TYPECODE(char, "b")
DECLARE_STRUCT_TYPECODE(short, "h")
DECLARE_STRUCT_TYPECODE(int, "i")
DECLARE_STRUCT_TYPECODE(long, "l")
DECLARE_STRUCT_TYPECODE(long long, "q")

DECLARE_STRUCT_TYPECODE(unsigned char, "B")
DECLARE_STRUCT_TYPECODE(unsigned short, "H")
DECLARE_STRUCT_TYPECODE(unsigned int, "I")
DECLARE_STRUCT_TYPECODE(unsigned long, "L")
DECLARE_STRUCT_TYPECODE(unsigned long long, "Q")

DECLARE_STRUCT_TYPECODE(char *, "s")
DECLARE_STRUCT_TYPECODE(char *, "p")
DECLARE_STRUCT_TYPECODE(void *, "P")
/*  DECLARE_STRUCT_TYPECODE(std::complex<float>, NPY_CFLOAT)
    DECLARE_STRUCT_TYPECODE(std::complex<double>, NPY_CDOUBLE) */

int get_ipl_bit_depth() const {
  // to do : handle IPL_DEPTH_1U?
  if (typeid(T) == typeid(unsigned char))  return IPL_DEPTH_8U;
  if (typeid(T) == typeid(char))           return IPL_DEPTH_8S;
  if (typeid(T) == typeid(unsigned short)) return IPL_DEPTH_16U;
  if (typeid(T) == typeid(short))          return IPL_DEPTH_16S;
  if (typeid(T) == typeid(int))            return IPL_DEPTH_32S;
  if (typeid(T) == typeid(float))          return IPL_DEPTH_32F;
  if (typeid(T) == typeid(double))         return IPL_DEPTH_64F;
  return 0;
}



/// END OF PYTHON C-API STUFF'S LIMINAL HASH-IFNDEF
#endif