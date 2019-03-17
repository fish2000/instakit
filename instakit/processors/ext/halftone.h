
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#ifndef atkinson_add_error
#define atkinson_add_error(b, e) ( ((b + e) <= 0x00) ? 0x00 : ( (( b + e) >= 0xFF) ? 0xFF : (b + e) ) )
#endif
