#ifndef Py_STRUCTCODE_API_H
#define Py_STRUCTCODE_API_H
#ifdef __cplusplus
extern "C" {
#endif

/* Header file for structcode API */

/* C API functions */
#define PyImgC_NPYCodeFromStructAtom_NUM 0
#define PyImgC_NPYCodeFromStructAtom_RETURN int
#define PyImgC_NPYCodeFromStructAtom_PROTO (PyObject *self, PyObject *args)

/* Total number of C API pointers */
#define PyImgC_API_pointers 1


#ifdef PyImgC_STRUCTCODE_MODULE
/* This section is used when compiling the module */

static \
    PyImgC_NPYCodeFromStructAtom_RETURN \
    PyImgC_NPYCodeFromStructAtom \
    PyImgC_NPYCodeFromStructAtom_PROTO;

#else
/* This section is used in modules that use the API */

static void **PyImgC_API;

#define PyImgC_NPYCodeFromStructAtom \
    (*(PyImgC_NPYCodeFromStructAtom_RETURN \
    (*)PyImgC_NPYCodeFromStructAtom_PROTO) \
        PyImgC_API[PyImgC_NPYCodeFromStructAtom_NUM])

/* Return -1 on error, 0 on success.
 * PyCapsule_Import will set an exception if there's an error.
 */
static int
PyImgC_import_structcode(void) {
    PyImgC_API = (void **)PyCapsule_Import("_structcode._C_API", 0);
    return (PyImgC_API != NULL) ? 0 : -1;
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* !defined(Py_STRUCTCODE_API_H) */