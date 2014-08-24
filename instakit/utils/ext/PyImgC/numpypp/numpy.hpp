/* Copyright 2010-2012 (C)
 * Luis Pedro Coelho <luis@luispedro.org>
 * License: MIT
 * Annotated and rearranged by FI$H 2000
 */

#ifndef NUMPYPP_NUMPY_HPP_
#define NUMPYPP_NUMPY_HPP_

#include <complex>
#include <Python.h>
#include <numpy/ndarrayobject.h>

namespace numpy {

    /// Forward declaration of `dtype_code()` master template, enabling the large-scale,
    /// full-on, macro-driven polymorphic template fiesta that immediately follows --
    /// facilitating our (comparatively gloriously) simple use of numpy's dtype system.
    ///     -fish
    template <typename T>
    inline npy_intp dtype_code();
    
    template <typename T>
    inline PyArray_Descr *dtype_struct();

    template<int>
    struct decoder;

    /// Meta-macro, templating all essential storage-class declaration permutations
    /// one needs when furnishing an API with the kind of clean, functional syntax
    /// one needs to not go totally fucking insane while managing dtype struct internals,
    /// here in post-apocalyptic c++-istan. THEYRE NOT BY ME LUIS WROTE EM, FUCK YES
    ///     -fish
    
    /// NOTA BENE: use the decoder like so:
    ///     CImg<decoder<NPY_UBYTE>::type> image(...);
    
    #define DECLARE_DTYPE_CODE(ctype, constant) \
        template <> inline \
        npy_intp dtype_code<ctype>() { return constant; } \
        \
        template <> inline \
        npy_intp dtype_code<const ctype>() { return constant; } \
        \
        template <> inline \
        npy_intp dtype_code<volatile ctype>() { return constant; } \
        \
        template <> inline \
        npy_intp dtype_code<volatile const ctype>() { return constant; } \
        \
        template <> inline \
        PyArray_Descr *dtype_struct<ctype>() { return PyArray_DescrFromType(constant); } \
        \
        template <> inline \
        PyArray_Descr *dtype_struct<const ctype>() { return PyArray_DescrFromType(constant); } \
        \
        template <> inline \
        PyArray_Descr *dtype_struct<volatile ctype>() { return PyArray_DescrFromType(constant); } \
        \
        template <> inline \
        PyArray_Descr *dtype_struct<volatile const ctype>() { return PyArray_DescrFromType(constant); } \
        \
        template <> \
        struct decoder<constant> { typedef ctype type; };
    
    /// Piping each of numpy's core dtype codes into the DECLARE_DTYPE_CODE()
    /// template macro needs only to happen once -- this handles all four possible
    /// declarative permutations that may festoon the underlying intrinsic type;
    /// it’s also, like, totally super legible and pretty, rite? Rite.
    ///     -fish
    DECLARE_DTYPE_CODE(bool, NPY_BOOL)
    DECLARE_DTYPE_CODE(float, NPY_FLOAT)
    DECLARE_DTYPE_CODE(char, NPY_BYTE)
    DECLARE_DTYPE_CODE(unsigned char, NPY_UBYTE)
    DECLARE_DTYPE_CODE(short, NPY_SHORT)
    DECLARE_DTYPE_CODE(unsigned short, NPY_USHORT)
    DECLARE_DTYPE_CODE(int, NPY_INT)
    DECLARE_DTYPE_CODE(long, NPY_LONG)
    DECLARE_DTYPE_CODE(unsigned long, NPY_ULONG)
    DECLARE_DTYPE_CODE(long long, NPY_LONGLONG)
    DECLARE_DTYPE_CODE(unsigned long long, NPY_ULONGLONG)
    DECLARE_DTYPE_CODE(double, NPY_DOUBLE)
    DECLARE_DTYPE_CODE(std::complex<float>, NPY_CFLOAT)
    DECLARE_DTYPE_CODE(std::complex<double>, NPY_CDOUBLE)
    DECLARE_DTYPE_CODE(unsigned int, NPY_UINT)

    /// N.B. I am so grateful for Mr. Pedro for writing
    /// all of these handy little numpy/c++ helpers;
    /// I would have sucked at trying to do most of it,
    /// especially these polymorphic dtype-checker funcs.
    ///     -fish
    template<typename T>
    bool check_type(PyArrayObject* array) {
        return PyArray_EquivTypenums(
            PyArray_TYPE(array),
            dtype_code<T>());
    }
    template<typename T>
    bool check_type(PyObject* array) {
        return check_type<T>(
            reinterpret_cast<PyArrayObject*>(array));
    }

    /// `no_ptr` polymorphic specialization, in support of:
    ///     * pointers,
    ///     * things that aren't pointers (obvi)
    ///     * and const pointers
    /// ... which that is plenty comprehensive, cuz FUCK VOLATILES
    ///     -fish
    template<typename T>
    struct no_ptr { typedef T type; };
    template<typename T>
    struct no_ptr<T*> { typedef T type; };
    template<typename T>
    struct no_ptr<const T*> { typedef T type; };

    template<typename T>
    T ndarray_cast(PyArrayObject* array) {
        assert(check_type<typename no_ptr<T>::type>(array));
        assert(PyArray_ISALIGNED(array));
        // The reason for the intermediate ``as_voidp`` variable is the following:
        // around version 1.7.0, numpy played with having ``PyArray_DATA`` return
        // ``char*`` as opposed to ``void*``. ``reinterpret_cast`` from void* was
        // actually against the standard in C++ pre C++11 (although G++ accepts
        // it).
        //
        // This roundabout way works on all these versions.
        void* as_voidp = PyArray_DATA(array);
        return const_cast<T>(static_cast<T>(as_voidp));
    }
    
    template<typename T>
    T ndarray_cast(PyObject* pyarray) {
        assert(PyArray_Check(pyarray));
        return ndarray_cast<T>((PyArrayObject*)pyarray);
    }

} /// namespace numpy
#endif // NUMPYPP_NUMPY_HPP_
