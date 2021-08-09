/* -*- c++ -*- */
/*
 * Copyright 2010, 2011, 2015, 2018, 2020, 2021 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_VOLK_COMPLEX_H
#define INCLUDED_VOLK_COMPLEX_H

/*!
 * \brief Provide typedefs and operators for all complex types in C and C++.
 *
 * The typedefs encompass all signed integer and floating point types.
 * Each operator function is intended to work across all data types.
 * Under C++, these operators are defined as inline templates.
 * Under C, these operators are defined as preprocessor macros.
 * The use of macros makes the operators agnostic to the type.
 *
 * The following operator functions are defined:
 * - lv_cmake - make a complex type from components
 * - lv_creal - get the real part of the complex number
 * - lv_cimag - get the imaginary part of the complex number
 * - lv_conj - take the conjugate of the complex number
 */

#include <volk/volk_common.h>

__VOLK_DECL_BEGIN

#include <complex.h>

// Obviously, we would love `typedef float complex lv_32fc_t` to work.
// However, this clashes with C++ definitions.
// error: expected initializer before ‘lv_32fc_t’
//    --> typedef float complex lv_32fc_t;
// https://stackoverflow.com/a/10540302

typedef char _Complex lv_8sc_t;
typedef short _Complex lv_16sc_t;
typedef long _Complex lv_32sc_t;
typedef long long _Complex lv_64sc_t;
typedef float _Complex lv_32fc_t;
typedef double _Complex lv_64fc_t;

#define lv_cmake(r, i) ((r) + _Complex_I * (i))
// We want `_Imaginary_I` to ensure the correct sign.
// https://en.cppreference.com/w/c/numeric/complex/Imaginary_I
// It does not compile. Complex numbers are a terribly implemented afterthought.
// #define lv_cmake(r, i) ((r) + _Imaginary_I * (i))

// When GNUC is available, use the complex extensions.
// The extensions always return the correct value type.
// http://gcc.gnu.org/onlinedocs/gcc/Complex.html
#ifdef __GNUC__

#define lv_creal(x) (__real__(x))

#define lv_cimag(x) (__imag__(x))

#define lv_conj(x) (~(x))

// When not available, use the c99 complex function family,
// which always returns double regardless of the input type,
// unless we have C99 and thus tgmath.h overriding functions
// with type-generic versions.
#else /* __GNUC__ */


#define lv_creal(x) (creal(x))

#define lv_cimag(x) (cimag(x))

#define lv_conj(x) (conj(x))

#endif /* __GNUC__ */

__VOLK_DECL_END

#endif /* INCLUDE_VOLK_COMPLEX_H */
