/* -*- c++ -*- */
/*
 * Copyright 2010, 2011, 2015-2017, 2019, 2020 Free Software Foundation, Inc.
 * Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef INCLUDED_LIBVOLK_COMMON_H
#define INCLUDED_LIBVOLK_COMMON_H

////////////////////////////////////////////////////////////////////////
// Cross-platform attribute macros
////////////////////////////////////////////////////////////////////////
#if _MSC_VER
#define __VOLK_ATTR_ALIGNED(x) __declspec(align(x))
#define __VOLK_ATTR_UNUSED
#define __VOLK_ATTR_INLINE __forceinline
#define __VOLK_ATTR_DEPRECATED __declspec(deprecated)
#define __VOLK_ATTR_EXPORT __declspec(dllexport)
#define __VOLK_ATTR_IMPORT __declspec(dllimport)
#define __VOLK_PREFETCH(addr)
#define __VOLK_ASM __asm
#elif defined(__clang__)
// AppleClang also defines __GNUC__, so do this check first.  These
// will probably be the same as for __GNUC__, but let's keep them
// separate just to be safe.
#define __VOLK_ATTR_ALIGNED(x) __attribute__((aligned(x)))
#define __VOLK_ATTR_UNUSED __attribute__((unused))
#define __VOLK_ATTR_INLINE __attribute__((always_inline))
#define __VOLK_ATTR_DEPRECATED __attribute__((deprecated))
#define __VOLK_ASM __asm__
#define __VOLK_ATTR_EXPORT __attribute__((visibility("default")))
#define __VOLK_ATTR_IMPORT __attribute__((visibility("default")))
#define __VOLK_PREFETCH(addr) __builtin_prefetch(addr)
#elif defined __GNUC__
#define __VOLK_ATTR_ALIGNED(x) __attribute__((aligned(x)))
#define __VOLK_ATTR_UNUSED __attribute__((unused))
#define __VOLK_ATTR_INLINE __attribute__((always_inline))
#define __VOLK_ATTR_DEPRECATED __attribute__((deprecated))
#define __VOLK_ASM __asm__
#if __GNUC__ >= 4
#define __VOLK_ATTR_EXPORT __attribute__((visibility("default")))
#define __VOLK_ATTR_IMPORT __attribute__((visibility("default")))
#else
#define __VOLK_ATTR_EXPORT
#define __VOLK_ATTR_IMPORT
#endif
#define __VOLK_PREFETCH(addr) __builtin_prefetch(addr)
#elif _MSC_VER
#define __VOLK_ATTR_ALIGNED(x) __declspec(align(x))
#define __VOLK_ATTR_UNUSED
#define __VOLK_ATTR_INLINE __forceinline
#define __VOLK_ATTR_DEPRECATED __declspec(deprecated)
#define __VOLK_ATTR_EXPORT __declspec(dllexport)
#define __VOLK_ATTR_IMPORT __declspec(dllimport)
#define __VOLK_PREFETCH(addr)
#define __VOLK_ASM __asm
#else
#define __VOLK_ATTR_ALIGNED(x)
#define __VOLK_ATTR_UNUSED
#define __VOLK_ATTR_INLINE
#define __VOLK_ATTR_DEPRECATED
#define __VOLK_ATTR_EXPORT
#define __VOLK_ATTR_IMPORT
#define __VOLK_PREFETCH(addr)
#define __VOLK_ASM __asm__
#endif

////////////////////////////////////////////////////////////////////////
// Ignore annoying warnings in MSVC
////////////////////////////////////////////////////////////////////////
#if defined(_MSC_VER)
#pragma warning(disable : 4244) //'conversion' conversion from 'type1' to 'type2',
                                // possible loss of data
#pragma warning(disable : 4305) //'identifier' : truncation from 'type1' to 'type2'
#endif

////////////////////////////////////////////////////////////////////////
// C-linkage declaration macros
// FIXME: due to the usage of complex.h, require gcc for c-linkage
// Hope that extern "C" works for all relevant compilers nowadays.
////////////////////////////////////////////////////////////////////////
#if defined(__cplusplus)
#define __VOLK_DECL_BEGIN extern "C" {
#define __VOLK_DECL_END }
#else
#define __VOLK_DECL_BEGIN
#define __VOLK_DECL_END
#endif

////////////////////////////////////////////////////////////////////////
// Define VOLK_API for library symbols
// https://gcc.gnu.org/wiki/Visibility
////////////////////////////////////////////////////////////////////////
#ifdef volk_EXPORTS
#define VOLK_API __VOLK_ATTR_EXPORT
#else
#define VOLK_API __VOLK_ATTR_IMPORT
#endif

////////////////////////////////////////////////////////////////////////
// The bit128 union used by some
////////////////////////////////////////////////////////////////////////
#include <stdint.h>

#ifdef LV_HAVE_SSE
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

union bit128 {
    uint8_t i8[16];
    uint16_t i16[8];
    uint32_t i[4];
    float f[4];
    double d[2];

#ifdef LV_HAVE_SSE
    __m128 float_vec;
#endif

#ifdef LV_HAVE_SSE2
    __m128i int_vec;
    __m128d double_vec;
#endif
};

union bit256 {
    uint8_t i8[32];
    uint16_t i16[16];
    uint32_t i[8];
    float f[8];
    double d[4];

#ifdef LV_HAVE_AVX
    __m256 float_vec;
    __m256i int_vec;
    __m256d double_vec;
#endif
};

#define bit128_p(x) ((union bit128*)(x))
#define bit256_p(x) ((union bit256*)(x))

////////////////////////////////////////////////////////////////////////
// log2f
////////////////////////////////////////////////////////////////////////
#include <math.h>
// +-Inf -> +-127.0f in order to match the behaviour of the SIMD kernels
static inline float log2f_non_ieee(float f)
{
    float const result = log2f(f);
    return isinf(result) ? copysignf(127.0f, result) : result;
}

////////////////////////////////////////////////////////////////////////
// Constant used to do log10 calculations as faster log2
////////////////////////////////////////////////////////////////////////
// precalculated 10.0 / log2f_non_ieee(10.0) to allow for constexpr
#define volk_log2to10factor (0x1.815182p1) // 3.01029995663981209120

////////////////////////////////////////////////////////////////////////
// arctan(x) polynomial expansion
////////////////////////////////////////////////////////////////////////
static inline float volk_arctan_poly(const float x)
{
    /*
     * arctan(x) polynomial expansion on the interval [-1, 1]
     * Maximum relative error < 6.6e-7
     */
    const float a1 = +0x1.ffffeap-1f;
    const float a3 = -0x1.55437p-2f;
    const float a5 = +0x1.972be6p-3f;
    const float a7 = -0x1.1436ap-3f;
    const float a9 = +0x1.5785aap-4f;
    const float a11 = -0x1.2f3004p-5f;
    const float a13 = +0x1.01a37cp-7f;

    const float x_times_x = x * x;
    float arctan = a13;
    arctan = fmaf(x_times_x, arctan, a11);
    arctan = fmaf(x_times_x, arctan, a9);
    arctan = fmaf(x_times_x, arctan, a7);
    arctan = fmaf(x_times_x, arctan, a5);
    arctan = fmaf(x_times_x, arctan, a3);
    arctan = fmaf(x_times_x, arctan, a1);
    arctan *= x;

    return arctan;
}
////////////////////////////////////////////////////////////////////////
// arctan(x)
////////////////////////////////////////////////////////////////////////
static inline float volk_arctan(const float x)
{
    /*
     *  arctan(x) + arctan(1 / x) == sign(x) * pi / 2
     */
    const float pi_2 = 0x1.921fb6p0f;

    if (fabs(x) < 1.f) {
        return volk_arctan_poly(x);
    } else {
        return copysignf(pi_2, x) - volk_arctan_poly(1.f / x);
    }
}
////////////////////////////////////////////////////////////////////////
// arctan2(y, x)
////////////////////////////////////////////////////////////////////////
static inline float volk_atan2(const float y, const float x)
{
    /*
     *                /  arctan(y / x)         if x > 0
     *                |  arctan(y / x)  + PI   if x < 0 and y >= 0
     * atan2(y, x) =  |  arctan(y / x)  - PI   if x < 0 and y <  0
     *                |  sign(y) * PI / 2      if x = 0
     *                \  undefined             if x = 0 and y = 0
     * atan2f(0.f,  0.f) shall return  0.f
     * atan2f(0.f, -0.f) shall return -0.f
     */
    const float pi = 0x1.921fb6p1f;
    const float pi_2 = 0x1.921fb6p0f;

    if (fabs(x) == 0.f) {
        return (fabs(y) == 0.f) ? copysignf(0.f, y) : copysignf(pi_2, y);
    }
    const int swap = fabs(x) < fabs(y);
    const float input = swap ? (x / y) : (y / x);
    float result = volk_arctan_poly(input);
    result = swap ? (input >= 0.f ? pi_2 : -pi_2) - result : result;
    if (x < 0.f) {
        result += copysignf(pi, y);
    }
    return result;
}

#endif /*INCLUDED_LIBVOLK_COMMON_H*/
