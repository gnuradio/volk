/* -*- c++ -*- */
/*
 * Copyright 2015 Free Software Foundation, Inc.
 * Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * This file is intended to hold SSE intrinsics of intrinsics.
 * They should be used in VOLK kernels to avoid copy-pasta.
 */

#ifndef INCLUDE_VOLK_VOLK_SSE_INTRINSICS_H_
#define INCLUDE_VOLK_VOLK_SSE_INTRINSICS_H_
#include <xmmintrin.h>

/*
 * Approximate arctan(x) via polynomial expansion
 * on the interval [-1, 1]
 *
 * Maximum relative error ~6.5e-7
 * Polynomial evaluated via Horner's method
 */
static inline __m128 _mm_arctan_poly_sse(const __m128 x)
{
    const __m128 a1 = _mm_set1_ps(+0x1.ffffeap-1f);
    const __m128 a3 = _mm_set1_ps(-0x1.55437p-2f);
    const __m128 a5 = _mm_set1_ps(+0x1.972be6p-3f);
    const __m128 a7 = _mm_set1_ps(-0x1.1436ap-3f);
    const __m128 a9 = _mm_set1_ps(+0x1.5785aap-4f);
    const __m128 a11 = _mm_set1_ps(-0x1.2f3004p-5f);
    const __m128 a13 = _mm_set1_ps(+0x1.01a37cp-7f);

    const __m128 x_times_x = _mm_mul_ps(x, x);
    __m128 arctan;
    arctan = a13;
    arctan = _mm_mul_ps(x_times_x, arctan);
    arctan = _mm_add_ps(arctan, a11);
    arctan = _mm_mul_ps(x_times_x, arctan);
    arctan = _mm_add_ps(arctan, a9);
    arctan = _mm_mul_ps(x_times_x, arctan);
    arctan = _mm_add_ps(arctan, a7);
    arctan = _mm_mul_ps(x_times_x, arctan);
    arctan = _mm_add_ps(arctan, a5);
    arctan = _mm_mul_ps(x_times_x, arctan);
    arctan = _mm_add_ps(arctan, a3);
    arctan = _mm_mul_ps(x_times_x, arctan);
    arctan = _mm_add_ps(arctan, a1);
    arctan = _mm_mul_ps(x, arctan);

    return arctan;
}

/*
 * Approximate arcsin(x) via polynomial expansion
 * P(u) such that asin(x) = x * P(x^2) on |x| <= 0.5
 *
 * Maximum relative error ~1.5e-6
 * Polynomial evaluated via Horner's method
 */
static inline __m128 _mm_arcsin_poly_sse(const __m128 x)
{
    const __m128 c0 = _mm_set1_ps(0x1.ffffcep-1f);
    const __m128 c1 = _mm_set1_ps(0x1.55b648p-3f);
    const __m128 c2 = _mm_set1_ps(0x1.24d192p-4f);
    const __m128 c3 = _mm_set1_ps(0x1.0a788p-4f);

    const __m128 u = _mm_mul_ps(x, x);
    __m128 p = c3;
    p = _mm_mul_ps(u, p);
    p = _mm_add_ps(p, c2);
    p = _mm_mul_ps(u, p);
    p = _mm_add_ps(p, c1);
    p = _mm_mul_ps(u, p);
    p = _mm_add_ps(p, c0);

    return _mm_mul_ps(x, p);
}

static inline __m128 _mm_magnitudesquared_ps(__m128 cplxValue1, __m128 cplxValue2)
{
    __m128 iValue, qValue;
    // Arrange in i1i2i3i4 format
    iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
    // Arrange in q1q2q3q4 format
    qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));
    iValue = _mm_mul_ps(iValue, iValue); // Square the I values
    qValue = _mm_mul_ps(qValue, qValue); // Square the Q Values
    return _mm_add_ps(iValue, qValue);   // Add the I2 and Q2 values
}

static inline __m128 _mm_magnitude_ps(__m128 cplxValue1, __m128 cplxValue2)
{
    return _mm_sqrt_ps(_mm_magnitudesquared_ps(cplxValue1, cplxValue2));
}

static inline __m128 _mm_scaled_norm_dist_ps_sse(const __m128 symbols0,
                                                 const __m128 symbols1,
                                                 const __m128 points0,
                                                 const __m128 points1,
                                                 const __m128 scalar)
{
    // calculate scalar * |x - y|^2
    const __m128 diff0 = _mm_sub_ps(symbols0, points0);
    const __m128 diff1 = _mm_sub_ps(symbols1, points1);
    const __m128 norms = _mm_magnitudesquared_ps(diff0, diff1);
    return _mm_mul_ps(norms, scalar);
}

static inline __m128 _mm_accumulate_square_sum_ps(
    __m128 sq_acc, __m128 acc, __m128 val, __m128 rec, __m128 aux)
{
    aux = _mm_mul_ps(aux, val);
    aux = _mm_sub_ps(aux, acc);
    aux = _mm_mul_ps(aux, aux);
    aux = _mm_mul_ps(aux, rec);
    return _mm_add_ps(sq_acc, aux);
}

#endif /* INCLUDE_VOLK_VOLK_SSE_INTRINSICS_H_ */
