/* -*- c++ -*- */
/*
 * Copyright 2023 - 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * This file is intended to hold AVX2 FMA intrinsics.
 * They should be used in VOLK kernels to avoid copy-paste.
 */

#ifndef INCLUDE_VOLK_VOLK_AVX2_FMA_INTRINSICS_H_
#define INCLUDE_VOLK_VOLK_AVX2_FMA_INTRINSICS_H_
#include <immintrin.h>

/*
 * Approximate arctan(x) via polynomial expansion
 * on the interval [-1, 1]
 *
 * Maximum relative error ~6.5e-7
 * Polynomial evaluated via Horner's method
 */
static inline __m256 _mm256_arctan_poly_avx2_fma(const __m256 x)
{
    const __m256 a1 = _mm256_set1_ps(+0x1.ffffeap-1f);
    const __m256 a3 = _mm256_set1_ps(-0x1.55437p-2f);
    const __m256 a5 = _mm256_set1_ps(+0x1.972be6p-3f);
    const __m256 a7 = _mm256_set1_ps(-0x1.1436ap-3f);
    const __m256 a9 = _mm256_set1_ps(+0x1.5785aap-4f);
    const __m256 a11 = _mm256_set1_ps(-0x1.2f3004p-5f);
    const __m256 a13 = _mm256_set1_ps(+0x1.01a37cp-7f);

    const __m256 x_times_x = _mm256_mul_ps(x, x);
    __m256 arctan;
    arctan = a13;
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a11);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a9);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a7);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a5);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a3);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a1);
    arctan = _mm256_mul_ps(x, arctan);

    return arctan;
}

/*
 * Approximate sin(x) via polynomial expansion
 * on the interval [-pi/4, pi/4]
 *
 * Maximum absolute error ~7.3e-9
 * sin(x) = x + x^3 * (s1 + x^2 * (s2 + x^2 * s3))
 */
static inline __m256 _mm256_sin_poly_avx2_fma(const __m256 x)
{
    const __m256 s1 = _mm256_set1_ps(-0x1.555552p-3f);
    const __m256 s2 = _mm256_set1_ps(+0x1.110be2p-7f);
    const __m256 s3 = _mm256_set1_ps(-0x1.9ab22ap-13f);

    const __m256 x2 = _mm256_mul_ps(x, x);
    const __m256 x3 = _mm256_mul_ps(x2, x);

    __m256 poly = _mm256_fmadd_ps(x2, s3, s2);
    poly = _mm256_fmadd_ps(x2, poly, s1);
    return _mm256_fmadd_ps(x3, poly, x);
}

/*
 * Approximate cos(x) via polynomial expansion
 * on the interval [-pi/4, pi/4]
 *
 * Maximum absolute error ~1.1e-7
 * cos(x) = 1 + x^2 * (c1 + x^2 * (c2 + x^2 * c3))
 */
static inline __m256 _mm256_cos_poly_avx2_fma(const __m256 x)
{
    const __m256 c1 = _mm256_set1_ps(-0x1.fffff4p-2f);
    const __m256 c2 = _mm256_set1_ps(+0x1.554a46p-5f);
    const __m256 c3 = _mm256_set1_ps(-0x1.661be2p-10f);
    const __m256 one = _mm256_set1_ps(1.0f);

    const __m256 x2 = _mm256_mul_ps(x, x);

    __m256 poly = _mm256_fmadd_ps(x2, c3, c2);
    poly = _mm256_fmadd_ps(x2, poly, c1);
    return _mm256_fmadd_ps(x2, poly, one);
}

#endif /* INCLUDE_VOLK_VOLK_AVX2_FMA_INTRINSICS_H_ */
