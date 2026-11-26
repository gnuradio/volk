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
 * Approximate arcsin(x) via polynomial expansion
 * P(u) such that asin(x) = x * P(x^2) on |x| <= 0.5
 *
 * Maximum relative error ~1.5e-6
 * Polynomial evaluated via Horner's method
 */
static inline __m256 _mm256_arcsin_poly_avx2_fma(const __m256 x)
{
    const __m256 c0 = _mm256_set1_ps(0x1.ffffcep-1f);
    const __m256 c1 = _mm256_set1_ps(0x1.55b648p-3f);
    const __m256 c2 = _mm256_set1_ps(0x1.24d192p-4f);
    const __m256 c3 = _mm256_set1_ps(0x1.0a788p-4f);

    const __m256 u = _mm256_mul_ps(x, x);
    __m256 p = c3;
    p = _mm256_fmadd_ps(u, p, c2);
    p = _mm256_fmadd_ps(u, p, c1);
    p = _mm256_fmadd_ps(u, p, c0);

    return _mm256_mul_ps(x, p);
}

/*
 * Minimax polynomial for sin(x) on [-pi/4, pi/4]
 * Coefficients via Remez algorithm (Sollya)
 * Max |error| < 7.3e-9
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
 * Minimax polynomial for cos(x) on [-pi/4, pi/4]
 * Coefficients via Remez algorithm (Sollya)
 * Max |error| < 1.1e-7
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

// The half-wave Cody-Waite reduction is bounded to |x| <= 2^20 radians.
static inline int _mm256_trig_half_wave_in_range(__m256 x)
{
    const __m256i magnitude =
        _mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32(0x7fffffff));
    const __m256i outside =
        _mm256_cmpgt_epi32(magnitude, _mm256_castps_si256(_mm256_set1_ps(0x1p20f)));
    const int in_range = _mm256_testz_si256(outside, outside);
#if defined(__GNUC__)
    return __builtin_expect(in_range, 1);
#else
    return in_range;
#endif
}

// Remez polynomial on [0, pi/2 + 1/8], extended by odd symmetry.
// The extra interval covers rounding in x/pi before selecting the half-wave.
static inline __m256 _mm256_trig_half_wave_avx2_fma(__m256 x, int cosine)
{
    const __m256 q = _mm256_mul_ps(x, _mm256_set1_ps(0x1.45f306p-2f));
    // sin(x) = (-1)^n sin(x - n*pi), with n = round(x/pi).
    // cos(x) = (-1)^n sin(x - (n - 1/2)*pi), with n = ceil(x/pi).
    const __m256 n_f =
        cosine ? _mm256_round_ps(q, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)
               : _mm256_round_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256i n = _mm256_cvtps_epi32(n_f);
    const __m256 k = cosine ? _mm256_sub_ps(n_f, _mm256_set1_ps(0.5f)) : n_f;
    __m256 r = _mm256_fnmadd_ps(k, _mm256_set1_ps(0x1.921fb6p+1f), x);
    r = _mm256_fnmadd_ps(k, _mm256_set1_ps(-0x1.777a5cp-24f), r);

    const __m256 r2 = _mm256_mul_ps(r, r);
    __m256 p = _mm256_fmadd_ps(
        r2, _mm256_set1_ps(0x1.59a39cp-19f), _mm256_set1_ps(-0x1.9f1f3cp-13f));
    p = _mm256_fmadd_ps(r2, p, _mm256_set1_ps(0x1.110ceap-7f));
    p = _mm256_fmadd_ps(r2, p, _mm256_set1_ps(-0x1.55553ep-3f));
    const __m256 y = _mm256_fmadd_ps(_mm256_mul_ps(r2, r), p, r);
    // Rounding can overshoot by one ulp near the peaks.
    const __m256 bounded =
        _mm256_min_ps(_mm256_max_ps(y, _mm256_set1_ps(-1.0f)), _mm256_set1_ps(1.0f));
    const __m256 sign = _mm256_castsi256_ps(_mm256_slli_epi32(n, 31));
    return _mm256_xor_ps(bounded, sign);
}

// Reduce once and evaluate the two short polynomials for paired outputs.
static inline void _mm256_sincos_avx2_fma(__m256 x, __m256* sin_x, __m256* cos_x)
{
    const __m256 n_f = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(0x1.45f306p-1f)),
                                       _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256i n = _mm256_cvtps_epi32(n_f);
    __m256 r = _mm256_fnmadd_ps(n_f, _mm256_set1_ps(0x1.921fb6p+0f), x);
    r = _mm256_fnmadd_ps(n_f, _mm256_set1_ps(-0x1.777a5cp-25f), r);
    const __m256 sin_r = _mm256_sin_poly_avx2_fma(r);
    const __m256 cos_r = _mm256_cos_poly_avx2_fma(r);

    // Swap on bit 0, flip sine on bit 1, and flip cosine when those bits differ.
    const __m256 swap = _mm256_castsi256_ps(_mm256_slli_epi32(n, 31));
    const __m256 sin_sign = _mm256_and_ps(_mm256_castsi256_ps(_mm256_slli_epi32(n, 30)),
                                          _mm256_set1_ps(-0.0f));
    const __m256 cos_sign = _mm256_xor_ps(sin_sign, swap);
    *sin_x = _mm256_xor_ps(_mm256_blendv_ps(sin_r, cos_r, swap), sin_sign);
    *cos_x = _mm256_xor_ps(_mm256_blendv_ps(cos_r, sin_r, swap), cos_sign);
}

/*
 * Polynomial coefficients for log2(x)/(x-1) on [1, 2]
 * Generated with Sollya: remez(log2(x)/(x-1), 6, [1+1b-20, 2])
 * Max error: ~1.55e-6
 *
 * Usage: log2(x) ≈ poly(x) * (x - 1) for x ∈ [1, 2]
 * Polynomial evaluated via Horner's method with FMA
 */
static inline __m256 _mm256_log2_poly_avx2_fma(const __m256 x)
{
    const __m256 c0 = _mm256_set1_ps(+0x1.a8a726p+1f);
    const __m256 c1 = _mm256_set1_ps(-0x1.0b7f7ep+2f);
    const __m256 c2 = _mm256_set1_ps(+0x1.05d9ccp+2f);
    const __m256 c3 = _mm256_set1_ps(-0x1.4d476cp+1f);
    const __m256 c4 = _mm256_set1_ps(+0x1.04fc3ap+0f);
    const __m256 c5 = _mm256_set1_ps(-0x1.c97982p-3f);
    const __m256 c6 = _mm256_set1_ps(+0x1.57aa42p-6f);

    // Horner's method with FMA: c0 + x*(c1 + x*(c2 + ...))
    __m256 poly = c6;
    poly = _mm256_fmadd_ps(poly, x, c5);
    poly = _mm256_fmadd_ps(poly, x, c4);
    poly = _mm256_fmadd_ps(poly, x, c3);
    poly = _mm256_fmadd_ps(poly, x, c2);
    poly = _mm256_fmadd_ps(poly, x, c1);
    poly = _mm256_fmadd_ps(poly, x, c0);
    return poly;
}

#endif /* INCLUDE_VOLK_VOLK_AVX2_FMA_INTRINSICS_H_ */
