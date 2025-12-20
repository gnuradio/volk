/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_cos_32f
 *
 * \b Overview
 *
 * Computes cosine of the input vector and stores results in the output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_cos_32f(float* bVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li bVector: The vector where results will be stored.
 *
 * \b Example
 * Calculate cos(theta) for common angles.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   in[0] = 0.000;
 *   in[1] = 0.524;
 *   in[2] = 0.786;
 *   in[3] = 1.047;
 *   in[4] = 1.571;
 *   in[5] = 1.571;
 *   in[6] = 2.094;
 *   in[7] = 2.356;
 *   in[8] = 2.618;
 *   in[9] = 3.142;
 *
 *   volk_32f_cos_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("cos(%1.3f) = %1.3f\n", in[ii], out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifndef INCLUDED_volk_32f_cos_32f_a_H
#define INCLUDED_volk_32f_cos_32f_a_H

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_cos_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (; number < num_points; number++) {
        *bPtr++ = cosf(*aPtr++);
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void volk_32f_cos_32f_a_avx512f(float* cosVector,
                                              const float* inVector,
                                              unsigned int num_points)
{
    float* cosPtr = cosVector;
    const float* inPtr = inVector;

    unsigned int number = 0;
    unsigned int sixteenPoints = num_points / 16;

    // Constants for Cody-Waite argument reduction
    // n = round(x * 2/pi), then r = x - n * pi/2
    const __m512 two_over_pi = _mm512_set1_ps(0x1.45f306p-1f);    // 2/pi
    const __m512 pi_over_2_hi = _mm512_set1_ps(0x1.921fb6p+0f);   // pi/2 high
    const __m512 pi_over_2_lo = _mm512_set1_ps(-0x1.777a5cp-25f); // pi/2 low

    const __m512i ones = _mm512_set1_epi32(1);
    const __m512i twos = _mm512_set1_epi32(2);
    const __m512i sign_bit = _mm512_set1_epi32(0x80000000);

    for (; number < sixteenPoints; number++) {
        __m512 x = _mm512_load_ps(inPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m512 n_f = _mm512_roundscale_ps(_mm512_mul_ps(x, two_over_pi),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512i n = _mm512_cvtps_epi32(n_f);

        // r = x - n * (pi/2), using extended precision
        __m512 r = _mm512_fnmadd_ps(n_f, pi_over_2_hi, x);
        r = _mm512_fnmadd_ps(n_f, pi_over_2_lo, r);

        // Evaluate both sin and cos polynomials
        __m512 sin_r = _mm512_sin_poly_avx512(r);
        __m512 cos_r = _mm512_cos_poly_avx512(r);

        // Reconstruct cos(x) based on quadrant (n mod 4):
        // n&1 == 0: use cos_r, n&1 == 1: use sin_r
        // (n+1)&2 == 0: positive, (n+1)&2 != 0: negative
        __m512i n_and_1 = _mm512_and_si512(n, ones);
        __m512i n_plus_1_and_2 = _mm512_and_si512(_mm512_add_epi32(n, ones), twos);

        // swap_mask: where n&1 != 0, we use sin instead of cos
        __mmask16 swap_mask = _mm512_cmpeq_epi32_mask(n_and_1, ones);
        __m512 result = _mm512_mask_blend_ps(swap_mask, cos_r, sin_r);

        // neg_mask: where (n+1)&2 != 0, we negate the result (use integer xor for
        // AVX512F)
        __mmask16 neg_mask = _mm512_cmpeq_epi32_mask(n_plus_1_and_2, twos);
        result = _mm512_castsi512_ps(_mm512_mask_xor_epi32(_mm512_castps_si512(result),
                                                           neg_mask,
                                                           _mm512_castps_si512(result),
                                                           sign_bit));

        _mm512_store_ps(cosPtr, result);
        inPtr += 16;
        cosPtr += 16;
    }

    number = sixteenPoints * 16;
    for (; number < num_points; number++) {
        *cosPtr++ = cosf(*inPtr++);
    }
}
#endif /* LV_HAVE_AVX512F */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void
volk_32f_cos_32f_a_avx2_fma(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;

    // Constants for Cody-Waite argument reduction
    // n = round(x * 2/pi), then r = x - n * pi/2
    const __m256 two_over_pi = _mm256_set1_ps(0x1.45f306p-1f);    // 2/pi
    const __m256 pi_over_2_hi = _mm256_set1_ps(0x1.921fb6p+0f);   // pi/2 high
    const __m256 pi_over_2_lo = _mm256_set1_ps(-0x1.777a5cp-25f); // pi/2 low

    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i twos = _mm256_set1_epi32(2);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    for (; number < eighthPoints; number++) {
        __m256 x = _mm256_load_ps(aPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m256 n_f = _mm256_round_ps(_mm256_mul_ps(x, two_over_pi),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i n = _mm256_cvtps_epi32(n_f);

        // r = x - n * (pi/2), using extended precision
        __m256 r = _mm256_fnmadd_ps(n_f, pi_over_2_hi, x);
        r = _mm256_fnmadd_ps(n_f, pi_over_2_lo, r);

        // Evaluate both sin and cos polynomials
        __m256 sin_r = _mm256_sin_poly_avx2_fma(r);
        __m256 cos_r = _mm256_cos_poly_avx2_fma(r);

        // Reconstruct cos(x) based on quadrant (n mod 4):
        // n&1 == 0: use cos_r, n&1 == 1: use sin_r
        // (n+1)&2 == 0: positive, (n+1)&2 != 0: negative
        __m256i n_and_1 = _mm256_and_si256(n, ones);
        __m256i n_plus_1_and_2 = _mm256_and_si256(_mm256_add_epi32(n, ones), twos);

        // swap_mask: where n&1 != 0, we use sin instead of cos
        __m256 swap_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_and_1, ones));
        __m256 result = _mm256_blendv_ps(cos_r, sin_r, swap_mask);

        // neg_mask: where (n+1)&2 != 0, we negate the result
        __m256 neg_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_plus_1_and_2, twos));
        result = _mm256_xor_ps(result, _mm256_and_ps(neg_mask, sign_bit));

        _mm256_store_ps(bPtr, result);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = cosf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void
volk_32f_cos_32f_a_avx2(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;

    // Constants for Cody-Waite argument reduction
    // n = round(x * 2/pi), then r = x - n * pi/2
    const __m256 two_over_pi = _mm256_set1_ps(0x1.45f306p-1f);    // 2/pi
    const __m256 pi_over_2_hi = _mm256_set1_ps(0x1.921fb6p+0f);   // pi/2 high
    const __m256 pi_over_2_lo = _mm256_set1_ps(-0x1.777a5cp-25f); // pi/2 low

    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i twos = _mm256_set1_epi32(2);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    for (; number < eighthPoints; number++) {
        __m256 x = _mm256_load_ps(aPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m256 n_f = _mm256_round_ps(_mm256_mul_ps(x, two_over_pi),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i n = _mm256_cvtps_epi32(n_f);

        // r = x - n * (pi/2), using extended precision
        __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n_f, pi_over_2_hi));
        r = _mm256_sub_ps(r, _mm256_mul_ps(n_f, pi_over_2_lo));

        // Evaluate both sin and cos polynomials
        __m256 sin_r = _mm256_sin_poly_avx2(r);
        __m256 cos_r = _mm256_cos_poly_avx2(r);

        // Reconstruct cos(x) based on quadrant (n mod 4):
        // n&1 == 0: use cos_r, n&1 == 1: use sin_r
        // (n+1)&2 == 0: positive, (n+1)&2 != 0: negative
        __m256i n_and_1 = _mm256_and_si256(n, ones);
        __m256i n_plus_1_and_2 = _mm256_and_si256(_mm256_add_epi32(n, ones), twos);

        // swap_mask: where n&1 != 0, we use sin instead of cos
        __m256 swap_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_and_1, ones));
        __m256 result = _mm256_blendv_ps(cos_r, sin_r, swap_mask);

        // neg_mask: where (n+1)&2 != 0, we negate the result
        __m256 neg_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_plus_1_and_2, twos));
        result = _mm256_xor_ps(result, _mm256_and_ps(neg_mask, sign_bit));

        _mm256_store_ps(bPtr, result);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = cosf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
#include <volk/volk_sse_intrinsics.h>

static inline void
volk_32f_cos_32f_a_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;

    // Constants for Cody-Waite argument reduction
    // n = round(x * 2/pi), then r = x - n * pi/2
    const __m128 two_over_pi = _mm_set1_ps(0x1.45f306p-1f);    // 2/pi
    const __m128 pi_over_2_hi = _mm_set1_ps(0x1.921fb6p+0f);   // pi/2 high
    const __m128 pi_over_2_lo = _mm_set1_ps(-0x1.777a5cp-25f); // pi/2 low

    const __m128i ones = _mm_set1_epi32(1);
    const __m128i twos = _mm_set1_epi32(2);
    const __m128 sign_bit = _mm_set1_ps(-0.0f);

    for (; number < quarterPoints; number++) {
        __m128 x = _mm_load_ps(aPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m128 n_f = _mm_round_ps(_mm_mul_ps(x, two_over_pi),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i n = _mm_cvtps_epi32(n_f);

        // r = x - n * (pi/2), using extended precision
        __m128 r = _mm_sub_ps(x, _mm_mul_ps(n_f, pi_over_2_hi));
        r = _mm_sub_ps(r, _mm_mul_ps(n_f, pi_over_2_lo));

        // Evaluate both sin and cos polynomials
        __m128 sin_r = _mm_sin_poly_sse(r);
        __m128 cos_r = _mm_cos_poly_sse(r);

        // Reconstruct cos(x) based on quadrant (n mod 4):
        // n&1 == 0: use cos_r, n&1 == 1: use sin_r
        // (n+1)&2 == 0: positive, (n+1)&2 != 0: negative
        __m128i n_and_1 = _mm_and_si128(n, ones);
        __m128i n_plus_1_and_2 = _mm_and_si128(_mm_add_epi32(n, ones), twos);

        // swap_mask: where n&1 != 0, we use sin instead of cos
        __m128 swap_mask = _mm_castsi128_ps(_mm_cmpeq_epi32(n_and_1, ones));
        __m128 result = _mm_blendv_ps(cos_r, sin_r, swap_mask);

        // neg_mask: where (n+1)&2 != 0, we negate the result
        __m128 neg_mask = _mm_castsi128_ps(_mm_cmpeq_epi32(n_plus_1_and_2, twos));
        result = _mm_xor_ps(result, _mm_and_ps(neg_mask, sign_bit));

        _mm_store_ps(bPtr, result);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = cosf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 */

#endif /* INCLUDED_volk_32f_cos_32f_a_H */


#ifndef INCLUDED_volk_32f_cos_32f_u_H
#define INCLUDED_volk_32f_cos_32f_u_H

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void volk_32f_cos_32f_u_avx512f(float* cosVector,
                                              const float* inVector,
                                              unsigned int num_points)
{
    float* cosPtr = cosVector;
    const float* inPtr = inVector;

    unsigned int number = 0;
    unsigned int sixteenPoints = num_points / 16;

    // Constants for Cody-Waite argument reduction
    // n = round(x * 2/pi), then r = x - n * pi/2
    const __m512 two_over_pi = _mm512_set1_ps(0x1.45f306p-1f);    // 2/pi
    const __m512 pi_over_2_hi = _mm512_set1_ps(0x1.921fb6p+0f);   // pi/2 high
    const __m512 pi_over_2_lo = _mm512_set1_ps(-0x1.777a5cp-25f); // pi/2 low

    const __m512i ones = _mm512_set1_epi32(1);
    const __m512i twos = _mm512_set1_epi32(2);
    const __m512i sign_bit = _mm512_set1_epi32(0x80000000);

    for (; number < sixteenPoints; number++) {
        __m512 x = _mm512_loadu_ps(inPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m512 n_f = _mm512_roundscale_ps(_mm512_mul_ps(x, two_over_pi),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512i n = _mm512_cvtps_epi32(n_f);

        // r = x - n * (pi/2), using extended precision
        __m512 r = _mm512_fnmadd_ps(n_f, pi_over_2_hi, x);
        r = _mm512_fnmadd_ps(n_f, pi_over_2_lo, r);

        // Evaluate both sin and cos polynomials
        __m512 sin_r = _mm512_sin_poly_avx512(r);
        __m512 cos_r = _mm512_cos_poly_avx512(r);

        // Reconstruct cos(x) based on quadrant (n mod 4):
        // n&1 == 0: use cos_r, n&1 == 1: use sin_r
        // (n+1)&2 == 0: positive, (n+1)&2 != 0: negative
        __m512i n_and_1 = _mm512_and_si512(n, ones);
        __m512i n_plus_1_and_2 = _mm512_and_si512(_mm512_add_epi32(n, ones), twos);

        // swap_mask: where n&1 != 0, we use sin instead of cos
        __mmask16 swap_mask = _mm512_cmpeq_epi32_mask(n_and_1, ones);
        __m512 result = _mm512_mask_blend_ps(swap_mask, cos_r, sin_r);

        // neg_mask: where (n+1)&2 != 0, we negate the result (use integer xor for
        // AVX512F)
        __mmask16 neg_mask = _mm512_cmpeq_epi32_mask(n_plus_1_and_2, twos);
        result = _mm512_castsi512_ps(_mm512_mask_xor_epi32(_mm512_castps_si512(result),
                                                           neg_mask,
                                                           _mm512_castps_si512(result),
                                                           sign_bit));

        _mm512_storeu_ps(cosPtr, result);
        inPtr += 16;
        cosPtr += 16;
    }

    number = sixteenPoints * 16;
    for (; number < num_points; number++) {
        *cosPtr++ = cosf(*inPtr++);
    }
}
#endif /* LV_HAVE_AVX512F */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void
volk_32f_cos_32f_u_avx2_fma(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;

    // Constants for Cody-Waite argument reduction
    // n = round(x * 2/pi), then r = x - n * pi/2
    const __m256 two_over_pi = _mm256_set1_ps(0x1.45f306p-1f);    // 2/pi
    const __m256 pi_over_2_hi = _mm256_set1_ps(0x1.921fb6p+0f);   // pi/2 high
    const __m256 pi_over_2_lo = _mm256_set1_ps(-0x1.777a5cp-25f); // pi/2 low

    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i twos = _mm256_set1_epi32(2);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    for (; number < eighthPoints; number++) {
        __m256 x = _mm256_loadu_ps(aPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m256 n_f = _mm256_round_ps(_mm256_mul_ps(x, two_over_pi),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i n = _mm256_cvtps_epi32(n_f);

        // r = x - n * (pi/2), using extended precision
        __m256 r = _mm256_fnmadd_ps(n_f, pi_over_2_hi, x);
        r = _mm256_fnmadd_ps(n_f, pi_over_2_lo, r);

        // Evaluate both sin and cos polynomials
        __m256 sin_r = _mm256_sin_poly_avx2_fma(r);
        __m256 cos_r = _mm256_cos_poly_avx2_fma(r);

        // Reconstruct cos(x) based on quadrant (n mod 4):
        // n&1 == 0: use cos_r, n&1 == 1: use sin_r
        // (n+1)&2 == 0: positive, (n+1)&2 != 0: negative
        __m256i n_and_1 = _mm256_and_si256(n, ones);
        __m256i n_plus_1_and_2 = _mm256_and_si256(_mm256_add_epi32(n, ones), twos);

        // swap_mask: where n&1 != 0, we use sin instead of cos
        __m256 swap_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_and_1, ones));
        __m256 result = _mm256_blendv_ps(cos_r, sin_r, swap_mask);

        // neg_mask: where (n+1)&2 != 0, we negate the result
        __m256 neg_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_plus_1_and_2, twos));
        result = _mm256_xor_ps(result, _mm256_and_ps(neg_mask, sign_bit));

        _mm256_storeu_ps(bPtr, result);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = cosf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void
volk_32f_cos_32f_u_avx2(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;

    // Constants for Cody-Waite argument reduction
    // n = round(x * 2/pi), then r = x - n * pi/2
    const __m256 two_over_pi = _mm256_set1_ps(0x1.45f306p-1f);    // 2/pi
    const __m256 pi_over_2_hi = _mm256_set1_ps(0x1.921fb6p+0f);   // pi/2 high
    const __m256 pi_over_2_lo = _mm256_set1_ps(-0x1.777a5cp-25f); // pi/2 low

    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i twos = _mm256_set1_epi32(2);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    for (; number < eighthPoints; number++) {
        __m256 x = _mm256_loadu_ps(aPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m256 n_f = _mm256_round_ps(_mm256_mul_ps(x, two_over_pi),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i n = _mm256_cvtps_epi32(n_f);

        // r = x - n * (pi/2), using extended precision
        __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n_f, pi_over_2_hi));
        r = _mm256_sub_ps(r, _mm256_mul_ps(n_f, pi_over_2_lo));

        // Evaluate both sin and cos polynomials
        __m256 sin_r = _mm256_sin_poly_avx2(r);
        __m256 cos_r = _mm256_cos_poly_avx2(r);

        // Reconstruct cos(x) based on quadrant (n mod 4):
        // n&1 == 0: use cos_r, n&1 == 1: use sin_r
        // (n+1)&2 == 0: positive, (n+1)&2 != 0: negative
        __m256i n_and_1 = _mm256_and_si256(n, ones);
        __m256i n_plus_1_and_2 = _mm256_and_si256(_mm256_add_epi32(n, ones), twos);

        // swap_mask: where n&1 != 0, we use sin instead of cos
        __m256 swap_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_and_1, ones));
        __m256 result = _mm256_blendv_ps(cos_r, sin_r, swap_mask);

        // neg_mask: where (n+1)&2 != 0, we negate the result
        __m256 neg_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_plus_1_and_2, twos));
        result = _mm256_xor_ps(result, _mm256_and_ps(neg_mask, sign_bit));

        _mm256_storeu_ps(bPtr, result);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = cosf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
#include <volk/volk_sse_intrinsics.h>

static inline void
volk_32f_cos_32f_u_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;

    // Constants for Cody-Waite argument reduction
    // n = round(x * 2/pi), then r = x - n * pi/2
    const __m128 two_over_pi = _mm_set1_ps(0x1.45f306p-1f);    // 2/pi
    const __m128 pi_over_2_hi = _mm_set1_ps(0x1.921fb6p+0f);   // pi/2 high
    const __m128 pi_over_2_lo = _mm_set1_ps(-0x1.777a5cp-25f); // pi/2 low

    const __m128i ones = _mm_set1_epi32(1);
    const __m128i twos = _mm_set1_epi32(2);
    const __m128 sign_bit = _mm_set1_ps(-0.0f);

    for (; number < quarterPoints; number++) {
        __m128 x = _mm_loadu_ps(aPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m128 n_f = _mm_round_ps(_mm_mul_ps(x, two_over_pi),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i n = _mm_cvtps_epi32(n_f);

        // r = x - n * (pi/2), using extended precision
        __m128 r = _mm_sub_ps(x, _mm_mul_ps(n_f, pi_over_2_hi));
        r = _mm_sub_ps(r, _mm_mul_ps(n_f, pi_over_2_lo));

        // Evaluate both sin and cos polynomials
        __m128 sin_r = _mm_sin_poly_sse(r);
        __m128 cos_r = _mm_cos_poly_sse(r);

        // Reconstruct cos(x) based on quadrant (n mod 4):
        // n&1 == 0: use cos_r, n&1 == 1: use sin_r
        // (n+1)&2 == 0: positive, (n+1)&2 != 0: negative
        __m128i n_and_1 = _mm_and_si128(n, ones);
        __m128i n_plus_1_and_2 = _mm_and_si128(_mm_add_epi32(n, ones), twos);

        // swap_mask: where n&1 != 0, we use sin instead of cos
        __m128 swap_mask = _mm_castsi128_ps(_mm_cmpeq_epi32(n_and_1, ones));
        __m128 result = _mm_blendv_ps(cos_r, sin_r, swap_mask);

        // neg_mask: where (n+1)&2 != 0, we negate the result
        __m128 neg_mask = _mm_castsi128_ps(_mm_cmpeq_epi32(n_plus_1_and_2, twos));
        result = _mm_xor_ps(result, _mm_and_ps(neg_mask, sign_bit));

        _mm_storeu_ps(bPtr, result);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = cosf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

/* NEON polynomial-based cos using Cody-Waite argument reduction */
static inline void
volk_32f_cos_32f_neon(float* bVector, const float* aVector, unsigned int num_points)
{
    // Cody-Waite argument reduction: n = round(x * 2/pi), r = x - n * pi/2
    const float32x4_t two_over_pi = vdupq_n_f32(0x1.45f306p-1f);    // 2/pi
    const float32x4_t pi_over_2_hi = vdupq_n_f32(0x1.921fb6p+0f);   // pi/2 high
    const float32x4_t pi_over_2_lo = vdupq_n_f32(-0x1.777a5cp-25f); // pi/2 low

    const int32x4_t ones = vdupq_n_s32(1);
    const int32x4_t twos = vdupq_n_s32(2);
    const float32x4_t sign_bit = vdupq_n_f32(-0.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t neg_half = vdupq_n_f32(-0.5f);
    const float32x4_t fzeroes = vdupq_n_f32(0.0f);

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    for (; number < quarterPoints; number++) {
        float32x4_t x = vld1q_f32(aVector);
        aVector += 4;

        // n = round(x * 2/pi) - emulate round-to-nearest for ARMv7
        float32x4_t scaled = vmulq_f32(x, two_over_pi);
        uint32x4_t is_neg = vcltq_f32(scaled, fzeroes);
        float32x4_t adj = vbslq_f32(is_neg, neg_half, half);
        float32x4_t n_f = vcvtq_f32_s32(vcvtq_s32_f32(vaddq_f32(scaled, adj)));
        int32x4_t n = vcvtq_s32_f32(n_f);

        // r = x - n * (pi/2) using extended precision
        float32x4_t r = vmlsq_f32(x, n_f, pi_over_2_hi);
        r = vmlsq_f32(r, n_f, pi_over_2_lo);

        // Evaluate sin and cos polynomials
        float32x4_t sin_r = _vsin_poly_f32(r);
        float32x4_t cos_r = _vcos_poly_f32(r);

        // Quadrant-based reconstruction for cos:
        // n&1 == 0: use cos_r, n&1 == 1: use sin_r
        // (n+1)&2 == 0: positive, (n+1)&2 == 2: negative
        int32x4_t n_and_1 = vandq_s32(n, ones);
        int32x4_t n_plus_1_and_2 = vandq_s32(vaddq_s32(n, ones), twos);

        uint32x4_t swap_mask = vceqq_s32(n_and_1, ones);
        float32x4_t result = vbslq_f32(swap_mask, sin_r, cos_r);

        uint32x4_t neg_mask = vceqq_s32(n_plus_1_and_2, twos);
        result = vreinterpretq_f32_u32(
            veorq_u32(vreinterpretq_u32_f32(result),
                      vandq_u32(neg_mask, vreinterpretq_u32_f32(sign_bit))));

        vst1q_f32(bVector, result);
        bVector += 4;
    }

    for (number = quarterPoints * 4; number < num_points; number++) {
        *bVector++ = cosf(*aVector++);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

/* NEONv8 polynomial-based cos using Cody-Waite argument reduction with FMA */
static inline void
volk_32f_cos_32f_neonv8(float* bVector, const float* aVector, unsigned int num_points)
{
    // Cody-Waite argument reduction: n = round(x * 2/pi), r = x - n * pi/2
    const float32x4_t two_over_pi = vdupq_n_f32(0x1.45f306p-1f);    // 2/pi
    const float32x4_t pi_over_2_hi = vdupq_n_f32(0x1.921fb6p+0f);   // pi/2 high
    const float32x4_t pi_over_2_lo = vdupq_n_f32(-0x1.777a5cp-25f); // pi/2 low

    const int32x4_t ones = vdupq_n_s32(1);
    const int32x4_t twos = vdupq_n_s32(2);
    const float32x4_t sign_bit = vdupq_n_f32(-0.0f);

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    for (; number < quarterPoints; number++) {
        float32x4_t x = vld1q_f32(aVector);
        aVector += 4;

        // n = round(x * 2/pi) using ARMv8 vrndnq_f32
        float32x4_t n_f = vrndnq_f32(vmulq_f32(x, two_over_pi));
        int32x4_t n = vcvtq_s32_f32(n_f);

        // r = x - n * (pi/2) using FMA for extended precision
        float32x4_t r = vfmsq_f32(x, n_f, pi_over_2_hi);
        r = vfmsq_f32(r, n_f, pi_over_2_lo);

        // Evaluate sin and cos polynomials using FMA
        float32x4_t sin_r = _vsin_poly_neonv8(r);
        float32x4_t cos_r = _vcos_poly_neonv8(r);

        // Quadrant-based reconstruction for cos:
        // n&1 == 0: use cos_r, n&1 == 1: use sin_r
        // (n+1)&2 == 0: positive, (n+1)&2 == 2: negative
        int32x4_t n_and_1 = vandq_s32(n, ones);
        int32x4_t n_plus_1_and_2 = vandq_s32(vaddq_s32(n, ones), twos);

        uint32x4_t swap_mask = vceqq_s32(n_and_1, ones);
        float32x4_t result = vbslq_f32(swap_mask, sin_r, cos_r);

        uint32x4_t neg_mask = vceqq_s32(n_plus_1_and_2, twos);
        result = vreinterpretq_f32_u32(
            veorq_u32(vreinterpretq_u32_f32(result),
                      vandq_u32(neg_mask, vreinterpretq_u32_f32(sign_bit))));

        vst1q_f32(bVector, result);
        bVector += 4;
    }

    for (number = quarterPoints * 4; number < num_points; number++) {
        *bVector++ = cosf(*aVector++);
    }
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_cos_32f_rvv(float* bVector, const float* aVector, unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

    const vfloat32m2_t c4oPi = __riscv_vfmv_v_f_f32m2(1.2732395f, vlmax);
    const vfloat32m2_t cPio4a = __riscv_vfmv_v_f_f32m2(0.7853982f, vlmax);
    const vfloat32m2_t cPio4b = __riscv_vfmv_v_f_f32m2(7.946627e-09f, vlmax);
    const vfloat32m2_t cPio4c = __riscv_vfmv_v_f_f32m2(3.061617e-17f, vlmax);

    const vfloat32m2_t cf1 = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    const vfloat32m2_t cf4 = __riscv_vfmv_v_f_f32m2(4.0f, vlmax);

    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(0.0833333333f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(0.0027777778f, vlmax);
    const vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(4.9603175e-05f, vlmax);
    const vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(5.5114638e-07f, vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v = __riscv_vle32_v_f32m2(aVector, vl);
        vfloat32m2_t s = __riscv_vfabs(v, vl);
        vint32m2_t q = __riscv_vfcvt_x(__riscv_vfmul(s, c4oPi, vl), vl);
        vfloat32m2_t r = __riscv_vfcvt_f(__riscv_vadd(q, __riscv_vand(q, 1, vl), vl), vl);

        s = __riscv_vfnmsac(s, cPio4a, r, vl);
        s = __riscv_vfnmsac(s, cPio4b, r, vl);
        s = __riscv_vfnmsac(s, cPio4c, r, vl);

        s = __riscv_vfmul(s, 1 / 8.0f, vl);
        s = __riscv_vfmul(s, s, vl);
        vfloat32m2_t t = s;
        s = __riscv_vfmsub(s, c5, c4, vl);
        s = __riscv_vfmadd(s, t, c3, vl);
        s = __riscv_vfmsub(s, t, c2, vl);
        s = __riscv_vfmadd(s, t, cf1, vl);
        s = __riscv_vfmul(s, t, vl);
        s = __riscv_vfmul(s, __riscv_vfsub(cf4, s, vl), vl);
        s = __riscv_vfmul(s, __riscv_vfsub(cf4, s, vl), vl);
        s = __riscv_vfmul(s, __riscv_vfsub(cf4, s, vl), vl);
        s = __riscv_vfmul(s, 1 / 2.0f, vl);

        vfloat32m2_t sine =
            __riscv_vfsqrt(__riscv_vfmul(__riscv_vfrsub(s, 2.0f, vl), s, vl), vl);
        vfloat32m2_t cosine = __riscv_vfsub(cf1, s, vl);

        vbool16_t m1 = __riscv_vmsne(__riscv_vand(__riscv_vadd(q, 1, vl), 2, vl), 0, vl);
        vbool16_t m2 = __riscv_vmsne(__riscv_vand(__riscv_vadd(q, 2, vl), 4, vl), 0, vl);

        cosine = __riscv_vmerge(cosine, sine, m1, vl);
        cosine = __riscv_vfneg_mu(m2, cosine, cosine, vl);

        __riscv_vse32(bVector, cosine, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_cos_32f_u_H */
