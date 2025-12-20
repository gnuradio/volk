/* -*- c++ -*- */
/*
 * Copyright 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_sincos_32f_x2
 *
 * \b Overview
 *
 * Computes sine and cosine of the input vector simultaneously.
 * More efficient than calling sin and cos separately since
 * argument reduction is shared.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_sincos_32f_x2(float* sinVector, float* cosVector, const float* inVector,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inVector: The input vector of angles in radians.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li sinVector: The output vector of sine values.
 * \li cosVector: The output vector of cosine values.
 *
 * \b Example
 * Calculate sin and cos for common angles.
 * \code
 *   int N = 4;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* sin_out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* cos_out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   in[0] = 0.000;
 *   in[1] = 0.524;    // ~pi/6
 *   in[2] = 1.047;    // ~pi/3
 *   in[3] = 1.571;    // ~pi/2
 *
 *   volk_32f_sincos_32f_x2(sin_out, cos_out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("sincos(%1.3f) = (%1.3f, %1.3f)\n", in[ii], sin_out[ii], cos_out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(sin_out);
 *   volk_free(cos_out);
 * \endcode
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifndef INCLUDED_volk_32f_sincos_32f_x2_a_H
#define INCLUDED_volk_32f_sincos_32f_x2_a_H

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_sincos_32f_x2_generic(float* sinVector,
                                                  float* cosVector,
                                                  const float* inVector,
                                                  unsigned int num_points)
{
    for (unsigned int i = 0; i < num_points; i++) {
        sinVector[i] = sinf(inVector[i]);
        cosVector[i] = cosf(inVector[i]);
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void volk_32f_sincos_32f_x2_a_avx512f(float* sinVector,
                                                    float* cosVector,
                                                    const float* inVector,
                                                    unsigned int num_points)
{
    float* sinPtr = sinVector;
    float* cosPtr = cosVector;
    const float* inPtr = inVector;

    unsigned int number = 0;
    unsigned int sixteenPoints = num_points / 16;

    // Constants for Cody-Waite argument reduction
    const __m512 two_over_pi = _mm512_set1_ps(0x1.45f306p-1f);
    const __m512 pi_over_2_hi = _mm512_set1_ps(0x1.921fb6p+0f);
    const __m512 pi_over_2_lo = _mm512_set1_ps(-0x1.777a5cp-25f);

    const __m512i ones = _mm512_set1_epi32(1);
    const __m512i twos = _mm512_set1_epi32(2);
    const __m512i sign_bit = _mm512_set1_epi32(0x80000000);

    for (; number < sixteenPoints; number++) {
        __m512 x = _mm512_load_ps(inPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m512 n_f = _mm512_roundscale_ps(_mm512_mul_ps(x, two_over_pi),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512i n = _mm512_cvtps_epi32(n_f);

        // r = x - n * (pi/2)
        __m512 r = _mm512_fnmadd_ps(n_f, pi_over_2_hi, x);
        r = _mm512_fnmadd_ps(n_f, pi_over_2_lo, r);

        // Evaluate both sin and cos polynomials
        __m512 sin_r = _mm512_sin_poly_avx512(r);
        __m512 cos_r = _mm512_cos_poly_avx512(r);

        // Quadrant selection
        __m512i n_and_1 = _mm512_and_si512(n, ones);
        __m512i n_and_2 = _mm512_and_si512(n, twos);
        __m512i n_plus_1_and_2 = _mm512_and_si512(_mm512_add_epi32(n, ones), twos);

        // For sin: swap when n&1, negate when n&2
        __mmask16 sin_swap = _mm512_cmpeq_epi32_mask(n_and_1, ones);
        __m512 sin_result = _mm512_mask_blend_ps(sin_swap, sin_r, cos_r);
        __mmask16 sin_neg = _mm512_cmpeq_epi32_mask(n_and_2, twos);
        sin_result =
            _mm512_castsi512_ps(_mm512_mask_xor_epi32(_mm512_castps_si512(sin_result),
                                                      sin_neg,
                                                      _mm512_castps_si512(sin_result),
                                                      sign_bit));

        // For cos: swap when n&1, negate when (n+1)&2
        __mmask16 cos_swap = sin_swap;
        __m512 cos_result = _mm512_mask_blend_ps(cos_swap, cos_r, sin_r);
        __mmask16 cos_neg = _mm512_cmpeq_epi32_mask(n_plus_1_and_2, twos);
        cos_result =
            _mm512_castsi512_ps(_mm512_mask_xor_epi32(_mm512_castps_si512(cos_result),
                                                      cos_neg,
                                                      _mm512_castps_si512(cos_result),
                                                      sign_bit));

        _mm512_store_ps(sinPtr, sin_result);
        _mm512_store_ps(cosPtr, cos_result);
        inPtr += 16;
        sinPtr += 16;
        cosPtr += 16;
    }

    number = sixteenPoints * 16;
    for (; number < num_points; number++) {
        *sinPtr++ = sinf(*inPtr);
        *cosPtr++ = cosf(*inPtr++);
    }
}

#endif /* LV_HAVE_AVX512F for aligned */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_32f_sincos_32f_x2_a_avx2_fma(float* sinVector,
                                                     float* cosVector,
                                                     const float* inVector,
                                                     unsigned int num_points)
{
    float* sinPtr = sinVector;
    float* cosPtr = cosVector;
    const float* inPtr = inVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;

    // Constants for Cody-Waite argument reduction
    const __m256 two_over_pi = _mm256_set1_ps(0x1.45f306p-1f);
    const __m256 pi_over_2_hi = _mm256_set1_ps(0x1.921fb6p+0f);
    const __m256 pi_over_2_lo = _mm256_set1_ps(-0x1.777a5cp-25f);

    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i twos = _mm256_set1_epi32(2);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    for (; number < eighthPoints; number++) {
        __m256 x = _mm256_load_ps(inPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m256 n_f = _mm256_round_ps(_mm256_mul_ps(x, two_over_pi),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i n = _mm256_cvtps_epi32(n_f);

        // r = x - n * (pi/2)
        __m256 r = _mm256_fnmadd_ps(n_f, pi_over_2_hi, x);
        r = _mm256_fnmadd_ps(n_f, pi_over_2_lo, r);

        // Evaluate both sin and cos polynomials
        __m256 sin_r = _mm256_sin_poly_avx2_fma(r);
        __m256 cos_r = _mm256_cos_poly_avx2_fma(r);

        // Quadrant selection
        __m256i n_and_1 = _mm256_and_si256(n, ones);
        __m256i n_and_2 = _mm256_and_si256(n, twos);
        __m256i n_plus_1_and_2 = _mm256_and_si256(_mm256_add_epi32(n, ones), twos);

        // For sin: swap when n&1, negate when n&2
        __m256 sin_swap = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_and_1, ones));
        __m256 sin_result = _mm256_blendv_ps(sin_r, cos_r, sin_swap);
        __m256 sin_neg = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_and_2, twos));
        sin_result = _mm256_xor_ps(sin_result, _mm256_and_ps(sin_neg, sign_bit));

        // For cos: swap when n&1, negate when (n+1)&2
        __m256 cos_result = _mm256_blendv_ps(cos_r, sin_r, sin_swap);
        __m256 cos_neg = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_plus_1_and_2, twos));
        cos_result = _mm256_xor_ps(cos_result, _mm256_and_ps(cos_neg, sign_bit));

        _mm256_store_ps(sinPtr, sin_result);
        _mm256_store_ps(cosPtr, cos_result);
        inPtr += 8;
        sinPtr += 8;
        cosPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *sinPtr++ = sinf(*inPtr);
        *cosPtr++ = cosf(*inPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for aligned */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void volk_32f_sincos_32f_x2_a_avx2(float* sinVector,
                                                 float* cosVector,
                                                 const float* inVector,
                                                 unsigned int num_points)
{
    float* sinPtr = sinVector;
    float* cosPtr = cosVector;
    const float* inPtr = inVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;

    // Constants for Cody-Waite argument reduction
    const __m256 two_over_pi = _mm256_set1_ps(0x1.45f306p-1f);
    const __m256 pi_over_2_hi = _mm256_set1_ps(0x1.921fb6p+0f);
    const __m256 pi_over_2_lo = _mm256_set1_ps(-0x1.777a5cp-25f);

    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i twos = _mm256_set1_epi32(2);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    for (; number < eighthPoints; number++) {
        __m256 x = _mm256_load_ps(inPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m256 n_f = _mm256_round_ps(_mm256_mul_ps(x, two_over_pi),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i n = _mm256_cvtps_epi32(n_f);

        // r = x - n * (pi/2)
        __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n_f, pi_over_2_hi));
        r = _mm256_sub_ps(r, _mm256_mul_ps(n_f, pi_over_2_lo));

        // Evaluate both sin and cos polynomials
        __m256 sin_r = _mm256_sin_poly_avx2(r);
        __m256 cos_r = _mm256_cos_poly_avx2(r);

        // Quadrant selection
        __m256i n_and_1 = _mm256_and_si256(n, ones);
        __m256i n_and_2 = _mm256_and_si256(n, twos);
        __m256i n_plus_1_and_2 = _mm256_and_si256(_mm256_add_epi32(n, ones), twos);

        // For sin: swap when n&1, negate when n&2
        __m256 sin_swap = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_and_1, ones));
        __m256 sin_result = _mm256_blendv_ps(sin_r, cos_r, sin_swap);
        __m256 sin_neg = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_and_2, twos));
        sin_result = _mm256_xor_ps(sin_result, _mm256_and_ps(sin_neg, sign_bit));

        // For cos: swap when n&1, negate when (n+1)&2
        __m256 cos_result = _mm256_blendv_ps(cos_r, sin_r, sin_swap);
        __m256 cos_neg = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_plus_1_and_2, twos));
        cos_result = _mm256_xor_ps(cos_result, _mm256_and_ps(cos_neg, sign_bit));

        _mm256_store_ps(sinPtr, sin_result);
        _mm256_store_ps(cosPtr, cos_result);
        inPtr += 8;
        sinPtr += 8;
        cosPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *sinPtr++ = sinf(*inPtr);
        *cosPtr++ = cosf(*inPtr++);
    }
}

#endif /* LV_HAVE_AVX2 for aligned */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
#include <volk/volk_sse_intrinsics.h>

static inline void volk_32f_sincos_32f_x2_a_sse4_1(float* sinVector,
                                                   float* cosVector,
                                                   const float* inVector,
                                                   unsigned int num_points)
{
    float* sinPtr = sinVector;
    float* cosPtr = cosVector;
    const float* inPtr = inVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;

    // Constants for Cody-Waite argument reduction
    const __m128 two_over_pi = _mm_set1_ps(0x1.45f306p-1f);
    const __m128 pi_over_2_hi = _mm_set1_ps(0x1.921fb6p+0f);
    const __m128 pi_over_2_lo = _mm_set1_ps(-0x1.777a5cp-25f);

    const __m128i ones = _mm_set1_epi32(1);
    const __m128i twos = _mm_set1_epi32(2);
    const __m128 sign_bit = _mm_set1_ps(-0.0f);

    for (; number < quarterPoints; number++) {
        __m128 x = _mm_load_ps(inPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m128 n_f = _mm_round_ps(_mm_mul_ps(x, two_over_pi),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i n = _mm_cvtps_epi32(n_f);

        // r = x - n * (pi/2)
        __m128 r = _mm_sub_ps(x, _mm_mul_ps(n_f, pi_over_2_hi));
        r = _mm_sub_ps(r, _mm_mul_ps(n_f, pi_over_2_lo));

        // Evaluate both sin and cos polynomials
        __m128 sin_r = _mm_sin_poly_sse(r);
        __m128 cos_r = _mm_cos_poly_sse(r);

        // Quadrant selection
        __m128i n_and_1 = _mm_and_si128(n, ones);
        __m128i n_and_2 = _mm_and_si128(n, twos);
        __m128i n_plus_1_and_2 = _mm_and_si128(_mm_add_epi32(n, ones), twos);

        // For sin: swap when n&1, negate when n&2
        __m128 sin_swap = _mm_castsi128_ps(_mm_cmpeq_epi32(n_and_1, ones));
        __m128 sin_result = _mm_blendv_ps(sin_r, cos_r, sin_swap);
        __m128 sin_neg = _mm_castsi128_ps(_mm_cmpeq_epi32(n_and_2, twos));
        sin_result = _mm_xor_ps(sin_result, _mm_and_ps(sin_neg, sign_bit));

        // For cos: swap when n&1, negate when (n+1)&2
        __m128 cos_result = _mm_blendv_ps(cos_r, sin_r, sin_swap);
        __m128 cos_neg = _mm_castsi128_ps(_mm_cmpeq_epi32(n_plus_1_and_2, twos));
        cos_result = _mm_xor_ps(cos_result, _mm_and_ps(cos_neg, sign_bit));

        _mm_store_ps(sinPtr, sin_result);
        _mm_store_ps(cosPtr, cos_result);
        inPtr += 4;
        sinPtr += 4;
        cosPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *sinPtr++ = sinf(*inPtr);
        *cosPtr++ = cosf(*inPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for aligned */

#endif /* INCLUDED_volk_32f_sincos_32f_x2_a_H */


#ifndef INCLUDED_volk_32f_sincos_32f_x2_u_H
#define INCLUDED_volk_32f_sincos_32f_x2_u_H

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void volk_32f_sincos_32f_x2_u_avx512f(float* sinVector,
                                                    float* cosVector,
                                                    const float* inVector,
                                                    unsigned int num_points)
{
    float* sinPtr = sinVector;
    float* cosPtr = cosVector;
    const float* inPtr = inVector;

    unsigned int number = 0;
    unsigned int sixteenPoints = num_points / 16;

    // Constants for Cody-Waite argument reduction
    const __m512 two_over_pi = _mm512_set1_ps(0x1.45f306p-1f);
    const __m512 pi_over_2_hi = _mm512_set1_ps(0x1.921fb6p+0f);
    const __m512 pi_over_2_lo = _mm512_set1_ps(-0x1.777a5cp-25f);

    const __m512i ones = _mm512_set1_epi32(1);
    const __m512i twos = _mm512_set1_epi32(2);
    const __m512i sign_bit = _mm512_set1_epi32(0x80000000);

    for (; number < sixteenPoints; number++) {
        __m512 x = _mm512_loadu_ps(inPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m512 n_f = _mm512_roundscale_ps(_mm512_mul_ps(x, two_over_pi),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512i n = _mm512_cvtps_epi32(n_f);

        // r = x - n * (pi/2)
        __m512 r = _mm512_fnmadd_ps(n_f, pi_over_2_hi, x);
        r = _mm512_fnmadd_ps(n_f, pi_over_2_lo, r);

        // Evaluate both sin and cos polynomials
        __m512 sin_r = _mm512_sin_poly_avx512(r);
        __m512 cos_r = _mm512_cos_poly_avx512(r);

        // Quadrant selection
        __m512i n_and_1 = _mm512_and_si512(n, ones);
        __m512i n_and_2 = _mm512_and_si512(n, twos);
        __m512i n_plus_1_and_2 = _mm512_and_si512(_mm512_add_epi32(n, ones), twos);

        // For sin: swap when n&1, negate when n&2
        __mmask16 sin_swap = _mm512_cmpeq_epi32_mask(n_and_1, ones);
        __m512 sin_result = _mm512_mask_blend_ps(sin_swap, sin_r, cos_r);
        __mmask16 sin_neg = _mm512_cmpeq_epi32_mask(n_and_2, twos);
        sin_result =
            _mm512_castsi512_ps(_mm512_mask_xor_epi32(_mm512_castps_si512(sin_result),
                                                      sin_neg,
                                                      _mm512_castps_si512(sin_result),
                                                      sign_bit));

        // For cos: swap when n&1, negate when (n+1)&2
        __mmask16 cos_swap = sin_swap;
        __m512 cos_result = _mm512_mask_blend_ps(cos_swap, cos_r, sin_r);
        __mmask16 cos_neg = _mm512_cmpeq_epi32_mask(n_plus_1_and_2, twos);
        cos_result =
            _mm512_castsi512_ps(_mm512_mask_xor_epi32(_mm512_castps_si512(cos_result),
                                                      cos_neg,
                                                      _mm512_castps_si512(cos_result),
                                                      sign_bit));

        _mm512_storeu_ps(sinPtr, sin_result);
        _mm512_storeu_ps(cosPtr, cos_result);
        inPtr += 16;
        sinPtr += 16;
        cosPtr += 16;
    }

    number = sixteenPoints * 16;
    for (; number < num_points; number++) {
        *sinPtr++ = sinf(*inPtr);
        *cosPtr++ = cosf(*inPtr++);
    }
}

#endif /* LV_HAVE_AVX512F for unaligned */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_32f_sincos_32f_x2_u_avx2_fma(float* sinVector,
                                                     float* cosVector,
                                                     const float* inVector,
                                                     unsigned int num_points)
{
    float* sinPtr = sinVector;
    float* cosPtr = cosVector;
    const float* inPtr = inVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;

    // Constants for Cody-Waite argument reduction
    const __m256 two_over_pi = _mm256_set1_ps(0x1.45f306p-1f);
    const __m256 pi_over_2_hi = _mm256_set1_ps(0x1.921fb6p+0f);
    const __m256 pi_over_2_lo = _mm256_set1_ps(-0x1.777a5cp-25f);

    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i twos = _mm256_set1_epi32(2);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    for (; number < eighthPoints; number++) {
        __m256 x = _mm256_loadu_ps(inPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m256 n_f = _mm256_round_ps(_mm256_mul_ps(x, two_over_pi),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i n = _mm256_cvtps_epi32(n_f);

        // r = x - n * (pi/2)
        __m256 r = _mm256_fnmadd_ps(n_f, pi_over_2_hi, x);
        r = _mm256_fnmadd_ps(n_f, pi_over_2_lo, r);

        // Evaluate both sin and cos polynomials
        __m256 sin_r = _mm256_sin_poly_avx2_fma(r);
        __m256 cos_r = _mm256_cos_poly_avx2_fma(r);

        // Quadrant selection
        __m256i n_and_1 = _mm256_and_si256(n, ones);
        __m256i n_and_2 = _mm256_and_si256(n, twos);
        __m256i n_plus_1_and_2 = _mm256_and_si256(_mm256_add_epi32(n, ones), twos);

        // For sin: swap when n&1, negate when n&2
        __m256 sin_swap = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_and_1, ones));
        __m256 sin_result = _mm256_blendv_ps(sin_r, cos_r, sin_swap);
        __m256 sin_neg = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_and_2, twos));
        sin_result = _mm256_xor_ps(sin_result, _mm256_and_ps(sin_neg, sign_bit));

        // For cos: swap when n&1, negate when (n+1)&2
        __m256 cos_result = _mm256_blendv_ps(cos_r, sin_r, sin_swap);
        __m256 cos_neg = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_plus_1_and_2, twos));
        cos_result = _mm256_xor_ps(cos_result, _mm256_and_ps(cos_neg, sign_bit));

        _mm256_storeu_ps(sinPtr, sin_result);
        _mm256_storeu_ps(cosPtr, cos_result);
        inPtr += 8;
        sinPtr += 8;
        cosPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *sinPtr++ = sinf(*inPtr);
        *cosPtr++ = cosf(*inPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for unaligned */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void volk_32f_sincos_32f_x2_u_avx2(float* sinVector,
                                                 float* cosVector,
                                                 const float* inVector,
                                                 unsigned int num_points)
{
    float* sinPtr = sinVector;
    float* cosPtr = cosVector;
    const float* inPtr = inVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;

    // Constants for Cody-Waite argument reduction
    const __m256 two_over_pi = _mm256_set1_ps(0x1.45f306p-1f);
    const __m256 pi_over_2_hi = _mm256_set1_ps(0x1.921fb6p+0f);
    const __m256 pi_over_2_lo = _mm256_set1_ps(-0x1.777a5cp-25f);

    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i twos = _mm256_set1_epi32(2);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    for (; number < eighthPoints; number++) {
        __m256 x = _mm256_loadu_ps(inPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m256 n_f = _mm256_round_ps(_mm256_mul_ps(x, two_over_pi),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i n = _mm256_cvtps_epi32(n_f);

        // r = x - n * (pi/2)
        __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n_f, pi_over_2_hi));
        r = _mm256_sub_ps(r, _mm256_mul_ps(n_f, pi_over_2_lo));

        // Evaluate both sin and cos polynomials
        __m256 sin_r = _mm256_sin_poly_avx2(r);
        __m256 cos_r = _mm256_cos_poly_avx2(r);

        // Quadrant selection
        __m256i n_and_1 = _mm256_and_si256(n, ones);
        __m256i n_and_2 = _mm256_and_si256(n, twos);
        __m256i n_plus_1_and_2 = _mm256_and_si256(_mm256_add_epi32(n, ones), twos);

        // For sin: swap when n&1, negate when n&2
        __m256 sin_swap = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_and_1, ones));
        __m256 sin_result = _mm256_blendv_ps(sin_r, cos_r, sin_swap);
        __m256 sin_neg = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_and_2, twos));
        sin_result = _mm256_xor_ps(sin_result, _mm256_and_ps(sin_neg, sign_bit));

        // For cos: swap when n&1, negate when (n+1)&2
        __m256 cos_result = _mm256_blendv_ps(cos_r, sin_r, sin_swap);
        __m256 cos_neg = _mm256_castsi256_ps(_mm256_cmpeq_epi32(n_plus_1_and_2, twos));
        cos_result = _mm256_xor_ps(cos_result, _mm256_and_ps(cos_neg, sign_bit));

        _mm256_storeu_ps(sinPtr, sin_result);
        _mm256_storeu_ps(cosPtr, cos_result);
        inPtr += 8;
        sinPtr += 8;
        cosPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *sinPtr++ = sinf(*inPtr);
        *cosPtr++ = cosf(*inPtr++);
    }
}

#endif /* LV_HAVE_AVX2 for unaligned */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
#include <volk/volk_sse_intrinsics.h>

static inline void volk_32f_sincos_32f_x2_u_sse4_1(float* sinVector,
                                                   float* cosVector,
                                                   const float* inVector,
                                                   unsigned int num_points)
{
    float* sinPtr = sinVector;
    float* cosPtr = cosVector;
    const float* inPtr = inVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;

    // Constants for Cody-Waite argument reduction
    const __m128 two_over_pi = _mm_set1_ps(0x1.45f306p-1f);
    const __m128 pi_over_2_hi = _mm_set1_ps(0x1.921fb6p+0f);
    const __m128 pi_over_2_lo = _mm_set1_ps(-0x1.777a5cp-25f);

    const __m128i ones = _mm_set1_epi32(1);
    const __m128i twos = _mm_set1_epi32(2);
    const __m128 sign_bit = _mm_set1_ps(-0.0f);

    for (; number < quarterPoints; number++) {
        __m128 x = _mm_loadu_ps(inPtr);

        // Argument reduction: n = round(x * 2/pi)
        __m128 n_f = _mm_round_ps(_mm_mul_ps(x, two_over_pi),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i n = _mm_cvtps_epi32(n_f);

        // r = x - n * (pi/2)
        __m128 r = _mm_sub_ps(x, _mm_mul_ps(n_f, pi_over_2_hi));
        r = _mm_sub_ps(r, _mm_mul_ps(n_f, pi_over_2_lo));

        // Evaluate both sin and cos polynomials
        __m128 sin_r = _mm_sin_poly_sse(r);
        __m128 cos_r = _mm_cos_poly_sse(r);

        // Quadrant selection
        __m128i n_and_1 = _mm_and_si128(n, ones);
        __m128i n_and_2 = _mm_and_si128(n, twos);
        __m128i n_plus_1_and_2 = _mm_and_si128(_mm_add_epi32(n, ones), twos);

        // For sin: swap when n&1, negate when n&2
        __m128 sin_swap = _mm_castsi128_ps(_mm_cmpeq_epi32(n_and_1, ones));
        __m128 sin_result = _mm_blendv_ps(sin_r, cos_r, sin_swap);
        __m128 sin_neg = _mm_castsi128_ps(_mm_cmpeq_epi32(n_and_2, twos));
        sin_result = _mm_xor_ps(sin_result, _mm_and_ps(sin_neg, sign_bit));

        // For cos: swap when n&1, negate when (n+1)&2
        __m128 cos_result = _mm_blendv_ps(cos_r, sin_r, sin_swap);
        __m128 cos_neg = _mm_castsi128_ps(_mm_cmpeq_epi32(n_plus_1_and_2, twos));
        cos_result = _mm_xor_ps(cos_result, _mm_and_ps(cos_neg, sign_bit));

        _mm_storeu_ps(sinPtr, sin_result);
        _mm_storeu_ps(cosPtr, cos_result);
        inPtr += 4;
        sinPtr += 4;
        cosPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *sinPtr++ = sinf(*inPtr);
        *cosPtr++ = cosf(*inPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for unaligned */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

/* NEON polynomial-based sincos with shared argument reduction */
static inline void volk_32f_sincos_32f_x2_neon(float* sinVector,
                                               float* cosVector,
                                               const float* inVector,
                                               unsigned int num_points)
{
    // Cody-Waite argument reduction: n = round(x * 2/pi), r = x - n * pi/2
    const float32x4_t two_over_pi = vdupq_n_f32(0x1.45f306p-1f);
    const float32x4_t pi_over_2_hi = vdupq_n_f32(0x1.921fb6p+0f);
    const float32x4_t pi_over_2_lo = vdupq_n_f32(-0x1.777a5cp-25f);

    const int32x4_t ones = vdupq_n_s32(1);
    const int32x4_t twos = vdupq_n_s32(2);
    const float32x4_t sign_bit = vdupq_n_f32(-0.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t neg_half = vdupq_n_f32(-0.5f);
    const float32x4_t fzeroes = vdupq_n_f32(0.0f);

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    for (; number < quarterPoints; number++) {
        float32x4_t x = vld1q_f32(inVector);
        inVector += 4;

        // n = round(x * 2/pi) - emulate round-to-nearest for ARMv7
        float32x4_t scaled = vmulq_f32(x, two_over_pi);
        uint32x4_t is_neg = vcltq_f32(scaled, fzeroes);
        float32x4_t adj = vbslq_f32(is_neg, neg_half, half);
        float32x4_t n_f = vcvtq_f32_s32(vcvtq_s32_f32(vaddq_f32(scaled, adj)));
        int32x4_t n = vcvtq_s32_f32(n_f);

        // r = x - n * (pi/2) using extended precision
        float32x4_t r = vmlsq_f32(x, n_f, pi_over_2_hi);
        r = vmlsq_f32(r, n_f, pi_over_2_lo);

        // Evaluate sin and cos polynomials (shared for both outputs)
        float32x4_t sin_r = _vsin_poly_f32(r);
        float32x4_t cos_r = _vcos_poly_f32(r);

        // Quadrant selection
        int32x4_t n_and_1 = vandq_s32(n, ones);
        int32x4_t n_and_2 = vandq_s32(n, twos);
        int32x4_t n_plus_1_and_2 = vandq_s32(vaddq_s32(n, ones), twos);

        uint32x4_t swap_mask = vceqq_s32(n_and_1, ones);

        // Sin result: use cos when n&1, negate when n&2
        float32x4_t sin_result = vbslq_f32(swap_mask, cos_r, sin_r);
        uint32x4_t sin_neg = vceqq_s32(n_and_2, twos);
        sin_result = vreinterpretq_f32_u32(
            veorq_u32(vreinterpretq_u32_f32(sin_result),
                      vandq_u32(sin_neg, vreinterpretq_u32_f32(sign_bit))));

        // Cos result: use sin when n&1, negate when (n+1)&2
        float32x4_t cos_result = vbslq_f32(swap_mask, sin_r, cos_r);
        uint32x4_t cos_neg = vceqq_s32(n_plus_1_and_2, twos);
        cos_result = vreinterpretq_f32_u32(
            veorq_u32(vreinterpretq_u32_f32(cos_result),
                      vandq_u32(cos_neg, vreinterpretq_u32_f32(sign_bit))));

        vst1q_f32(sinVector, sin_result);
        vst1q_f32(cosVector, cos_result);
        sinVector += 4;
        cosVector += 4;
    }

    for (number = quarterPoints * 4; number < num_points; number++) {
        *sinVector++ = sinf(*inVector);
        *cosVector++ = cosf(*inVector++);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

/* NEONv8 polynomial-based sincos with FMA and shared argument reduction */
static inline void volk_32f_sincos_32f_x2_neonv8(float* sinVector,
                                                 float* cosVector,
                                                 const float* inVector,
                                                 unsigned int num_points)
{
    // Cody-Waite argument reduction: n = round(x * 2/pi), r = x - n * pi/2
    const float32x4_t two_over_pi = vdupq_n_f32(0x1.45f306p-1f);
    const float32x4_t pi_over_2_hi = vdupq_n_f32(0x1.921fb6p+0f);
    const float32x4_t pi_over_2_lo = vdupq_n_f32(-0x1.777a5cp-25f);

    const int32x4_t ones = vdupq_n_s32(1);
    const int32x4_t twos = vdupq_n_s32(2);
    const float32x4_t sign_bit = vdupq_n_f32(-0.0f);

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    for (; number < quarterPoints; number++) {
        float32x4_t x = vld1q_f32(inVector);
        inVector += 4;

        // n = round(x * 2/pi) using ARMv8 vrndnq_f32
        float32x4_t n_f = vrndnq_f32(vmulq_f32(x, two_over_pi));
        int32x4_t n = vcvtq_s32_f32(n_f);

        // r = x - n * (pi/2) using FMA for extended precision
        float32x4_t r = vfmsq_f32(x, n_f, pi_over_2_hi);
        r = vfmsq_f32(r, n_f, pi_over_2_lo);

        // Evaluate sin and cos polynomials using FMA (shared for both outputs)
        float32x4_t sin_r = _vsin_poly_neonv8(r);
        float32x4_t cos_r = _vcos_poly_neonv8(r);

        // Quadrant selection
        int32x4_t n_and_1 = vandq_s32(n, ones);
        int32x4_t n_and_2 = vandq_s32(n, twos);
        int32x4_t n_plus_1_and_2 = vandq_s32(vaddq_s32(n, ones), twos);

        uint32x4_t swap_mask = vceqq_s32(n_and_1, ones);

        // Sin result: use cos when n&1, negate when n&2
        float32x4_t sin_result = vbslq_f32(swap_mask, cos_r, sin_r);
        uint32x4_t sin_neg = vceqq_s32(n_and_2, twos);
        sin_result = vreinterpretq_f32_u32(
            veorq_u32(vreinterpretq_u32_f32(sin_result),
                      vandq_u32(sin_neg, vreinterpretq_u32_f32(sign_bit))));

        // Cos result: use sin when n&1, negate when (n+1)&2
        float32x4_t cos_result = vbslq_f32(swap_mask, sin_r, cos_r);
        uint32x4_t cos_neg = vceqq_s32(n_plus_1_and_2, twos);
        cos_result = vreinterpretq_f32_u32(
            veorq_u32(vreinterpretq_u32_f32(cos_result),
                      vandq_u32(cos_neg, vreinterpretq_u32_f32(sign_bit))));

        vst1q_f32(sinVector, sin_result);
        vst1q_f32(cosVector, cos_result);
        sinVector += 4;
        cosVector += 4;
    }

    for (number = quarterPoints * 4; number < num_points; number++) {
        *sinVector++ = sinf(*inVector);
        *cosVector++ = cosf(*inVector++);
    }
}
#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32f_sincos_32f_x2_u_H */
