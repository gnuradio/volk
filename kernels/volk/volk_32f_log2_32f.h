/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 * Copyright 2025-2026 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_log2_32f
 *
 * \b Overview
 *
 * Computes base 2 log of input vector and stores results in output vector.
 *
 * Note that this implementation is not conforming to the IEEE FP standard, i.e.,
 * +-Inf outputs are mapped to +-127.0f and +-NaN input values are not supported.
 *
 * This kernel was adapted from Jose Fonseca's Fast SSE2 log implementation
 * https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL TUNGSTEN GRAPHICS AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * This is the MIT License (MIT)
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_log2_32f(float* bVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: the input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li bVector: The output vector.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       in[ii] = std::pow(2.f,((float)ii));
 *   }
 *
 *   volk_32f_log2_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_log2_32f_a_H
#define INCLUDED_volk_32f_log2_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_log2_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *bPtr++ = log2f_non_ieee(*aPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
#include <volk/volk_sse_intrinsics.h>

static inline void
volk_32f_log2_32f_u_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const __m128i exp_mask = _mm_set1_epi32(0x7f800000);
    const __m128i mant_mask = _mm_set1_epi32(0x007fffff);
    const __m128i one_bits = _mm_set1_epi32(0x3f800000);
    const __m128i exp_bias = _mm_set1_epi32(127);
    const __m128 one = _mm_set1_ps(1.0f);

    for (; number < quarterPoints; number++) {
        __m128 aVal = _mm_loadu_ps(aPtr);

        // Check for special values
        __m128 zero_mask = _mm_cmpeq_ps(aVal, _mm_setzero_ps());
        __m128 neg_mask = _mm_cmplt_ps(aVal, _mm_setzero_ps());
        __m128 inf_mask = _mm_cmpeq_ps(aVal, _mm_set1_ps(INFINITY));
        __m128 nan_mask = _mm_cmpunord_ps(aVal, aVal);
        __m128 invalid_mask = _mm_or_ps(neg_mask, nan_mask);

        __m128i aVal_i = _mm_castps_si128(aVal);

        // Extract exponent: (aVal_i & exp_mask) >> 23 - bias
        __m128i exp_i = _mm_srli_epi32(_mm_and_si128(aVal_i, exp_mask), 23);
        exp_i = _mm_sub_epi32(exp_i, exp_bias);
        __m128 exp_f = _mm_cvtepi32_ps(exp_i);

        // Extract mantissa as float in [1, 2)
        __m128 frac =
            _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(aVal_i, mant_mask), one_bits));

        // Evaluate degree-6 polynomial
        __m128 poly = _mm_log2_poly_sse(frac);

        // result = exp + poly * (frac - 1)
        __m128 bVal = _mm_add_ps(exp_f, _mm_mul_ps(poly, _mm_sub_ps(frac, one)));

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        bVal = _mm_blendv_ps(bVal, _mm_set1_ps(-127.0f), zero_mask);
        bVal = _mm_blendv_ps(bVal, _mm_set1_ps(127.0f), inf_mask);
        bVal = _mm_blendv_ps(bVal, _mm_set1_ps(NAN), invalid_mask);

        _mm_storeu_ps(bPtr, bVal);

        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_log2_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_SSE4_1 for unaligned */

#ifdef LV_HAVE_SSE4_1

static inline void
volk_32f_log2_32f_a_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const __m128i exp_mask = _mm_set1_epi32(0x7f800000);
    const __m128i mant_mask = _mm_set1_epi32(0x007fffff);
    const __m128i one_bits = _mm_set1_epi32(0x3f800000);
    const __m128i exp_bias = _mm_set1_epi32(127);
    const __m128 one = _mm_set1_ps(1.0f);

    for (; number < quarterPoints; number++) {
        __m128 aVal = _mm_load_ps(aPtr);

        // Check for special values
        __m128 zero_mask = _mm_cmpeq_ps(aVal, _mm_setzero_ps());
        __m128 neg_mask = _mm_cmplt_ps(aVal, _mm_setzero_ps());
        __m128 inf_mask = _mm_cmpeq_ps(aVal, _mm_set1_ps(INFINITY));
        __m128 nan_mask = _mm_cmpunord_ps(aVal, aVal);
        __m128 invalid_mask = _mm_or_ps(neg_mask, nan_mask);

        __m128i aVal_i = _mm_castps_si128(aVal);

        // Extract exponent: (aVal_i & exp_mask) >> 23 - bias
        __m128i exp_i = _mm_srli_epi32(_mm_and_si128(aVal_i, exp_mask), 23);
        exp_i = _mm_sub_epi32(exp_i, exp_bias);
        __m128 exp_f = _mm_cvtepi32_ps(exp_i);

        // Extract mantissa as float in [1, 2)
        __m128 frac =
            _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(aVal_i, mant_mask), one_bits));

        // Evaluate degree-6 polynomial
        __m128 poly = _mm_log2_poly_sse(frac);

        // result = exp + poly * (frac - 1)
        __m128 bVal = _mm_add_ps(exp_f, _mm_mul_ps(poly, _mm_sub_ps(frac, one)));

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        bVal = _mm_blendv_ps(bVal, _mm_set1_ps(-127.0f), zero_mask);
        bVal = _mm_blendv_ps(bVal, _mm_set1_ps(127.0f), inf_mask);
        bVal = _mm_blendv_ps(bVal, _mm_set1_ps(NAN), invalid_mask);

        _mm_store_ps(bPtr, bVal);

        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_log2_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void
volk_32f_log2_32f_u_avx2(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);
    const __m256i one_bits = _mm256_set1_epi32(0x3f800000);
    const __m256i exp_bias = _mm256_set1_epi32(127);
    const __m256 one = _mm256_set1_ps(1.0f);

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aPtr);

        // Check for special values
        __m256 zero_mask = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_EQ_OQ);
        __m256 neg_mask = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_LT_OQ);
        __m256 inf_mask = _mm256_cmp_ps(aVal, _mm256_set1_ps(INFINITY), _CMP_EQ_OQ);
        __m256 nan_mask = _mm256_cmp_ps(aVal, aVal, _CMP_UNORD_Q);
        __m256 invalid_mask = _mm256_or_ps(neg_mask, nan_mask);

        __m256i aVal_i = _mm256_castps_si256(aVal);

        // Extract exponent
        __m256i exp_i = _mm256_srli_epi32(_mm256_and_si256(aVal_i, exp_mask), 23);
        exp_i = _mm256_sub_epi32(exp_i, exp_bias);
        __m256 exp_f = _mm256_cvtepi32_ps(exp_i);

        // Extract mantissa as float in [1, 2)
        __m256 frac = _mm256_castsi256_ps(
            _mm256_or_si256(_mm256_and_si256(aVal_i, mant_mask), one_bits));

        // Evaluate degree-6 polynomial
        __m256 poly = _mm256_log2_poly_avx2(frac);

        // result = exp + poly * (frac - 1)
        __m256 bVal = _mm256_add_ps(exp_f, _mm256_mul_ps(poly, _mm256_sub_ps(frac, one)));

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        bVal = _mm256_blendv_ps(bVal, _mm256_set1_ps(-127.0f), zero_mask);
        bVal = _mm256_blendv_ps(bVal, _mm256_set1_ps(127.0f), inf_mask);
        bVal = _mm256_blendv_ps(bVal, _mm256_set1_ps(NAN), invalid_mask);

        _mm256_storeu_ps(bPtr, bVal);

        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_log2_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_AVX2 for unaligned */

#ifdef LV_HAVE_AVX2

static inline void
volk_32f_log2_32f_a_avx2(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);
    const __m256i one_bits = _mm256_set1_epi32(0x3f800000);
    const __m256i exp_bias = _mm256_set1_epi32(127);
    const __m256 one = _mm256_set1_ps(1.0f);

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aPtr);

        // Check for special values
        __m256 zero_mask = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_EQ_OQ);
        __m256 neg_mask = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_LT_OQ);
        __m256 inf_mask = _mm256_cmp_ps(aVal, _mm256_set1_ps(INFINITY), _CMP_EQ_OQ);
        __m256 nan_mask = _mm256_cmp_ps(aVal, aVal, _CMP_UNORD_Q);
        __m256 invalid_mask = _mm256_or_ps(neg_mask, nan_mask);

        __m256i aVal_i = _mm256_castps_si256(aVal);

        // Extract exponent
        __m256i exp_i = _mm256_srli_epi32(_mm256_and_si256(aVal_i, exp_mask), 23);
        exp_i = _mm256_sub_epi32(exp_i, exp_bias);
        __m256 exp_f = _mm256_cvtepi32_ps(exp_i);

        // Extract mantissa as float in [1, 2)
        __m256 frac = _mm256_castsi256_ps(
            _mm256_or_si256(_mm256_and_si256(aVal_i, mant_mask), one_bits));

        // Evaluate degree-6 polynomial
        __m256 poly = _mm256_log2_poly_avx2(frac);

        // result = exp + poly * (frac - 1)
        __m256 bVal = _mm256_add_ps(exp_f, _mm256_mul_ps(poly, _mm256_sub_ps(frac, one)));

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        bVal = _mm256_blendv_ps(bVal, _mm256_set1_ps(-127.0f), zero_mask);
        bVal = _mm256_blendv_ps(bVal, _mm256_set1_ps(127.0f), inf_mask);
        bVal = _mm256_blendv_ps(bVal, _mm256_set1_ps(NAN), invalid_mask);

        _mm256_store_ps(bPtr, bVal);

        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_log2_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_32f_log2_32f_u_avx2_fma(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);
    const __m256i one_bits = _mm256_set1_epi32(0x3f800000);
    const __m256i exp_bias = _mm256_set1_epi32(127);
    const __m256 one = _mm256_set1_ps(1.0f);

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aPtr);

        // Check for special values
        __m256 zero_mask = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_EQ_OQ);
        __m256 neg_mask = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_LT_OQ);
        __m256 inf_mask = _mm256_cmp_ps(aVal, _mm256_set1_ps(INFINITY), _CMP_EQ_OQ);
        __m256 nan_mask = _mm256_cmp_ps(aVal, aVal, _CMP_UNORD_Q);
        __m256 invalid_mask = _mm256_or_ps(neg_mask, nan_mask);

        __m256i aVal_i = _mm256_castps_si256(aVal);

        // Extract exponent
        __m256i exp_i = _mm256_srli_epi32(_mm256_and_si256(aVal_i, exp_mask), 23);
        exp_i = _mm256_sub_epi32(exp_i, exp_bias);
        __m256 exp_f = _mm256_cvtepi32_ps(exp_i);

        // Extract mantissa as float in [1, 2)
        __m256 frac = _mm256_castsi256_ps(
            _mm256_or_si256(_mm256_and_si256(aVal_i, mant_mask), one_bits));

        // Evaluate degree-6 polynomial with FMA
        __m256 poly = _mm256_log2_poly_avx2_fma(frac);

        // result = exp + poly * (frac - 1)
        __m256 bVal = _mm256_fmadd_ps(poly, _mm256_sub_ps(frac, one), exp_f);

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        bVal = _mm256_blendv_ps(bVal, _mm256_set1_ps(-127.0f), zero_mask);
        bVal = _mm256_blendv_ps(bVal, _mm256_set1_ps(127.0f), inf_mask);
        bVal = _mm256_blendv_ps(bVal, _mm256_set1_ps(NAN), invalid_mask);

        _mm256_storeu_ps(bPtr, bVal);

        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_log2_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for unaligned */

#if LV_HAVE_AVX2 && LV_HAVE_FMA

static inline void volk_32f_log2_32f_a_avx2_fma(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);
    const __m256i one_bits = _mm256_set1_epi32(0x3f800000);
    const __m256i exp_bias = _mm256_set1_epi32(127);
    const __m256 one = _mm256_set1_ps(1.0f);

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aPtr);

        // Check for special values
        __m256 zero_mask = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_EQ_OQ);
        __m256 neg_mask = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_LT_OQ);
        __m256 inf_mask = _mm256_cmp_ps(aVal, _mm256_set1_ps(INFINITY), _CMP_EQ_OQ);
        __m256 nan_mask = _mm256_cmp_ps(aVal, aVal, _CMP_UNORD_Q);
        __m256 invalid_mask = _mm256_or_ps(neg_mask, nan_mask);

        __m256i aVal_i = _mm256_castps_si256(aVal);

        // Extract exponent
        __m256i exp_i = _mm256_srli_epi32(_mm256_and_si256(aVal_i, exp_mask), 23);
        exp_i = _mm256_sub_epi32(exp_i, exp_bias);
        __m256 exp_f = _mm256_cvtepi32_ps(exp_i);

        // Extract mantissa as float in [1, 2)
        __m256 frac = _mm256_castsi256_ps(
            _mm256_or_si256(_mm256_and_si256(aVal_i, mant_mask), one_bits));

        // Evaluate degree-6 polynomial with FMA
        __m256 poly = _mm256_log2_poly_avx2_fma(frac);

        // result = exp + poly * (frac - 1)
        __m256 bVal = _mm256_fmadd_ps(poly, _mm256_sub_ps(frac, one), exp_f);

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        bVal = _mm256_blendv_ps(bVal, _mm256_set1_ps(-127.0f), zero_mask);
        bVal = _mm256_blendv_ps(bVal, _mm256_set1_ps(127.0f), inf_mask);
        bVal = _mm256_blendv_ps(bVal, _mm256_set1_ps(NAN), invalid_mask);

        _mm256_store_ps(bPtr, bVal);

        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_log2_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void
volk_32f_log2_32f_u_avx512(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const __m512i exp_mask = _mm512_set1_epi32(0x7f800000);
    const __m512i mant_mask = _mm512_set1_epi32(0x007fffff);
    const __m512i one_bits = _mm512_set1_epi32(0x3f800000);
    const __m512i exp_bias = _mm512_set1_epi32(127);
    const __m512 one = _mm512_set1_ps(1.0f);

    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_loadu_ps(aPtr);

        // Check for special values
        __mmask16 zero_mask = _mm512_cmp_ps_mask(aVal, _mm512_setzero_ps(), _CMP_EQ_OQ);
        __mmask16 neg_mask = _mm512_cmp_ps_mask(aVal, _mm512_setzero_ps(), _CMP_LT_OQ);
        __mmask16 inf_mask =
            _mm512_cmp_ps_mask(aVal, _mm512_set1_ps(INFINITY), _CMP_EQ_OQ);
        __mmask16 nan_mask = _mm512_cmp_ps_mask(aVal, aVal, _CMP_UNORD_Q);
        __mmask16 invalid_mask = _kor_mask16(neg_mask, nan_mask);

        __m512i aVal_i = _mm512_castps_si512(aVal);

        // Extract exponent
        __m512i exp_i = _mm512_srli_epi32(_mm512_and_si512(aVal_i, exp_mask), 23);
        exp_i = _mm512_sub_epi32(exp_i, exp_bias);
        __m512 exp_f = _mm512_cvtepi32_ps(exp_i);

        // Extract mantissa as float in [1, 2)
        __m512 frac = _mm512_castsi512_ps(
            _mm512_or_si512(_mm512_and_si512(aVal_i, mant_mask), one_bits));

        // Evaluate degree-6 polynomial with FMA
        __m512 poly = _mm512_log2_poly_avx512(frac);

        // result = exp + poly * (frac - 1)
        __m512 bVal = _mm512_fmadd_ps(poly, _mm512_sub_ps(frac, one), exp_f);

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        bVal = _mm512_mask_blend_ps(zero_mask, bVal, _mm512_set1_ps(-127.0f));
        bVal = _mm512_mask_blend_ps(inf_mask, bVal, _mm512_set1_ps(127.0f));
        bVal = _mm512_mask_blend_ps(invalid_mask, bVal, _mm512_set1_ps(NAN));

        _mm512_storeu_ps(bPtr, bVal);

        aPtr += 16;
        bPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_log2_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_AVX512F for unaligned */

#ifdef LV_HAVE_AVX512F

static inline void
volk_32f_log2_32f_a_avx512(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const __m512i exp_mask = _mm512_set1_epi32(0x7f800000);
    const __m512i mant_mask = _mm512_set1_epi32(0x007fffff);
    const __m512i one_bits = _mm512_set1_epi32(0x3f800000);
    const __m512i exp_bias = _mm512_set1_epi32(127);
    const __m512 one = _mm512_set1_ps(1.0f);

    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_load_ps(aPtr);

        // Check for special values
        __mmask16 zero_mask = _mm512_cmp_ps_mask(aVal, _mm512_setzero_ps(), _CMP_EQ_OQ);
        __mmask16 neg_mask = _mm512_cmp_ps_mask(aVal, _mm512_setzero_ps(), _CMP_LT_OQ);
        __mmask16 inf_mask =
            _mm512_cmp_ps_mask(aVal, _mm512_set1_ps(INFINITY), _CMP_EQ_OQ);
        __mmask16 nan_mask = _mm512_cmp_ps_mask(aVal, aVal, _CMP_UNORD_Q);
        __mmask16 invalid_mask = _kor_mask16(neg_mask, nan_mask);

        __m512i aVal_i = _mm512_castps_si512(aVal);

        // Extract exponent
        __m512i exp_i = _mm512_srli_epi32(_mm512_and_si512(aVal_i, exp_mask), 23);
        exp_i = _mm512_sub_epi32(exp_i, exp_bias);
        __m512 exp_f = _mm512_cvtepi32_ps(exp_i);

        // Extract mantissa as float in [1, 2)
        __m512 frac = _mm512_castsi512_ps(
            _mm512_or_si512(_mm512_and_si512(aVal_i, mant_mask), one_bits));

        // Evaluate degree-6 polynomial with FMA
        __m512 poly = _mm512_log2_poly_avx512(frac);

        // result = exp + poly * (frac - 1)
        __m512 bVal = _mm512_fmadd_ps(poly, _mm512_sub_ps(frac, one), exp_f);

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        bVal = _mm512_mask_blend_ps(zero_mask, bVal, _mm512_set1_ps(-127.0f));
        bVal = _mm512_mask_blend_ps(inf_mask, bVal, _mm512_set1_ps(127.0f));
        bVal = _mm512_mask_blend_ps(invalid_mask, bVal, _mm512_set1_ps(NAN));

        _mm512_store_ps(bPtr, bVal);

        aPtr += 16;
        bPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_log2_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_AVX512F */

#if LV_HAVE_AVX512F && LV_HAVE_AVX512DQ

static inline void volk_32f_log2_32f_u_avx512dq(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const __m512i exp_mask = _mm512_set1_epi32(0x7f800000);
    const __m512i mant_mask = _mm512_set1_epi32(0x007fffff);
    const __m512i one_bits = _mm512_set1_epi32(0x3f800000);
    const __m512i exp_bias = _mm512_set1_epi32(127);
    const __m512 one = _mm512_set1_ps(1.0f);

    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_loadu_ps(aPtr);

        // Use fpclass for special value detection (AVX512DQ feature)
        // 0x01 = QNaN, 0x02 = +0, 0x04 = -0, 0x08 = +Inf, 0x10 = -Inf, 0x80 = SNaN
        __mmask16 nan_mask = _mm512_fpclass_ps_mask(aVal, 0x81);  // NaN (QNaN | SNaN)
        __mmask16 zero_mask = _mm512_fpclass_ps_mask(aVal, 0x06); // Zero (+0 | -0)
        __mmask16 inf_mask = _mm512_fpclass_ps_mask(aVal, 0x08);  // +Inf only
        __mmask16 neg_mask = _mm512_cmp_ps_mask(aVal, _mm512_setzero_ps(), _CMP_LT_OQ);
        __mmask16 invalid_mask = _kor_mask16(nan_mask, neg_mask); // neg or NaN -> NaN

        __m512i aVal_i = _mm512_castps_si512(aVal);

        // Extract exponent
        __m512i exp_i = _mm512_srli_epi32(_mm512_and_si512(aVal_i, exp_mask), 23);
        exp_i = _mm512_sub_epi32(exp_i, exp_bias);
        __m512 exp_f = _mm512_cvtepi32_ps(exp_i);

        // Extract mantissa as float in [1, 2)
        __m512 frac = _mm512_castsi512_ps(
            _mm512_or_si512(_mm512_and_si512(aVal_i, mant_mask), one_bits));

        // Evaluate degree-6 polynomial with FMA
        __m512 poly = _mm512_log2_poly_avx512(frac);

        // result = exp + poly * (frac - 1)
        __m512 bVal = _mm512_fmadd_ps(poly, _mm512_sub_ps(frac, one), exp_f);

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        bVal = _mm512_mask_blend_ps(zero_mask, bVal, _mm512_set1_ps(-127.0f));
        bVal = _mm512_mask_blend_ps(inf_mask, bVal, _mm512_set1_ps(127.0f));
        bVal = _mm512_mask_blend_ps(invalid_mask, bVal, _mm512_set1_ps(NAN));

        _mm512_storeu_ps(bPtr, bVal);

        aPtr += 16;
        bPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_log2_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_AVX512F && LV_HAVE_AVX512DQ for unaligned */

#if LV_HAVE_AVX512F && LV_HAVE_AVX512DQ

static inline void volk_32f_log2_32f_a_avx512dq(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const __m512i exp_mask = _mm512_set1_epi32(0x7f800000);
    const __m512i mant_mask = _mm512_set1_epi32(0x007fffff);
    const __m512i one_bits = _mm512_set1_epi32(0x3f800000);
    const __m512i exp_bias = _mm512_set1_epi32(127);
    const __m512 one = _mm512_set1_ps(1.0f);

    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_load_ps(aPtr);

        // Use fpclass for special value detection (AVX512DQ feature)
        // 0x01 = QNaN, 0x02 = +0, 0x04 = -0, 0x08 = +Inf, 0x10 = -Inf, 0x80 = SNaN
        __mmask16 nan_mask = _mm512_fpclass_ps_mask(aVal, 0x81);  // NaN (QNaN | SNaN)
        __mmask16 zero_mask = _mm512_fpclass_ps_mask(aVal, 0x06); // Zero (+0 | -0)
        __mmask16 inf_mask = _mm512_fpclass_ps_mask(aVal, 0x08);  // +Inf only
        __mmask16 neg_mask = _mm512_cmp_ps_mask(aVal, _mm512_setzero_ps(), _CMP_LT_OQ);
        __mmask16 invalid_mask = _kor_mask16(nan_mask, neg_mask); // neg or NaN -> NaN

        __m512i aVal_i = _mm512_castps_si512(aVal);

        // Extract exponent
        __m512i exp_i = _mm512_srli_epi32(_mm512_and_si512(aVal_i, exp_mask), 23);
        exp_i = _mm512_sub_epi32(exp_i, exp_bias);
        __m512 exp_f = _mm512_cvtepi32_ps(exp_i);

        // Extract mantissa as float in [1, 2)
        __m512 frac = _mm512_castsi512_ps(
            _mm512_or_si512(_mm512_and_si512(aVal_i, mant_mask), one_bits));

        // Evaluate degree-6 polynomial with FMA
        __m512 poly = _mm512_log2_poly_avx512(frac);

        // result = exp + poly * (frac - 1)
        __m512 bVal = _mm512_fmadd_ps(poly, _mm512_sub_ps(frac, one), exp_f);

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        bVal = _mm512_mask_blend_ps(zero_mask, bVal, _mm512_set1_ps(-127.0f));
        bVal = _mm512_mask_blend_ps(inf_mask, bVal, _mm512_set1_ps(127.0f));
        bVal = _mm512_mask_blend_ps(invalid_mask, bVal, _mm512_set1_ps(NAN));

        _mm512_store_ps(bPtr, bVal);

        aPtr += 16;
        bPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_log2_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_AVX512F && LV_HAVE_AVX512DQ */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void
volk_32f_log2_32f_neon(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number;
    const unsigned int quarterPoints = num_points / 4;

    const int32x4_t exp_mask = vdupq_n_s32(0x7f800000);
    const int32x4_t mant_mask = vdupq_n_s32(0x007fffff);
    const int32x4_t one_bits = vdupq_n_s32(0x3f800000);
    const int32x4_t exp_bias = vdupq_n_s32(127);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t inf_val = vdupq_n_f32(INFINITY);
    const float32x4_t nan_val = vdupq_n_f32(NAN);
    const float32x4_t neg_inf_out = vdupq_n_f32(-127.0f);
    const float32x4_t pos_inf_out = vdupq_n_f32(127.0f);

    for (number = 0; number < quarterPoints; ++number) {
        float32x4_t aVal = vld1q_f32(aPtr);

        // Check for special values
        uint32x4_t neg_mask = vcltq_f32(aVal, zero);
        uint32x4_t zero_mask = vceqq_f32(aVal, zero);
        uint32x4_t inf_mask = vceqq_f32(aVal, inf_val);
        uint32x4_t nan_mask = vmvnq_u32(vceqq_f32(aVal, aVal));
        uint32x4_t invalid_mask = vorrq_u32(neg_mask, nan_mask);

        int32x4_t aVal_i = vreinterpretq_s32_f32(aVal);

        // Extract exponent
        int32x4_t exp_i = vshrq_n_s32(vandq_s32(aVal_i, exp_mask), 23);
        exp_i = vsubq_s32(exp_i, exp_bias);
        float32x4_t exp_f = vcvtq_f32_s32(exp_i);

        // Extract mantissa as float in [1, 2)
        int32x4_t frac_i = vorrq_s32(vandq_s32(aVal_i, mant_mask), one_bits);
        float32x4_t frac = vreinterpretq_f32_s32(frac_i);

        // Evaluate degree-6 polynomial
        float32x4_t poly = _vlog2_poly_f32(frac);

        // result = exp + poly * (frac - 1)
        float32x4_t bVal = vaddq_f32(exp_f, vmulq_f32(poly, vsubq_f32(frac, one)));

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        bVal = vbslq_f32(zero_mask, neg_inf_out, bVal);
        bVal = vbslq_f32(inf_mask, pos_inf_out, bVal);
        bVal = vbslq_f32(invalid_mask, nan_val, bVal);

        vst1q_f32(bPtr, bVal);

        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_log2_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8

static inline void
volk_32f_log2_32f_neonv8(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number;
    const unsigned int quarterPoints = num_points / 4;

    const int32x4_t exp_mask = vdupq_n_s32(0x7f800000);
    const int32x4_t mant_mask = vdupq_n_s32(0x007fffff);
    const int32x4_t one_bits = vdupq_n_s32(0x3f800000);
    const int32x4_t exp_bias = vdupq_n_s32(127);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t inf_val = vdupq_n_f32(INFINITY);
    const float32x4_t nan_val = vdupq_n_f32(NAN);
    const float32x4_t neg_inf_out = vdupq_n_f32(-127.0f);
    const float32x4_t pos_inf_out = vdupq_n_f32(127.0f);

    for (number = 0; number < quarterPoints; ++number) {
        float32x4_t aVal = vld1q_f32(aPtr);

        // Check for special values
        uint32x4_t neg_mask = vcltq_f32(aVal, zero);
        uint32x4_t zero_mask = vceqq_f32(aVal, zero);
        uint32x4_t inf_mask = vceqq_f32(aVal, inf_val);
        uint32x4_t nan_mask = vmvnq_u32(vceqq_f32(aVal, aVal));
        uint32x4_t invalid_mask = vorrq_u32(neg_mask, nan_mask);

        int32x4_t aVal_i = vreinterpretq_s32_f32(aVal);

        // Extract exponent
        int32x4_t exp_i = vshrq_n_s32(vandq_s32(aVal_i, exp_mask), 23);
        exp_i = vsubq_s32(exp_i, exp_bias);
        float32x4_t exp_f = vcvtq_f32_s32(exp_i);

        // Extract mantissa as float in [1, 2)
        int32x4_t frac_i = vorrq_s32(vandq_s32(aVal_i, mant_mask), one_bits);
        float32x4_t frac = vreinterpretq_f32_s32(frac_i);

        // Evaluate degree-6 polynomial with FMA
        float32x4_t poly = _vlog2_poly_neonv8(frac);

        // result = exp + poly * (frac - 1)
        float32x4_t bVal = vfmaq_f32(exp_f, poly, vsubq_f32(frac, one));

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        bVal = vbslq_f32(zero_mask, neg_inf_out, bVal);
        bVal = vbslq_f32(inf_mask, pos_inf_out, bVal);
        bVal = vbslq_f32(invalid_mask, nan_val, bVal);

        vst1q_f32(bPtr, bVal);

        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_log2_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_NEONV8 */


#endif /* INCLUDED_volk_32f_log2_32f_a_H */

#ifndef INCLUDED_volk_32f_log2_32f_u_H
#define INCLUDED_volk_32f_log2_32f_u_H

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void
volk_32f_log2_32f_rvv(float* bVector, const float* aVector, unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

    const vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    const vint32m2_t one_bits = __riscv_vreinterpret_i32m2(one);
    const vint32m2_t mant_mask = __riscv_vmv_v_x_i32m2(0x7FFFFF, vlmax);
    const vint32m2_t exp_bias = __riscv_vmv_v_x_i32m2(127, vlmax);

    const vfloat32m2_t zero = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    const vfloat32m2_t inf_val = __riscv_vfmv_v_f_f32m2(INFINITY, vlmax);
    const vfloat32m2_t nan_val = __riscv_vfmv_v_f_f32m2(NAN, vlmax);
    const vfloat32m2_t neg_inf_out = __riscv_vfmv_v_f_f32m2(-127.0f, vlmax);
    const vfloat32m2_t pos_inf_out = __riscv_vfmv_v_f_f32m2(127.0f, vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v = __riscv_vle32_v_f32m2(aVector, vl);

        // Check for special values
        vbool16_t zero_mask = __riscv_vmfeq(v, zero, vl);
        vbool16_t neg_mask = __riscv_vmflt(v, zero, vl);
        vbool16_t inf_mask = __riscv_vmfeq(v, inf_val, vl);
        vbool16_t nan_mask = __riscv_vmfne(v, v, vl);
        vbool16_t invalid_mask = __riscv_vmor(neg_mask, nan_mask, vl);

        vfloat32m2_t a = __riscv_vfabs(v, vl);
        vfloat32m2_t exp_f = __riscv_vfcvt_f(
            __riscv_vsub(
                __riscv_vsra(__riscv_vreinterpret_i32m2(a), 23, vl), exp_bias, vl),
            vl);
        vfloat32m2_t frac = __riscv_vreinterpret_f32m2(__riscv_vor(
            __riscv_vand(__riscv_vreinterpret_i32m2(v), mant_mask, vl), one_bits, vl));

        // Evaluate degree-6 polynomial with FMA
        vfloat32m2_t poly = __riscv_vlog2_poly_f32m2(frac, vl, vlmax);

        // result = exp + poly * (frac - 1)
        vfloat32m2_t result =
            __riscv_vfmacc(exp_f, poly, __riscv_vfsub(frac, one, vl), vl);

        // Replace special values: zero → -127, inf → 127, neg/NaN → NaN
        result = __riscv_vmerge(result, neg_inf_out, zero_mask, vl);
        result = __riscv_vmerge(result, pos_inf_out, inf_mask, vl);
        result = __riscv_vmerge(result, nan_val, invalid_mask, vl);

        __riscv_vse32(bVector, result, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_log2_32f_u_H */
