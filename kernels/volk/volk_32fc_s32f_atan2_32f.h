/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 * Copyright 2023, 2024 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32f_atan2_32f
 *
 * \b Overview
 *
 * Computes the arctan for each value in a complex vector and applies
 * a normalization factor.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_atan2_32f(float* outputVector, const lv_32fc_t* complexVector,
 * const float normalizeFactor, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: The byte-aligned input vector containing interleaved IQ data (I = cos,
 * Q = sin). \li normalizeFactor: The atan results are divided by this normalization
 * factor. \li num_points: The number of complex values in \p inputVector.
 *
 * \b Outputs
 * \li outputVector: The vector where the results will be stored.
 *
 * \b Example
 * Calculate the arctangent of points around the unit circle.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float scale = 1.f; // we want unit circle
 *
 *   for(unsigned int ii = 0; ii < N/2; ++ii){
 *       // Generate points around the unit circle
 *       float real = -4.f * ((float)ii / (float)N) + 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *       in[ii+N/2] = lv_cmake(-real, -imag);
 *   }
 *
 *   volk_32fc_s32f_atan2_32f(out, in, scale, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("atan2(%1.2f, %1.2f) = %1.2f\n",
 *           lv_cimag(in[ii]), lv_creal(in[ii]), out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32f_atan2_32f_a_H
#define INCLUDED_volk_32fc_s32f_atan2_32f_a_H

#include <math.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_32fc_s32f_atan2_32f_generic(float* outputVector,
                                                    const lv_32fc_t* inputVector,
                                                    const float normalizeFactor,
                                                    unsigned int num_points)
{
    float* outPtr = outputVector;
    const float* inPtr = (float*)inputVector;
    const float invNormalizeFactor = 1.f / normalizeFactor;

    for (unsigned int number = 0; number < num_points; number++) {
        const float real = *inPtr++;
        const float imag = *inPtr++;
        *outPtr++ = atan2f(imag, real) * invNormalizeFactor;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_GENERIC
#include <volk/volk_common.h>
static inline void volk_32fc_s32f_atan2_32f_polynomial(float* outputVector,
                                                       const lv_32fc_t* inputVector,
                                                       const float normalizeFactor,
                                                       unsigned int num_points)
{
    float* outPtr = outputVector;
    const float* inPtr = (float*)inputVector;
    const float invNormalizeFactor = 1.f / normalizeFactor;

    for (unsigned int number = 0; number < num_points; number++) {
        const float x = *inPtr++;
        const float y = *inPtr++;
        *outPtr++ = volk_atan2(y, x) * invNormalizeFactor;
    }
}
#endif /* LV_HAVE_GENERIC */

#if LV_HAVE_AVX512F && LV_HAVE_AVX512DQ
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>
static inline void volk_32fc_s32f_atan2_32f_a_avx512dq(float* outputVector,
                                                       const lv_32fc_t* complexVector,
                                                       const float normalizeFactor,
                                                       unsigned int num_points)
{
    const float* in = (float*)complexVector;
    float* out = (float*)outputVector;

    const float invNormalizeFactor = 1.f / normalizeFactor;
    const __m512 vinvNormalizeFactor = _mm512_set1_ps(invNormalizeFactor);
    const __m512 pi = _mm512_set1_ps(0x1.921fb6p1f);
    const __m512 pi_2 = _mm512_set1_ps(0x1.921fb6p0f);
    const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));

    unsigned int number = 0;
    const unsigned int sixteenth_points = num_points / 16;
    for (; number < sixteenth_points; number++) {
        __m512 z1 = _mm512_load_ps(in);
        in += 16;
        __m512 z2 = _mm512_load_ps(in);
        in += 16;

        __m512 x = _mm512_real(z1, z2);
        __m512 y = _mm512_imag(z1, z2);

        // Detect NaN in original inputs before division
        __mmask16 input_nan_mask = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q) |
                                   _mm512_cmp_ps_mask(y, y, _CMP_UNORD_Q);

        // Handle infinity cases per IEEE 754
        const __m512 zero = _mm512_setzero_ps();
        const __m512 pi_4 = _mm512_set1_ps(0x1.921fb6p-1f);      // π/4
        const __m512 three_pi_4 = _mm512_set1_ps(0x1.2d97c8p1f); // 3π/4

        __mmask16 y_inf_mask = _mm512_fpclass_ps_mask(y, 0x18); // ±inf
        __mmask16 x_inf_mask = _mm512_fpclass_ps_mask(x, 0x18); // ±inf
        __mmask16 x_pos_mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OS);

        // Build infinity result
        __m512 inf_result = zero;
        // Both infinite: ±π/4 or ±3π/4
        __mmask16 both_inf = y_inf_mask & x_inf_mask;
        __m512 both_inf_result = _mm512_mask_blend_ps(x_pos_mask, three_pi_4, pi_4);
        both_inf_result = _mm512_or_ps(both_inf_result, _mm512_and_ps(y, sign_mask));
        inf_result = _mm512_mask_blend_ps(both_inf, inf_result, both_inf_result);

        // y infinite, x finite: ±π/2
        __mmask16 y_inf_only = y_inf_mask & ~x_inf_mask;
        __m512 y_inf_result = _mm512_or_ps(pi_2, _mm512_and_ps(y, sign_mask));
        inf_result = _mm512_mask_blend_ps(y_inf_only, inf_result, y_inf_result);

        // x infinite, y finite: 0 or ±π
        __mmask16 x_inf_only = x_inf_mask & ~y_inf_mask;
        __m512 x_inf_result =
            _mm512_mask_blend_ps(x_pos_mask,
                                 _mm512_or_ps(pi, _mm512_and_ps(y, sign_mask)),
                                 _mm512_or_ps(zero, _mm512_and_ps(y, sign_mask)));
        inf_result = _mm512_mask_blend_ps(x_inf_only, inf_result, x_inf_result);

        __mmask16 any_inf_mask = y_inf_mask | x_inf_mask;

        __mmask16 swap_mask = _mm512_cmp_ps_mask(
            _mm512_and_ps(y, abs_mask), _mm512_and_ps(x, abs_mask), _CMP_GT_OS);
        __m512 numerator = _mm512_mask_blend_ps(swap_mask, y, x);
        __m512 denominator = _mm512_mask_blend_ps(swap_mask, x, y);
        __m512 input = _mm512_div_ps(numerator, denominator);

        // Only handle NaN from division (0/0, inf/inf), not from NaN inputs
        // Replace with numerator to preserve sign (e.g., atan2(-0, 0) = -0)
        __mmask16 div_nan_mask =
            _mm512_cmp_ps_mask(input, input, _CMP_UNORD_Q) & ~input_nan_mask;
        input = _mm512_mask_blend_ps(div_nan_mask, input, numerator);
        __m512 result = _mm512_arctan_poly_avx512(input);

        input =
            _mm512_sub_ps(_mm512_or_ps(pi_2, _mm512_and_ps(input, sign_mask)), result);
        result = _mm512_mask_blend_ps(swap_mask, result, input);

        __m512 x_sign_mask =
            _mm512_castsi512_ps(_mm512_srai_epi32(_mm512_castps_si512(x), 31));

        result = _mm512_add_ps(
            _mm512_and_ps(_mm512_xor_ps(pi, _mm512_and_ps(sign_mask, y)), x_sign_mask),
            result);

        // Select infinity result or normal result
        result = _mm512_mask_blend_ps(any_inf_mask, result, inf_result);

        result = _mm512_mul_ps(result, vinvNormalizeFactor);

        _mm512_store_ps(out, result);
        out += 16;
    }

    number = sixteenth_points * 16;
    volk_32fc_s32f_atan2_32f_polynomial(
        out, complexVector + number, normalizeFactor, num_points - number);
}
#endif /* LV_HAVE_AVX512F && LV_HAVE_AVX512DQ for aligned */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
static inline void volk_32fc_s32f_atan2_32f_a_avx2_fma(float* outputVector,
                                                       const lv_32fc_t* complexVector,
                                                       const float normalizeFactor,
                                                       unsigned int num_points)
{
    const float* in = (float*)complexVector;
    float* out = (float*)outputVector;

    const float invNormalizeFactor = 1.f / normalizeFactor;
    const __m256 vinvNormalizeFactor = _mm256_set1_ps(invNormalizeFactor);
    const __m256 pi = _mm256_set1_ps(0x1.921fb6p1f);
    const __m256 pi_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;
    for (; number < eighth_points; number++) {
        __m256 z1 = _mm256_load_ps(in);
        in += 8;
        __m256 z2 = _mm256_load_ps(in);
        in += 8;

        __m256 x = _mm256_real(z1, z2);
        __m256 y = _mm256_imag(z1, z2);

        // Detect NaN in original inputs before division
        __m256 input_nan_mask = _mm256_or_ps(_mm256_cmp_ps(x, x, _CMP_UNORD_Q),
                                             _mm256_cmp_ps(y, y, _CMP_UNORD_Q));

        // Handle infinity cases per IEEE 754
        const __m256 zero = _mm256_setzero_ps();
        const __m256 inf = _mm256_set1_ps(HUGE_VALF);
        const __m256 pi_4 = _mm256_set1_ps(0x1.921fb6p-1f);      // π/4
        const __m256 three_pi_4 = _mm256_set1_ps(0x1.2d97c8p1f); // 3π/4

        __m256 y_abs = _mm256_and_ps(y, abs_mask);
        __m256 x_abs = _mm256_and_ps(x, abs_mask);
        __m256 y_inf_mask = _mm256_cmp_ps(y_abs, inf, _CMP_EQ_OQ); // |y| == inf
        __m256 x_inf_mask = _mm256_cmp_ps(x_abs, inf, _CMP_EQ_OQ); // |x| == inf
        __m256 x_pos_mask = _mm256_cmp_ps(x, zero, _CMP_GT_OS);

        // Build infinity result
        __m256 inf_result = zero;
        // Both infinite: ±π/4 or ±3π/4
        __m256 both_inf = _mm256_and_ps(y_inf_mask, x_inf_mask);
        __m256 both_inf_result = _mm256_blendv_ps(three_pi_4, pi_4, x_pos_mask);
        both_inf_result = _mm256_or_ps(both_inf_result, _mm256_and_ps(y, sign_mask));
        inf_result = _mm256_blendv_ps(inf_result, both_inf_result, both_inf);

        // y infinite, x finite: ±π/2
        __m256 y_inf_only = _mm256_andnot_ps(x_inf_mask, y_inf_mask);
        __m256 y_inf_result = _mm256_or_ps(pi_2, _mm256_and_ps(y, sign_mask));
        inf_result = _mm256_blendv_ps(inf_result, y_inf_result, y_inf_only);

        // x infinite, y finite: 0 or ±π
        __m256 x_inf_only = _mm256_andnot_ps(y_inf_mask, x_inf_mask);
        __m256 x_inf_result =
            _mm256_blendv_ps(_mm256_or_ps(pi, _mm256_and_ps(y, sign_mask)),
                             _mm256_or_ps(zero, _mm256_and_ps(y, sign_mask)),
                             x_pos_mask);
        inf_result = _mm256_blendv_ps(inf_result, x_inf_result, x_inf_only);

        __m256 any_inf_mask = _mm256_or_ps(y_inf_mask, x_inf_mask);

        __m256 swap_mask = _mm256_cmp_ps(
            _mm256_and_ps(y, abs_mask), _mm256_and_ps(x, abs_mask), _CMP_GT_OS);
        __m256 numerator = _mm256_blendv_ps(y, x, swap_mask);
        __m256 denominator = _mm256_blendv_ps(x, y, swap_mask);
        __m256 input = _mm256_div_ps(numerator, denominator);

        // Only handle NaN from division (0/0, inf/inf), not from NaN inputs
        // Replace with numerator to preserve sign (e.g., atan2(-0, 0) = -0)
        __m256 div_nan_mask =
            _mm256_andnot_ps(input_nan_mask, _mm256_cmp_ps(input, input, _CMP_UNORD_Q));
        input = _mm256_blendv_ps(input, numerator, div_nan_mask);
        __m256 result = _mm256_arctan_poly_avx2_fma(input);

        input =
            _mm256_sub_ps(_mm256_or_ps(pi_2, _mm256_and_ps(input, sign_mask)), result);
        result = _mm256_blendv_ps(result, input, swap_mask);

        __m256 x_sign_mask =
            _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(x), 31));

        result = _mm256_add_ps(
            _mm256_and_ps(_mm256_xor_ps(pi, _mm256_and_ps(sign_mask, y)), x_sign_mask),
            result);

        // Select infinity result or normal result
        result = _mm256_blendv_ps(result, inf_result, any_inf_mask);

        result = _mm256_mul_ps(result, vinvNormalizeFactor);

        _mm256_store_ps(out, result);
        out += 8;
    }

    number = eighth_points * 8;
    volk_32fc_s32f_atan2_32f_polynomial(
        out, (lv_32fc_t*)in, normalizeFactor, num_points - number);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for aligned */

#if LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>
static inline void volk_32fc_s32f_atan2_32f_a_avx2(float* outputVector,
                                                   const lv_32fc_t* complexVector,
                                                   const float normalizeFactor,
                                                   unsigned int num_points)
{
    const float* in = (float*)complexVector;
    float* out = (float*)outputVector;

    const float invNormalizeFactor = 1.f / normalizeFactor;
    const __m256 vinvNormalizeFactor = _mm256_set1_ps(invNormalizeFactor);
    const __m256 pi = _mm256_set1_ps(0x1.921fb6p1f);
    const __m256 pi_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;
    for (; number < eighth_points; number++) {
        __m256 z1 = _mm256_load_ps(in);
        in += 8;
        __m256 z2 = _mm256_load_ps(in);
        in += 8;

        __m256 x = _mm256_real(z1, z2);
        __m256 y = _mm256_imag(z1, z2);

        // Detect NaN in original inputs before division
        __m256 input_nan_mask = _mm256_or_ps(_mm256_cmp_ps(x, x, _CMP_UNORD_Q),
                                             _mm256_cmp_ps(y, y, _CMP_UNORD_Q));

        // Handle infinity cases per IEEE 754
        const __m256 zero = _mm256_setzero_ps();
        const __m256 inf = _mm256_set1_ps(HUGE_VALF);
        const __m256 pi_4 = _mm256_set1_ps(0x1.921fb6p-1f);      // π/4
        const __m256 three_pi_4 = _mm256_set1_ps(0x1.2d97c8p1f); // 3π/4

        __m256 y_abs = _mm256_and_ps(y, abs_mask);
        __m256 x_abs = _mm256_and_ps(x, abs_mask);
        __m256 y_inf_mask = _mm256_cmp_ps(y_abs, inf, _CMP_EQ_OQ); // |y| == inf
        __m256 x_inf_mask = _mm256_cmp_ps(x_abs, inf, _CMP_EQ_OQ); // |x| == inf
        __m256 x_pos_mask = _mm256_cmp_ps(x, zero, _CMP_GT_OS);

        // Build infinity result
        __m256 inf_result = zero;
        // Both infinite: ±π/4 or ±3π/4
        __m256 both_inf = _mm256_and_ps(y_inf_mask, x_inf_mask);
        __m256 both_inf_result = _mm256_blendv_ps(three_pi_4, pi_4, x_pos_mask);
        both_inf_result = _mm256_or_ps(both_inf_result, _mm256_and_ps(y, sign_mask));
        inf_result = _mm256_blendv_ps(inf_result, both_inf_result, both_inf);

        // y infinite, x finite: ±π/2
        __m256 y_inf_only = _mm256_andnot_ps(x_inf_mask, y_inf_mask);
        __m256 y_inf_result = _mm256_or_ps(pi_2, _mm256_and_ps(y, sign_mask));
        inf_result = _mm256_blendv_ps(inf_result, y_inf_result, y_inf_only);

        // x infinite, y finite: 0 or ±π
        __m256 x_inf_only = _mm256_andnot_ps(y_inf_mask, x_inf_mask);
        __m256 x_inf_result =
            _mm256_blendv_ps(_mm256_or_ps(pi, _mm256_and_ps(y, sign_mask)),
                             _mm256_or_ps(zero, _mm256_and_ps(y, sign_mask)),
                             x_pos_mask);
        inf_result = _mm256_blendv_ps(inf_result, x_inf_result, x_inf_only);

        __m256 any_inf_mask = _mm256_or_ps(y_inf_mask, x_inf_mask);

        __m256 swap_mask = _mm256_cmp_ps(
            _mm256_and_ps(y, abs_mask), _mm256_and_ps(x, abs_mask), _CMP_GT_OS);
        __m256 numerator = _mm256_blendv_ps(y, x, swap_mask);
        __m256 denominator = _mm256_blendv_ps(x, y, swap_mask);
        __m256 input = _mm256_div_ps(numerator, denominator);

        // Only handle NaN from division (0/0, inf/inf), not from NaN inputs
        // Replace with numerator to preserve sign (e.g., atan2(-0, 0) = -0)
        __m256 div_nan_mask =
            _mm256_andnot_ps(input_nan_mask, _mm256_cmp_ps(input, input, _CMP_UNORD_Q));
        input = _mm256_blendv_ps(input, numerator, div_nan_mask);
        __m256 result = _mm256_arctan_poly_avx(input);

        input =
            _mm256_sub_ps(_mm256_or_ps(pi_2, _mm256_and_ps(input, sign_mask)), result);
        result = _mm256_blendv_ps(result, input, swap_mask);

        __m256 x_sign_mask =
            _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(x), 31));

        result = _mm256_add_ps(
            _mm256_and_ps(_mm256_xor_ps(pi, _mm256_and_ps(sign_mask, y)), x_sign_mask),
            result);

        // Select infinity result or normal result
        result = _mm256_blendv_ps(result, inf_result, any_inf_mask);

        result = _mm256_mul_ps(result, vinvNormalizeFactor);

        _mm256_store_ps(out, result);
        out += 8;
    }

    number = eighth_points * 8;
    volk_32fc_s32f_atan2_32f_polynomial(
        out, (lv_32fc_t*)in, normalizeFactor, num_points - number);
}
#endif /* LV_HAVE_AVX2 for aligned */
#endif /* INCLUDED_volk_32fc_s32f_atan2_32f_a_H */

#ifndef INCLUDED_volk_32fc_s32f_atan2_32f_u_H
#define INCLUDED_volk_32fc_s32f_atan2_32f_u_H

#if LV_HAVE_AVX512F && LV_HAVE_AVX512DQ
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>
static inline void volk_32fc_s32f_atan2_32f_u_avx512dq(float* outputVector,
                                                       const lv_32fc_t* complexVector,
                                                       const float normalizeFactor,
                                                       unsigned int num_points)
{
    const float* in = (float*)complexVector;
    float* out = (float*)outputVector;

    const float invNormalizeFactor = 1.f / normalizeFactor;
    const __m512 vinvNormalizeFactor = _mm512_set1_ps(invNormalizeFactor);
    const __m512 pi = _mm512_set1_ps(0x1.921fb6p1f);
    const __m512 pi_2 = _mm512_set1_ps(0x1.921fb6p0f);
    const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));

    const unsigned int sixteenth_points = num_points / 16;

    for (unsigned int number = 0; number < sixteenth_points; number++) {
        __m512 z1 = _mm512_loadu_ps(in);
        in += 16;
        __m512 z2 = _mm512_loadu_ps(in);
        in += 16;

        __m512 x = _mm512_real(z1, z2);
        __m512 y = _mm512_imag(z1, z2);

        // Detect NaN in original inputs before division
        __mmask16 input_nan_mask = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q) |
                                   _mm512_cmp_ps_mask(y, y, _CMP_UNORD_Q);

        // Handle infinity cases per IEEE 754
        const __m512 zero = _mm512_setzero_ps();
        const __m512 pi_4 = _mm512_set1_ps(0x1.921fb6p-1f);      // π/4
        const __m512 three_pi_4 = _mm512_set1_ps(0x1.2d97c8p1f); // 3π/4

        __mmask16 y_inf_mask = _mm512_fpclass_ps_mask(y, 0x18); // ±inf
        __mmask16 x_inf_mask = _mm512_fpclass_ps_mask(x, 0x18); // ±inf
        __mmask16 x_pos_mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OS);

        // Build infinity result
        __m512 inf_result = zero;
        // Both infinite: ±π/4 or ±3π/4
        __mmask16 both_inf = y_inf_mask & x_inf_mask;
        __m512 both_inf_result = _mm512_mask_blend_ps(x_pos_mask, three_pi_4, pi_4);
        both_inf_result = _mm512_or_ps(both_inf_result, _mm512_and_ps(y, sign_mask));
        inf_result = _mm512_mask_blend_ps(both_inf, inf_result, both_inf_result);

        // y infinite, x finite: ±π/2
        __mmask16 y_inf_only = y_inf_mask & ~x_inf_mask;
        __m512 y_inf_result = _mm512_or_ps(pi_2, _mm512_and_ps(y, sign_mask));
        inf_result = _mm512_mask_blend_ps(y_inf_only, inf_result, y_inf_result);

        // x infinite, y finite: 0 or ±π
        __mmask16 x_inf_only = x_inf_mask & ~y_inf_mask;
        __m512 x_inf_result =
            _mm512_mask_blend_ps(x_pos_mask,
                                 _mm512_or_ps(pi, _mm512_and_ps(y, sign_mask)),
                                 _mm512_or_ps(zero, _mm512_and_ps(y, sign_mask)));
        inf_result = _mm512_mask_blend_ps(x_inf_only, inf_result, x_inf_result);

        __mmask16 any_inf_mask = y_inf_mask | x_inf_mask;

        __mmask16 swap_mask = _mm512_cmp_ps_mask(
            _mm512_and_ps(y, abs_mask), _mm512_and_ps(x, abs_mask), _CMP_GT_OS);
        __m512 numerator = _mm512_mask_blend_ps(swap_mask, y, x);
        __m512 denominator = _mm512_mask_blend_ps(swap_mask, x, y);
        __m512 input = _mm512_div_ps(numerator, denominator);

        // Only handle NaN from division (0/0, inf/inf), not from NaN inputs
        // Replace with numerator to preserve sign (e.g., atan2(-0, 0) = -0)
        __mmask16 div_nan_mask =
            _mm512_cmp_ps_mask(input, input, _CMP_UNORD_Q) & ~input_nan_mask;
        input = _mm512_mask_blend_ps(div_nan_mask, input, numerator);
        __m512 result = _mm512_arctan_poly_avx512(input);

        input =
            _mm512_sub_ps(_mm512_or_ps(pi_2, _mm512_and_ps(input, sign_mask)), result);
        result = _mm512_mask_blend_ps(swap_mask, result, input);

        __m512 x_sign_mask =
            _mm512_castsi512_ps(_mm512_srai_epi32(_mm512_castps_si512(x), 31));

        result = _mm512_add_ps(
            _mm512_and_ps(_mm512_xor_ps(pi, _mm512_and_ps(sign_mask, y)), x_sign_mask),
            result);

        // Select infinity result or normal result
        result = _mm512_mask_blend_ps(any_inf_mask, result, inf_result);

        result = _mm512_mul_ps(result, vinvNormalizeFactor);

        _mm512_storeu_ps(out, result);
        out += 16;
    }

    unsigned int number = sixteenth_points * 16;
    volk_32fc_s32f_atan2_32f_polynomial(
        out, complexVector + number, normalizeFactor, num_points - number);
}
#endif /* LV_HAVE_AVX512F && LV_HAVE_AVX512DQ for unaligned */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
static inline void volk_32fc_s32f_atan2_32f_u_avx2_fma(float* outputVector,
                                                       const lv_32fc_t* complexVector,
                                                       const float normalizeFactor,
                                                       unsigned int num_points)
{
    const float* in = (float*)complexVector;
    float* out = (float*)outputVector;

    const float invNormalizeFactor = 1.f / normalizeFactor;
    const __m256 vinvNormalizeFactor = _mm256_set1_ps(invNormalizeFactor);
    const __m256 pi = _mm256_set1_ps(0x1.921fb6p1f);
    const __m256 pi_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;
    for (; number < eighth_points; number++) {
        __m256 z1 = _mm256_loadu_ps(in);
        in += 8;
        __m256 z2 = _mm256_loadu_ps(in);
        in += 8;

        __m256 x = _mm256_real(z1, z2);
        __m256 y = _mm256_imag(z1, z2);

        // Detect NaN in original inputs before division
        __m256 input_nan_mask = _mm256_or_ps(_mm256_cmp_ps(x, x, _CMP_UNORD_Q),
                                             _mm256_cmp_ps(y, y, _CMP_UNORD_Q));

        // Handle infinity cases per IEEE 754
        const __m256 zero = _mm256_setzero_ps();
        const __m256 inf = _mm256_set1_ps(HUGE_VALF);
        const __m256 pi_4 = _mm256_set1_ps(0x1.921fb6p-1f);      // π/4
        const __m256 three_pi_4 = _mm256_set1_ps(0x1.2d97c8p1f); // 3π/4

        __m256 y_abs = _mm256_and_ps(y, abs_mask);
        __m256 x_abs = _mm256_and_ps(x, abs_mask);
        __m256 y_inf_mask = _mm256_cmp_ps(y_abs, inf, _CMP_EQ_OQ); // |y| == inf
        __m256 x_inf_mask = _mm256_cmp_ps(x_abs, inf, _CMP_EQ_OQ); // |x| == inf
        __m256 x_pos_mask = _mm256_cmp_ps(x, zero, _CMP_GT_OS);

        // Build infinity result
        __m256 inf_result = zero;
        // Both infinite: ±π/4 or ±3π/4
        __m256 both_inf = _mm256_and_ps(y_inf_mask, x_inf_mask);
        __m256 both_inf_result = _mm256_blendv_ps(three_pi_4, pi_4, x_pos_mask);
        both_inf_result = _mm256_or_ps(both_inf_result, _mm256_and_ps(y, sign_mask));
        inf_result = _mm256_blendv_ps(inf_result, both_inf_result, both_inf);

        // y infinite, x finite: ±π/2
        __m256 y_inf_only = _mm256_andnot_ps(x_inf_mask, y_inf_mask);
        __m256 y_inf_result = _mm256_or_ps(pi_2, _mm256_and_ps(y, sign_mask));
        inf_result = _mm256_blendv_ps(inf_result, y_inf_result, y_inf_only);

        // x infinite, y finite: 0 or ±π
        __m256 x_inf_only = _mm256_andnot_ps(y_inf_mask, x_inf_mask);
        __m256 x_inf_result =
            _mm256_blendv_ps(_mm256_or_ps(pi, _mm256_and_ps(y, sign_mask)),
                             _mm256_or_ps(zero, _mm256_and_ps(y, sign_mask)),
                             x_pos_mask);
        inf_result = _mm256_blendv_ps(inf_result, x_inf_result, x_inf_only);

        __m256 any_inf_mask = _mm256_or_ps(y_inf_mask, x_inf_mask);

        __m256 swap_mask = _mm256_cmp_ps(
            _mm256_and_ps(y, abs_mask), _mm256_and_ps(x, abs_mask), _CMP_GT_OS);
        __m256 numerator = _mm256_blendv_ps(y, x, swap_mask);
        __m256 denominator = _mm256_blendv_ps(x, y, swap_mask);
        __m256 input = _mm256_div_ps(numerator, denominator);

        // Only handle NaN from division (0/0, inf/inf), not from NaN inputs
        // Replace with numerator to preserve sign (e.g., atan2(-0, 0) = -0)
        __m256 div_nan_mask =
            _mm256_andnot_ps(input_nan_mask, _mm256_cmp_ps(input, input, _CMP_UNORD_Q));
        input = _mm256_blendv_ps(input, numerator, div_nan_mask);
        __m256 result = _mm256_arctan_poly_avx2_fma(input);

        input =
            _mm256_sub_ps(_mm256_or_ps(pi_2, _mm256_and_ps(input, sign_mask)), result);
        result = _mm256_blendv_ps(result, input, swap_mask);

        __m256 x_sign_mask =
            _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(x), 31));

        result = _mm256_add_ps(
            _mm256_and_ps(_mm256_xor_ps(pi, _mm256_and_ps(sign_mask, y)), x_sign_mask),
            result);

        // Select infinity result or normal result
        result = _mm256_blendv_ps(result, inf_result, any_inf_mask);

        result = _mm256_mul_ps(result, vinvNormalizeFactor);

        _mm256_storeu_ps(out, result);
        out += 8;
    }

    number = eighth_points * 8;
    volk_32fc_s32f_atan2_32f_polynomial(
        out, (lv_32fc_t*)in, normalizeFactor, num_points - number);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for unaligned */

#if LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>
static inline void volk_32fc_s32f_atan2_32f_u_avx2(float* outputVector,
                                                   const lv_32fc_t* complexVector,
                                                   const float normalizeFactor,
                                                   unsigned int num_points)
{
    const float* in = (float*)complexVector;
    float* out = (float*)outputVector;

    const float invNormalizeFactor = 1.f / normalizeFactor;
    const __m256 vinvNormalizeFactor = _mm256_set1_ps(invNormalizeFactor);
    const __m256 pi = _mm256_set1_ps(0x1.921fb6p1f);
    const __m256 pi_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;
    for (; number < eighth_points; number++) {
        __m256 z1 = _mm256_loadu_ps(in);
        in += 8;
        __m256 z2 = _mm256_loadu_ps(in);
        in += 8;

        __m256 x = _mm256_real(z1, z2);
        __m256 y = _mm256_imag(z1, z2);

        // Detect NaN in original inputs before division
        __m256 input_nan_mask = _mm256_or_ps(_mm256_cmp_ps(x, x, _CMP_UNORD_Q),
                                             _mm256_cmp_ps(y, y, _CMP_UNORD_Q));

        // Handle infinity cases per IEEE 754
        const __m256 zero = _mm256_setzero_ps();
        const __m256 inf = _mm256_set1_ps(HUGE_VALF);
        const __m256 pi_4 = _mm256_set1_ps(0x1.921fb6p-1f);      // π/4
        const __m256 three_pi_4 = _mm256_set1_ps(0x1.2d97c8p1f); // 3π/4

        __m256 y_abs = _mm256_and_ps(y, abs_mask);
        __m256 x_abs = _mm256_and_ps(x, abs_mask);
        __m256 y_inf_mask = _mm256_cmp_ps(y_abs, inf, _CMP_EQ_OQ); // |y| == inf
        __m256 x_inf_mask = _mm256_cmp_ps(x_abs, inf, _CMP_EQ_OQ); // |x| == inf
        __m256 x_pos_mask = _mm256_cmp_ps(x, zero, _CMP_GT_OS);

        // Build infinity result
        __m256 inf_result = zero;
        // Both infinite: ±π/4 or ±3π/4
        __m256 both_inf = _mm256_and_ps(y_inf_mask, x_inf_mask);
        __m256 both_inf_result = _mm256_blendv_ps(three_pi_4, pi_4, x_pos_mask);
        both_inf_result = _mm256_or_ps(both_inf_result, _mm256_and_ps(y, sign_mask));
        inf_result = _mm256_blendv_ps(inf_result, both_inf_result, both_inf);

        // y infinite, x finite: ±π/2
        __m256 y_inf_only = _mm256_andnot_ps(x_inf_mask, y_inf_mask);
        __m256 y_inf_result = _mm256_or_ps(pi_2, _mm256_and_ps(y, sign_mask));
        inf_result = _mm256_blendv_ps(inf_result, y_inf_result, y_inf_only);

        // x infinite, y finite: 0 or ±π
        __m256 x_inf_only = _mm256_andnot_ps(y_inf_mask, x_inf_mask);
        __m256 x_inf_result =
            _mm256_blendv_ps(_mm256_or_ps(pi, _mm256_and_ps(y, sign_mask)),
                             _mm256_or_ps(zero, _mm256_and_ps(y, sign_mask)),
                             x_pos_mask);
        inf_result = _mm256_blendv_ps(inf_result, x_inf_result, x_inf_only);

        __m256 any_inf_mask = _mm256_or_ps(y_inf_mask, x_inf_mask);

        __m256 swap_mask = _mm256_cmp_ps(
            _mm256_and_ps(y, abs_mask), _mm256_and_ps(x, abs_mask), _CMP_GT_OS);
        __m256 numerator = _mm256_blendv_ps(y, x, swap_mask);
        __m256 denominator = _mm256_blendv_ps(x, y, swap_mask);
        __m256 input = _mm256_div_ps(numerator, denominator);

        // Only handle NaN from division (0/0, inf/inf), not from NaN inputs
        // Replace with numerator to preserve sign (e.g., atan2(-0, 0) = -0)
        __m256 div_nan_mask =
            _mm256_andnot_ps(input_nan_mask, _mm256_cmp_ps(input, input, _CMP_UNORD_Q));
        input = _mm256_blendv_ps(input, numerator, div_nan_mask);
        __m256 result = _mm256_arctan_poly_avx(input);

        input =
            _mm256_sub_ps(_mm256_or_ps(pi_2, _mm256_and_ps(input, sign_mask)), result);
        result = _mm256_blendv_ps(result, input, swap_mask);

        __m256 x_sign_mask =
            _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(x), 31));

        result = _mm256_add_ps(
            _mm256_and_ps(_mm256_xor_ps(pi, _mm256_and_ps(sign_mask, y)), x_sign_mask),
            result);

        // Select infinity result or normal result
        result = _mm256_blendv_ps(result, inf_result, any_inf_mask);

        result = _mm256_mul_ps(result, vinvNormalizeFactor);

        _mm256_storeu_ps(out, result);
        out += 8;
    }

    number = eighth_points * 8;
    volk_32fc_s32f_atan2_32f_polynomial(
        out, (lv_32fc_t*)in, normalizeFactor, num_points - number);
}
#endif /* LV_HAVE_AVX2 for unaligned */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_32fc_s32f_atan2_32f_rvv(float* outputVector,
                                                const lv_32fc_t* inputVector,
                                                const float normalizeFactor,
                                                unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

    const vfloat32m2_t norm = __riscv_vfmv_v_f_f32m2(1 / normalizeFactor, vlmax);
    const vfloat32m2_t cpi = __riscv_vfmv_v_f_f32m2(3.1415927f, vlmax);
    const vfloat32m2_t cpio2 = __riscv_vfmv_v_f_f32m2(1.5707964f, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(+0x1.ffffeap-1f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(-0x1.55437p-2f, vlmax);
    const vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(+0x1.972be6p-3f, vlmax);
    const vfloat32m2_t c7 = __riscv_vfmv_v_f_f32m2(-0x1.1436ap-3f, vlmax);
    const vfloat32m2_t c9 = __riscv_vfmv_v_f_f32m2(+0x1.5785aap-4f, vlmax);
    const vfloat32m2_t c11 = __riscv_vfmv_v_f_f32m2(-0x1.2f3004p-5f, vlmax);
    const vfloat32m2_t c13 = __riscv_vfmv_v_f_f32m2(+0x1.01a37cp-7f, vlmax);

    const vfloat32m2_t zero = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    const vfloat32m2_t inf = __riscv_vfmv_v_f_f32m2(HUGE_VALF, vlmax);
    const vfloat32m2_t pi_4 = __riscv_vfmv_v_f_f32m2(0x1.921fb6p-1f, vlmax);      // π/4
    const vfloat32m2_t three_pi_4 = __riscv_vfmv_v_f_f32m2(0x1.2d97c8p1f, vlmax); // 3π/4

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vuint64m4_t v = __riscv_vle64_v_u64m4((const uint64_t*)inputVector, vl);
        vfloat32m2_t vr = __riscv_vreinterpret_f32m2(__riscv_vnsrl(v, 0, vl));
        vfloat32m2_t vi = __riscv_vreinterpret_f32m2(__riscv_vnsrl(v, 32, vl));

        // Detect NaN in original inputs before division
        vbool16_t input_nan_mask =
            __riscv_vmor(__riscv_vmfne(vr, vr, vl), __riscv_vmfne(vi, vi, vl), vl);

        // Handle infinity cases per IEEE 754
        vfloat32m2_t vr_abs = __riscv_vfabs(vr, vl);
        vfloat32m2_t vi_abs = __riscv_vfabs(vi, vl);
        vbool16_t vr_inf_mask = __riscv_vmfeq(vr_abs, inf, vl); // |vr| == inf
        vbool16_t vi_inf_mask = __riscv_vmfeq(vi_abs, inf, vl); // |vi| == inf
        vbool16_t vr_pos_mask = __riscv_vmfgt(vr, zero, vl);

        // Build infinity result
        vfloat32m2_t inf_result = zero;
        // Both infinite: ±π/4 or ±3π/4
        vbool16_t both_inf = __riscv_vmand(vi_inf_mask, vr_inf_mask, vl);
        vfloat32m2_t both_inf_result = __riscv_vmerge(three_pi_4, pi_4, vr_pos_mask, vl);
        both_inf_result = __riscv_vfsgnj(both_inf_result, vi, vl); // Copy sign from vi
        inf_result = __riscv_vmerge(inf_result, both_inf_result, both_inf, vl);

        // vi infinite, vr finite: ±π/2
        vbool16_t vi_inf_only = __riscv_vmandn(vi_inf_mask, vr_inf_mask, vl);
        vfloat32m2_t vi_inf_result = __riscv_vfsgnj(cpio2, vi, vl); // π/2 with sign of vi
        inf_result = __riscv_vmerge(inf_result, vi_inf_result, vi_inf_only, vl);

        // vr infinite, vi finite: 0 or ±π
        vbool16_t vr_inf_only = __riscv_vmandn(vr_inf_mask, vi_inf_mask, vl);
        vfloat32m2_t vr_inf_result =
            __riscv_vmerge(__riscv_vfsgnj(cpi, vi, vl),  // π with sign of vi
                           __riscv_vfsgnj(zero, vi, vl), // 0 with sign of vi
                           vr_pos_mask,
                           vl);
        inf_result = __riscv_vmerge(inf_result, vr_inf_result, vr_inf_only, vl);

        vbool16_t any_inf_mask = __riscv_vmor(vi_inf_mask, vr_inf_mask, vl);

        vbool16_t mswap = __riscv_vmfgt(vi_abs, vr_abs, vl);
        vfloat32m2_t numerator = __riscv_vmerge(vi, vr, mswap, vl);
        vfloat32m2_t denominator = __riscv_vmerge(vr, vi, mswap, vl);
        vfloat32m2_t x = __riscv_vfdiv(numerator, denominator, vl);

        // Only handle NaN from division (0/0, inf/inf), not from NaN inputs
        // Replace with numerator to preserve sign (e.g., atan2(-0, 0) = -0)
        vbool16_t x_nan_mask = __riscv_vmfne(x, x, vl);
        // div_nan_mask = x_nan_mask & ~input_nan_mask (vmandn computes vs2 & ~vs1)
        vbool16_t div_nan_mask = __riscv_vmandn(x_nan_mask, input_nan_mask, vl);
        x = __riscv_vmerge(x, numerator, div_nan_mask, vl);

        vfloat32m2_t xx = __riscv_vfmul(x, x, vl);
        vfloat32m2_t p = c13;
        p = __riscv_vfmadd(p, xx, c11, vl);
        p = __riscv_vfmadd(p, xx, c9, vl);
        p = __riscv_vfmadd(p, xx, c7, vl);
        p = __riscv_vfmadd(p, xx, c5, vl);
        p = __riscv_vfmadd(p, xx, c3, vl);
        p = __riscv_vfmadd(p, xx, c1, vl);
        p = __riscv_vfmul(p, x, vl);

        x = __riscv_vfsub(__riscv_vfsgnj(cpio2, x, vl), p, vl);
        p = __riscv_vmerge(p, x, mswap, vl);
        p = __riscv_vfadd_mu(
            RISCV_VMFLTZ(32m2, vr, vl), p, p, __riscv_vfsgnjx(cpi, vi, vl), vl);

        // Select infinity result or normal result
        p = __riscv_vmerge(p, inf_result, any_inf_mask, vl);

        __riscv_vse32(outputVector, __riscv_vfmul(p, norm, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_32fc_s32f_atan2_32f_rvvseg(float* outputVector,
                                                   const lv_32fc_t* inputVector,
                                                   const float normalizeFactor,
                                                   unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

    const vfloat32m2_t norm = __riscv_vfmv_v_f_f32m2(1 / normalizeFactor, vlmax);
    const vfloat32m2_t cpi = __riscv_vfmv_v_f_f32m2(3.1415927f, vlmax);
    const vfloat32m2_t cpio2 = __riscv_vfmv_v_f_f32m2(1.5707964f, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(+0x1.ffffeap-1f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(-0x1.55437p-2f, vlmax);
    const vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(+0x1.972be6p-3f, vlmax);
    const vfloat32m2_t c7 = __riscv_vfmv_v_f_f32m2(-0x1.1436ap-3f, vlmax);
    const vfloat32m2_t c9 = __riscv_vfmv_v_f_f32m2(+0x1.5785aap-4f, vlmax);
    const vfloat32m2_t c11 = __riscv_vfmv_v_f_f32m2(-0x1.2f3004p-5f, vlmax);
    const vfloat32m2_t c13 = __riscv_vfmv_v_f_f32m2(+0x1.01a37cp-7f, vlmax);

    const vfloat32m2_t zero = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    const vfloat32m2_t inf = __riscv_vfmv_v_f_f32m2(HUGE_VALF, vlmax);
    const vfloat32m2_t pi_4 = __riscv_vfmv_v_f_f32m2(0x1.921fb6p-1f, vlmax);      // π/4
    const vfloat32m2_t three_pi_4 = __riscv_vfmv_v_f_f32m2(0x1.2d97c8p1f, vlmax); // 3π/4

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2x2_t v = __riscv_vlseg2e32_v_f32m2x2((const float*)inputVector, vl);
        vfloat32m2_t vr = __riscv_vget_f32m2(v, 0), vi = __riscv_vget_f32m2(v, 1);

        // Detect NaN in original inputs before division
        vbool16_t input_nan_mask =
            __riscv_vmor(__riscv_vmfne(vr, vr, vl), __riscv_vmfne(vi, vi, vl), vl);

        // Handle infinity cases per IEEE 754
        vfloat32m2_t vr_abs = __riscv_vfabs(vr, vl);
        vfloat32m2_t vi_abs = __riscv_vfabs(vi, vl);
        vbool16_t vr_inf_mask = __riscv_vmfeq(vr_abs, inf, vl); // |vr| == inf
        vbool16_t vi_inf_mask = __riscv_vmfeq(vi_abs, inf, vl); // |vi| == inf
        vbool16_t vr_pos_mask = __riscv_vmfgt(vr, zero, vl);

        // Build infinity result
        vfloat32m2_t inf_result = zero;
        // Both infinite: ±π/4 or ±3π/4
        vbool16_t both_inf = __riscv_vmand(vi_inf_mask, vr_inf_mask, vl);
        vfloat32m2_t both_inf_result = __riscv_vmerge(three_pi_4, pi_4, vr_pos_mask, vl);
        both_inf_result = __riscv_vfsgnj(both_inf_result, vi, vl); // Copy sign from vi
        inf_result = __riscv_vmerge(inf_result, both_inf_result, both_inf, vl);

        // vi infinite, vr finite: ±π/2
        vbool16_t vi_inf_only = __riscv_vmandn(vi_inf_mask, vr_inf_mask, vl);
        vfloat32m2_t vi_inf_result = __riscv_vfsgnj(cpio2, vi, vl); // π/2 with sign of vi
        inf_result = __riscv_vmerge(inf_result, vi_inf_result, vi_inf_only, vl);

        // vr infinite, vi finite: 0 or ±π
        vbool16_t vr_inf_only = __riscv_vmandn(vr_inf_mask, vi_inf_mask, vl);
        vfloat32m2_t vr_inf_result =
            __riscv_vmerge(__riscv_vfsgnj(cpi, vi, vl),  // π with sign of vi
                           __riscv_vfsgnj(zero, vi, vl), // 0 with sign of vi
                           vr_pos_mask,
                           vl);
        inf_result = __riscv_vmerge(inf_result, vr_inf_result, vr_inf_only, vl);

        vbool16_t any_inf_mask = __riscv_vmor(vi_inf_mask, vr_inf_mask, vl);

        vbool16_t mswap = __riscv_vmfgt(vi_abs, vr_abs, vl);
        vfloat32m2_t numerator = __riscv_vmerge(vi, vr, mswap, vl);
        vfloat32m2_t denominator = __riscv_vmerge(vr, vi, mswap, vl);
        vfloat32m2_t x = __riscv_vfdiv(numerator, denominator, vl);

        // Only handle NaN from division (0/0, inf/inf), not from NaN inputs
        // Replace with numerator to preserve sign (e.g., atan2(-0, 0) = -0)
        vbool16_t x_nan_mask = __riscv_vmfne(x, x, vl);
        // div_nan_mask = x_nan_mask & ~input_nan_mask (vmandn computes vs2 & ~vs1)
        vbool16_t div_nan_mask = __riscv_vmandn(x_nan_mask, input_nan_mask, vl);
        x = __riscv_vmerge(x, numerator, div_nan_mask, vl);

        vfloat32m2_t xx = __riscv_vfmul(x, x, vl);
        vfloat32m2_t p = c13;
        p = __riscv_vfmadd(p, xx, c11, vl);
        p = __riscv_vfmadd(p, xx, c9, vl);
        p = __riscv_vfmadd(p, xx, c7, vl);
        p = __riscv_vfmadd(p, xx, c5, vl);
        p = __riscv_vfmadd(p, xx, c3, vl);
        p = __riscv_vfmadd(p, xx, c1, vl);
        p = __riscv_vfmul(p, x, vl);

        x = __riscv_vfsub(__riscv_vfsgnj(cpio2, x, vl), p, vl);
        p = __riscv_vmerge(p, x, mswap, vl);
        p = __riscv_vfadd_mu(
            RISCV_VMFLTZ(32m2, vr, vl), p, p, __riscv_vfsgnjx(cpi, vi, vl), vl);

        // Select infinity result or normal result
        p = __riscv_vmerge(p, inf_result, any_inf_mask, vl);

        __riscv_vse32(outputVector, __riscv_vfmul(p, norm, vl), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_32fc_s32f_atan2_32f_u_H */
