/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 * Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_atan_32f
 *
 * \b Overview
 *
 * Computes arcsine of input vector and stores results in output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_atan_32f(float* bVector, const float* aVector, unsigned int num_points)
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
 * Calculate common angles around the top half of the unit circle.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   in[0] = 0.f;
 *   in[1] = 1.f/std::sqrt(3.f);
 *   in[2] = 1.f;
 *   in[3] = std::sqrt(3.f);
 *   in[4] = in[5] = 1e99;
 *   for(unsigned int ii = 6; ii < N; ++ii){
 *       in[ii] = - in[N-ii-1];
 *   }
 *
 *   volk_32f_atan_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("atan(%1.3f) = %1.3f\n", in[ii], out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */
#include <math.h>

#ifndef INCLUDED_volk_32f_atan_32f_a_H
#define INCLUDED_volk_32f_atan_32f_a_H

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
static inline void
volk_32f_atan_32f_a_avx2_fma(float* out, const float* in, unsigned int num_points)
{
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 pi_over_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    unsigned int number = 0;
    unsigned int eighth_points = num_points / 8;
    for (; number < eighth_points; number++) {
        __m256 x = _mm256_load_ps(in);
        __m256 swap_mask = _mm256_cmp_ps(_mm256_and_ps(x, abs_mask), one, _CMP_GT_OS);
        __m256 x_star = _mm256_div_ps(_mm256_blendv_ps(x, one, swap_mask),
                                      _mm256_blendv_ps(one, x, swap_mask));
        __m256 result = _m256_arctan_poly_avx2_fma(x_star);
        __m256 term = _mm256_and_ps(x_star, sign_mask);
        term = _mm256_or_ps(pi_over_2, term);
        term = _mm256_sub_ps(term, result);
        result = _mm256_blendv_ps(result, term, swap_mask);
        _mm256_store_ps(out, result);
        in += 8;
        out += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *out++ = volk_arctan(*in++);
    }
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for aligned */

#if LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>
static inline void
volk_32f_atan_32f_a_avx2(float* out, const float* in, unsigned int num_points)
{
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 pi_over_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    unsigned int number = 0;
    unsigned int eighth_points = num_points / 8;
    for (; number < eighth_points; number++) {
        __m256 x = _mm256_load_ps(in);
        __m256 swap_mask = _mm256_cmp_ps(_mm256_and_ps(x, abs_mask), one, _CMP_GT_OS);
        __m256 x_star = _mm256_div_ps(_mm256_blendv_ps(x, one, swap_mask),
                                      _mm256_blendv_ps(one, x, swap_mask));
        __m256 result = _m256_arctan_poly_avx(x_star);
        __m256 term = _mm256_and_ps(x_star, sign_mask);
        term = _mm256_or_ps(pi_over_2, term);
        term = _mm256_sub_ps(term, result);
        result = _mm256_blendv_ps(result, term, swap_mask);
        _mm256_store_ps(out, result);
        in += 8;
        out += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *out++ = volk_arctan(*in++);
    }
}
#endif /* LV_HAVE_AVX for aligned */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
#include <volk/volk_sse_intrinsics.h>
static inline void
volk_32f_atan_32f_a_sse4_1(float* out, const float* in, unsigned int num_points)
{
    const __m128 one = _mm_set1_ps(1.f);
    const __m128 pi_over_2 = _mm_set1_ps(0x1.921fb6p0f);
    const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    const __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;
    for (; number < quarter_points; number++) {
        __m128 x = _mm_load_ps(in);
        __m128 swap_mask = _mm_cmpgt_ps(_mm_and_ps(x, abs_mask), one);
        __m128 x_star = _mm_div_ps(_mm_blendv_ps(x, one, swap_mask),
                                   _mm_blendv_ps(one, x, swap_mask));
        __m128 result = _mm_arctan_poly_sse(x_star);
        __m128 term = _mm_and_ps(x_star, sign_mask);
        term = _mm_or_ps(pi_over_2, term);
        term = _mm_sub_ps(term, result);
        result = _mm_blendv_ps(result, term, swap_mask);
        _mm_store_ps(out, result);
        in += 4;
        out += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        *out++ = volk_arctan(*in++);
    }
}
#endif /* LV_HAVE_SSE4_1 for aligned */
#endif /* INCLUDED_volk_32f_atan_32f_a_H */

#ifndef INCLUDED_volk_32f_atan_32f_u_H
#define INCLUDED_volk_32f_atan_32f_u_H

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
static inline void
volk_32f_atan_32f_u_avx2_fma(float* out, const float* in, unsigned int num_points)
{
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 pi_over_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    unsigned int number = 0;
    unsigned int eighth_points = num_points / 8;
    for (; number < eighth_points; number++) {
        __m256 x = _mm256_loadu_ps(in);
        __m256 swap_mask = _mm256_cmp_ps(_mm256_and_ps(x, abs_mask), one, _CMP_GT_OS);
        __m256 x_star = _mm256_div_ps(_mm256_blendv_ps(x, one, swap_mask),
                                      _mm256_blendv_ps(one, x, swap_mask));
        __m256 result = _m256_arctan_poly_avx2_fma(x_star);
        __m256 term = _mm256_and_ps(x_star, sign_mask);
        term = _mm256_or_ps(pi_over_2, term);
        term = _mm256_sub_ps(term, result);
        result = _mm256_blendv_ps(result, term, swap_mask);
        _mm256_storeu_ps(out, result);
        in += 8;
        out += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *out++ = volk_arctan(*in++);
    }
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for unaligned */

#if LV_HAVE_AVX
#include <immintrin.h>
static inline void
volk_32f_atan_32f_u_avx2(float* out, const float* in, unsigned int num_points)
{
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 pi_over_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    unsigned int number = 0;
    unsigned int eighth_points = num_points / 8;
    for (; number < eighth_points; number++) {
        __m256 x = _mm256_loadu_ps(in);
        __m256 swap_mask = _mm256_cmp_ps(_mm256_and_ps(x, abs_mask), one, _CMP_GT_OS);
        __m256 x_star = _mm256_div_ps(_mm256_blendv_ps(x, one, swap_mask),
                                      _mm256_blendv_ps(one, x, swap_mask));
        __m256 result = _m256_arctan_poly_avx(x_star);
        __m256 term = _mm256_and_ps(x_star, sign_mask);
        term = _mm256_or_ps(pi_over_2, term);
        term = _mm256_sub_ps(term, result);
        result = _mm256_blendv_ps(result, term, swap_mask);
        _mm256_storeu_ps(out, result);
        in += 8;
        out += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *out++ = volk_arctan(*in++);
    }
}
#endif /* LV_HAVE_AVX for unaligned */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
#include <volk/volk_sse_intrinsics.h>
static inline void
volk_32f_atan_32f_u_sse4_1(float* out, const float* in, unsigned int num_points)
{
    const __m128 one = _mm_set1_ps(1.f);
    const __m128 pi_over_2 = _mm_set1_ps(0x1.921fb6p0f);
    const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    const __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;
    for (; number < quarter_points; number++) {
        __m128 x = _mm_loadu_ps(in);
        __m128 swap_mask = _mm_cmpgt_ps(_mm_and_ps(x, abs_mask), one);
        __m128 x_star = _mm_div_ps(_mm_blendv_ps(x, one, swap_mask),
                                   _mm_blendv_ps(one, x, swap_mask));
        __m128 result = _mm_arctan_poly_sse(x_star);
        __m128 term = _mm_and_ps(x_star, sign_mask);
        term = _mm_or_ps(pi_over_2, term);
        term = _mm_sub_ps(term, result);
        result = _mm_blendv_ps(result, term, swap_mask);
        _mm_storeu_ps(out, result);
        in += 4;
        out += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        *out++ = volk_arctan(*in++);
    }
}
#endif /* LV_HAVE_SSE4_1 for unaligned */

#ifdef LV_HAVE_GENERIC
static inline void
volk_32f_atan_32f_polynomial(float* out, const float* in, unsigned int num_points)
{
    unsigned int number = 0;
    for (; number < num_points; number++) {
        *out++ = volk_arctan(*in++);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_GENERIC
static inline void
volk_32f_atan_32f_generic(float* out, const float* in, unsigned int num_points)
{
    unsigned int number = 0;
    for (; number < num_points; number++) {
        *out++ = atanf(*in++);
    }
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32f_atan_32f_u_H */
