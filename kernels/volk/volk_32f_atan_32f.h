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

#define POLY_ORDER (13) // Use either 11, 12, 13 or 15
/*
 * arctan(x) polynomial expansion on the interval [-1, 1]
 */
#if (POLY_ORDER == 11)
static inline float arctan_approximation(const float x)
{
    /*
     * Max relative error < 4.4e-6
     */
    const float a1 = +0x1.ffff6ep-1f;
    const float a3 = -0x1.54fca2p-2f;
    const float a5 = +0x1.90aaa2p-3f;
    const float a7 = -0x1.f09d2ep-4f;
    const float a9 = +0x1.d6e42cp-5f;
    const float a11 = -0x1.b9c81ep-7f;

    const float x_times_x = x * x;
    float arctan = a11;
    arctan = fmaf(x_times_x, arctan, a9);
    arctan = fmaf(x_times_x, arctan, a7);
    arctan = fmaf(x_times_x, arctan, a5);
    arctan = fmaf(x_times_x, arctan, a3);
    arctan = fmaf(x_times_x, arctan, a1);
    arctan *= x;

    return arctan;
}
#elif (POLY_ORDER == 12) // Order 13 with a1 set to 1
static inline float arctan_approximation(const float x)
{
    /*
     * Max relative error < 7.5e-7
     */
    //          a1 == 1 implicitly
    const float a3 = -0x1.5548a4p-2f;
    const float a5 = +0x1.978224p-3f;
    const float a7 = -0x1.156488p-3f;
    const float a9 = +0x1.5b822cp-4f;
    const float a11 = -0x1.35a172p-5f;
    const float a13 = +0x1.09a14ep-7f;

    const float x_times_x = x * x;
    float arctan = a13;
    arctan = fmaf(x_times_x, arctan, a11);
    arctan = fmaf(x_times_x, arctan, a9);
    arctan = fmaf(x_times_x, arctan, a7);
    arctan = fmaf(x_times_x, arctan, a5);
    arctan = fmaf(x_times_x, arctan, a3);
    arctan *= x_times_x;
    arctan = fmaf(x, arctan, x);

    return arctan;
}
#elif (POLY_ORDER == 13)
static inline float arctan_approximation(const float x)
{
    /*
     * Max relative error < 6.6e-7
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
#elif (POLY_ORDER == 15)
static inline float arctan_approximation(const float x)
{
    /*
     * Max relative error < 1.0e-7
     */
    const float a1 = +0x1.fffffcp-1f;
    const float a3 = -0x1.55519ep-2f;
    const float a5 = +0x1.98f6a8p-3f;
    const float a7 = -0x1.1f0a92p-3f;
    const float a9 = +0x1.95b654p-4f;
    const float a11 = -0x1.e65492p-5f;
    const float a13 = +0x1.8c0c36p-6f;
    const float a15 = -0x1.32316ep-8f;

    const float x_times_x = x * x;
    float arctan = a15;
    arctan = fmaf(x_times_x, arctan, a13);
    arctan = fmaf(x_times_x, arctan, a11);
    arctan = fmaf(x_times_x, arctan, a9);
    arctan = fmaf(x_times_x, arctan, a7);
    arctan = fmaf(x_times_x, arctan, a5);
    arctan = fmaf(x_times_x, arctan, a3);
    arctan = fmaf(x_times_x, arctan, a1);
    arctan *= x;

    return arctan;
}
#else
#error Undefined polynomial order.
#endif

#ifndef INCLUDED_volk_32f_atan_32f_a_H
#define INCLUDED_volk_32f_atan_32f_a_H

static inline float arctan(const float x)
{
    /*
     *  arctan(x) + arctan(1 / x) == sign(x) * pi / 2
     */
    const float pi_over_2 = 0x1.921fb6p0f;

    if (fabs(x) < 1.f) {
        return arctan_approximation(x);
    } else {
        return copysignf(pi_over_2, x) - arctan_approximation(1.f / x);
    }
}

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
        __m256 result = _m256_arctan_approximation_avx2_fma(x_star);
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
        *out++ = arctan(*in++);
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
        __m256 result = _m256_arctan_approximation_avx(x_star);
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
        *out++ = arctan(*in++);
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
        __m128 result = _mm_arctan_approximation_sse(x_star);
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
        *out++ = arctan(*in++);
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
        __m256 result = _m256_arctan_approximation_avx2_fma(x_star);
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
        *out++ = arctan(*in++);
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
        __m256 result = _m256_arctan_approximation_avx(x_star);
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
        *out++ = arctan(*in++);
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
        __m128 result = _mm_arctan_approximation_sse(x_star);
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
        *out++ = arctan(*in++);
    }
}
#endif /* LV_HAVE_SSE4_1 for unaligned */

#ifdef LV_HAVE_GENERIC
static inline void
volk_32f_atan_32f_polynomial(float* out, const float* in, unsigned int num_points)
{
    unsigned int number = 0;
    for (; number < num_points; number++) {
        *out++ = arctan(*in++);
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
