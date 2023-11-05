/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_x2_square_dist_32f
 *
 * \b Overview
 *
 * Calculates the square distance between a single complex input for each
 * point in a complex vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_square_dist_32f(float* target, lv_32fc_t* src0, lv_32fc_t* points,
 * unsigned int num_points) { \endcode
 *
 * \b Inputs
 * \li src0: The complex input. Only the first point is used.
 * \li points: A complex vector of reference points.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li target: A vector of distances between src0 and the vector of points.
 *
 * \b Example
 * Calculate the distance between an input and reference points in a square
 * 16-qam constellation.
 * \code
 *   int N = 16;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* constellation  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* rx  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float const_vals[] = {-3, -1, 1, 3};
 *
 *   // Generate 16-QAM constellation points
 *   unsigned int jj = 0;
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       constellation[ii] = lv_cmake(const_vals[ii%4], const_vals[jj]);
 *       if((ii+1)%4 == 0) ++jj;
 *   }
 *
 *   *rx = lv_cmake(0.5f, 2.f);
 *
 *   volk_32fc_x2_square_dist_32f(out, rx, constellation, N);
 *
 *   printf("Distance from each constellation point:\n");
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("%.4f  ", out[ii]);
 *       if((ii+1)%4 == 0) printf("\n");
 *   }
 *
 *   volk_free(rx);
 *   volk_free(constellation);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_square_dist_32f_a_H
#define INCLUDED_volk_32fc_x2_square_dist_32f_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_x2_square_dist_32f_a_avx2(float* target,
                                                       lv_32fc_t* src0,
                                                       lv_32fc_t* points,
                                                       unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 8;
    __m128 xmm0, xmm9, xmm10;
    __m256 xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    lv_32fc_t diff;
    float sq_dist;
    int bound = num_bytes >> 6;
    int leftovers0 = (num_bytes >> 5) & 1;
    int leftovers1 = (num_bytes >> 4) & 1;
    int leftovers2 = (num_bytes >> 3) & 1;
    int i = 0;

    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    xmm1 = _mm256_setzero_ps();
    xmm0 = _mm_load_ps((float*)src0);
    xmm0 = _mm_permute_ps(xmm0, 0b01000100);
    xmm1 = _mm256_insertf128_ps(xmm1, xmm0, 0);
    xmm1 = _mm256_insertf128_ps(xmm1, xmm0, 1);

    for (; i < bound; ++i) {
        xmm2 = _mm256_load_ps((float*)&points[0]);
        xmm3 = _mm256_load_ps((float*)&points[4]);
        points += 8;

        xmm4 = _mm256_sub_ps(xmm1, xmm2);
        xmm5 = _mm256_sub_ps(xmm1, xmm3);
        xmm6 = _mm256_mul_ps(xmm4, xmm4);
        xmm7 = _mm256_mul_ps(xmm5, xmm5);

        xmm4 = _mm256_hadd_ps(xmm6, xmm7);
        xmm4 = _mm256_permutevar8x32_ps(xmm4, idx);

        _mm256_store_ps(target, xmm4);

        target += 8;
    }

    for (i = 0; i < leftovers0; ++i) {

        xmm2 = _mm256_load_ps((float*)&points[0]);

        xmm4 = _mm256_sub_ps(xmm1, xmm2);

        points += 4;

        xmm6 = _mm256_mul_ps(xmm4, xmm4);

        xmm4 = _mm256_hadd_ps(xmm6, xmm6);
        xmm4 = _mm256_permutevar8x32_ps(xmm4, idx);

        xmm9 = _mm256_extractf128_ps(xmm4, 1);
        _mm_store_ps(target, xmm9);

        target += 4;
    }

    for (i = 0; i < leftovers1; ++i) {
        xmm9 = _mm_load_ps((float*)&points[0]);

        xmm10 = _mm_sub_ps(xmm0, xmm9);

        points += 2;

        xmm9 = _mm_mul_ps(xmm10, xmm10);

        xmm10 = _mm_hadd_ps(xmm9, xmm9);

        _mm_storeh_pi((__m64*)target, xmm10);

        target += 2;
    }

    for (i = 0; i < leftovers2; ++i) {

        diff = src0[0] - points[0];

        sq_dist = lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff);

        target[0] = sq_dist;
    }
}

#endif /*LV_HAVE_AVX2*/

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <xmmintrin.h>

static inline void volk_32fc_x2_square_dist_32f_a_sse3(float* target,
                                                       lv_32fc_t* src0,
                                                       lv_32fc_t* points,
                                                       unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 8;

    __m128 xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    lv_32fc_t diff;
    float sq_dist;
    int bound = num_bytes >> 5;
    int i = 0;

    xmm1 = _mm_setzero_ps();
    xmm1 = _mm_loadl_pi(xmm1, (__m64*)src0);
    xmm1 = _mm_movelh_ps(xmm1, xmm1);

    for (; i < bound; ++i) {
        xmm2 = _mm_load_ps((float*)&points[0]);
        xmm4 = _mm_sub_ps(xmm1, xmm2);
        xmm3 = _mm_load_ps((float*)&points[2]);
        xmm5 = _mm_sub_ps(xmm1, xmm3);

        xmm6 = _mm_mul_ps(xmm4, xmm4);
        xmm7 = _mm_mul_ps(xmm5, xmm5);

        xmm4 = _mm_hadd_ps(xmm6, xmm7);

        _mm_store_ps(target, xmm4);

        points += 4;
        target += 4;
    }

    if (num_bytes >> 4 & 1) {

        xmm2 = _mm_load_ps((float*)&points[0]);

        xmm4 = _mm_sub_ps(xmm1, xmm2);

        points += 2;

        xmm6 = _mm_mul_ps(xmm4, xmm4);

        xmm4 = _mm_hadd_ps(xmm6, xmm6);

        _mm_storeh_pi((__m64*)target, xmm4);

        target += 2;
    }

    if (num_bytes >> 3 & 1) {

        diff = src0[0] - points[0];

        sq_dist = lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff);

        target[0] = sq_dist;
    }
}

#endif /*LV_HAVE_SSE3*/


#ifdef LV_HAVE_NEON
#include <arm_neon.h>
static inline void volk_32fc_x2_square_dist_32f_neon(float* target,
                                                     lv_32fc_t* src0,
                                                     lv_32fc_t* points,
                                                     unsigned int num_points)
{
    const unsigned int quarter_points = num_points / 4;
    unsigned int number;

    float32x4x2_t a_vec, b_vec;
    float32x4x2_t diff_vec;
    float32x4_t tmp, tmp1, dist_sq;
    a_vec.val[0] = vdupq_n_f32(lv_creal(src0[0]));
    a_vec.val[1] = vdupq_n_f32(lv_cimag(src0[0]));
    for (number = 0; number < quarter_points; ++number) {
        b_vec = vld2q_f32((float*)points);
        diff_vec.val[0] = vsubq_f32(a_vec.val[0], b_vec.val[0]);
        diff_vec.val[1] = vsubq_f32(a_vec.val[1], b_vec.val[1]);
        tmp = vmulq_f32(diff_vec.val[0], diff_vec.val[0]);
        tmp1 = vmulq_f32(diff_vec.val[1], diff_vec.val[1]);

        dist_sq = vaddq_f32(tmp, tmp1);
        vst1q_f32(target, dist_sq);
        points += 4;
        target += 4;
    }
    for (number = quarter_points * 4; number < num_points; ++number) {
        lv_32fc_t diff = src0[0] - *points++;
        *target++ = lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff);
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_GENERIC
static inline void volk_32fc_x2_square_dist_32f_generic(float* target,
                                                        lv_32fc_t* src0,
                                                        lv_32fc_t* points,
                                                        unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 8;

    lv_32fc_t diff;
    float sq_dist;
    unsigned int i = 0;

    for (; i<num_bytes>> 3; ++i) {
        diff = src0[0] - points[i];

        sq_dist = lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff);

        target[i] = sq_dist;
    }
}

#endif /*LV_HAVE_GENERIC*/


#endif /*INCLUDED_volk_32fc_x2_square_dist_32f_a_H*/

#ifndef INCLUDED_volk_32fc_x2_square_dist_32f_u_H
#define INCLUDED_volk_32fc_x2_square_dist_32f_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_x2_square_dist_32f_u_avx2(float* target,
                                                       lv_32fc_t* src0,
                                                       lv_32fc_t* points,
                                                       unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 8;
    __m128 xmm0, xmm9;
    __m256 xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    lv_32fc_t diff;
    float sq_dist;
    int bound = num_bytes >> 6;
    int leftovers1 = (num_bytes >> 3) & 0b11;
    int i = 0;

    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    xmm1 = _mm256_setzero_ps();
    xmm0 = _mm_loadu_ps((float*)src0);
    xmm0 = _mm_permute_ps(xmm0, 0b01000100);
    xmm1 = _mm256_insertf128_ps(xmm1, xmm0, 0);
    xmm1 = _mm256_insertf128_ps(xmm1, xmm0, 1);

    for (; i < bound; ++i) {
        xmm2 = _mm256_loadu_ps((float*)&points[0]);
        xmm3 = _mm256_loadu_ps((float*)&points[4]);
        points += 8;

        xmm4 = _mm256_sub_ps(xmm1, xmm2);
        xmm5 = _mm256_sub_ps(xmm1, xmm3);
        xmm6 = _mm256_mul_ps(xmm4, xmm4);
        xmm7 = _mm256_mul_ps(xmm5, xmm5);

        xmm4 = _mm256_hadd_ps(xmm6, xmm7);
        xmm4 = _mm256_permutevar8x32_ps(xmm4, idx);

        _mm256_storeu_ps(target, xmm4);

        target += 8;
    }

    if (num_bytes >> 5 & 1) {

        xmm2 = _mm256_loadu_ps((float*)&points[0]);

        xmm4 = _mm256_sub_ps(xmm1, xmm2);

        points += 4;

        xmm6 = _mm256_mul_ps(xmm4, xmm4);

        xmm4 = _mm256_hadd_ps(xmm6, xmm6);
        xmm4 = _mm256_permutevar8x32_ps(xmm4, idx);

        xmm9 = _mm256_extractf128_ps(xmm4, 1);
        _mm_storeu_ps(target, xmm9);

        target += 4;
    }

    for (i = 0; i < leftovers1; ++i) {

        diff = src0[0] - points[0];
        points += 1;

        sq_dist = lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff);

        target[0] = sq_dist;
        target += 1;
    }
}

#endif /*LV_HAVE_AVX2*/

#endif /*INCLUDED_volk_32fc_x2_square_dist_32f_u_H*/
