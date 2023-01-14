/* -*- c++ -*- */
/*
 * Copyright 2021 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_index_min_16u
 *
 * \b Overview
 *
 * Returns Argmin_i mag(x[i]). Finds and returns the index which contains the
 * minimum magnitude for complex points in the given vector.
 *
 * Note that num_points is a uint32_t, but the return value is
 * uint16_t. Providing a vector larger than the max of a uint16_t
 * (65536) would miss anything outside of this boundary. The kernel
 * will check the length of num_points and cap it to this max value,
 * anyways.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_index_min_16u(uint16_t* target, lv_32fc_t* source, uint32_t
 * num_points) \endcode
 *
 * \b Inputs
 * \li source: The complex input vector.
 * \li num_points: The number of samples.
 *
 * \b Outputs
 * \li target: The index of the point with minimum magnitude.
 *
 * \b Example
 * Calculate the index of the minimum value of \f$x^2 + x\f$ for points around
 * the unit circle.
 * \code
 *   int N = 10;
 *   uint32_t alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   uint16_t* min = (uint16_t*)volk_malloc(sizeof(uint16_t), alignment);
 *
 *   for(uint32_t ii = 0; ii < N/2; ++ii){
 *       float real = 2.f * ((float)ii / (float)N) - 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *       in[ii] = in[ii] * in[ii] + in[ii];
 *       in[N-ii] = lv_cmake(real, imag);
 *       in[N-ii] = in[N-ii] * in[N-ii] + in[N-ii];
 *   }
 *
 *   volk_32fc_index_min_16u(min, in, N);
 *
 *   printf("index of min value = %u\n",  *min);
 *
 *   volk_free(in);
 *   volk_free(min);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_index_min_16u_a_H
#define INCLUDED_volk_32fc_index_min_16u_a_H

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <volk/volk_common.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void volk_32fc_index_min_16u_a_avx2_variant_0(uint16_t* target,
                                                            const lv_32fc_t* source,
                                                            uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    const __m256i indices_increment = _mm256_set1_epi32(8);
    /*
     * At the start of each loop iteration current_indices holds the indices of
     * the complex numbers loaded from memory. Explanation for odd order is given
     * in implementation of vector_32fc_index_min_variant0().
     */
    __m256i current_indices = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    __m256 min_values = _mm256_set1_ps(FLT_MAX);
    __m256i min_indices = _mm256_setzero_si256();

    for (unsigned i = 0; i < num_points / 8u; ++i) {
        __m256 in0 = _mm256_load_ps((float*)source);
        __m256 in1 = _mm256_load_ps((float*)(source + 4));
        vector_32fc_index_min_variant0(
            in0, in1, &min_values, &min_indices, &current_indices, indices_increment);
        source += 8;
    }

    // determine minimum value and index in the result of the vectorized loop
    __VOLK_ATTR_ALIGNED(32) float min_values_buffer[8];
    __VOLK_ATTR_ALIGNED(32) uint32_t min_indices_buffer[8];
    _mm256_store_ps(min_values_buffer, min_values);
    _mm256_store_si256((__m256i*)min_indices_buffer, min_indices);

    float min = FLT_MAX;
    uint32_t index = 0;
    for (unsigned i = 0; i < 8; i++) {
        if (min_values_buffer[i] < min) {
            min = min_values_buffer[i];
            index = min_indices_buffer[i];
        }
    }

    // handle tail not processed by the vectorized loop
    for (unsigned i = num_points & (~7u); i < num_points; ++i) {
        const float abs_squared =
            lv_creal(*source) * lv_creal(*source) + lv_cimag(*source) * lv_cimag(*source);
        if (abs_squared < min) {
            min = abs_squared;
            index = i;
        }
        ++source;
    }

    *target = index;
}

#endif /*LV_HAVE_AVX2*/

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void volk_32fc_index_min_16u_a_avx2_variant_1(uint16_t* target,
                                                            const lv_32fc_t* source,
                                                            uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    const __m256i indices_increment = _mm256_set1_epi32(8);
    /*
     * At the start of each loop iteration current_indices holds the indices of
     * the complex numbers loaded from memory. Explanation for odd order is given
     * in implementation of vector_32fc_index_min_variant0().
     */
    __m256i current_indices = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    __m256 min_values = _mm256_set1_ps(FLT_MAX);
    __m256i min_indices = _mm256_setzero_si256();

    for (unsigned i = 0; i < num_points / 8u; ++i) {
        __m256 in0 = _mm256_load_ps((float*)source);
        __m256 in1 = _mm256_load_ps((float*)(source + 4));
        vector_32fc_index_min_variant1(
            in0, in1, &min_values, &min_indices, &current_indices, indices_increment);
        source += 8;
    }

    // determine minimum value and index in the result of the vectorized loop
    __VOLK_ATTR_ALIGNED(32) float min_values_buffer[8];
    __VOLK_ATTR_ALIGNED(32) uint32_t min_indices_buffer[8];
    _mm256_store_ps(min_values_buffer, min_values);
    _mm256_store_si256((__m256i*)min_indices_buffer, min_indices);

    float min = FLT_MAX;
    uint32_t index = 0;
    for (unsigned i = 0; i < 8; i++) {
        if (min_values_buffer[i] < min) {
            min = min_values_buffer[i];
            index = min_indices_buffer[i];
        }
    }

    // handle tail not processed by the vectorized loop
    for (unsigned i = num_points & (~7u); i < num_points; ++i) {
        const float abs_squared =
            lv_creal(*source) * lv_creal(*source) + lv_cimag(*source) * lv_cimag(*source);
        if (abs_squared < min) {
            min = abs_squared;
            index = i;
        }
        ++source;
    }

    *target = index;
}

#endif /*LV_HAVE_AVX2*/

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <xmmintrin.h>

static inline void volk_32fc_index_min_16u_a_sse3(uint16_t* target,
                                                  const lv_32fc_t* source,
                                                  uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    union bit128 holderf;
    union bit128 holderi;
    float sq_dist = 0.0;

    union bit128 xmm5, xmm4;
    __m128 xmm1, xmm2, xmm3;
    __m128i xmm8, xmm11, xmm12, xmm9, xmm10;

    xmm5.int_vec = _mm_setzero_si128();
    xmm4.int_vec = _mm_setzero_si128();
    holderf.int_vec = _mm_setzero_si128();
    holderi.int_vec = _mm_setzero_si128();

    xmm8 = _mm_setr_epi32(0, 1, 2, 3);
    xmm9 = _mm_setzero_si128();
    xmm10 = _mm_setr_epi32(4, 4, 4, 4);
    xmm3 = _mm_set_ps1(FLT_MAX);

    int bound = num_points >> 2;

    for (int i = 0; i < bound; ++i) {
        xmm1 = _mm_load_ps((float*)source);
        xmm2 = _mm_load_ps((float*)&source[2]);

        source += 4;

        xmm1 = _mm_mul_ps(xmm1, xmm1);
        xmm2 = _mm_mul_ps(xmm2, xmm2);

        xmm1 = _mm_hadd_ps(xmm1, xmm2);

        xmm3 = _mm_min_ps(xmm1, xmm3);

        xmm4.float_vec = _mm_cmpgt_ps(xmm1, xmm3);
        xmm5.float_vec = _mm_cmpeq_ps(xmm1, xmm3);

        xmm11 = _mm_and_si128(xmm8, xmm5.int_vec);
        xmm12 = _mm_and_si128(xmm9, xmm4.int_vec);

        xmm9 = _mm_add_epi32(xmm11, xmm12);

        xmm8 = _mm_add_epi32(xmm8, xmm10);
    }

    if (num_points >> 1 & 1) {
        xmm2 = _mm_load_ps((float*)source);

        xmm1 = _mm_movelh_ps(bit128_p(&xmm8)->float_vec, bit128_p(&xmm8)->float_vec);
        xmm8 = bit128_p(&xmm1)->int_vec;

        xmm2 = _mm_mul_ps(xmm2, xmm2);

        source += 2;

        xmm1 = _mm_hadd_ps(xmm2, xmm2);

        xmm3 = _mm_min_ps(xmm1, xmm3);

        xmm10 = _mm_setr_epi32(2, 2, 2, 2);

        xmm4.float_vec = _mm_cmpgt_ps(xmm1, xmm3);
        xmm5.float_vec = _mm_cmpeq_ps(xmm1, xmm3);

        xmm11 = _mm_and_si128(xmm8, xmm5.int_vec);
        xmm12 = _mm_and_si128(xmm9, xmm4.int_vec);

        xmm9 = _mm_add_epi32(xmm11, xmm12);

        xmm8 = _mm_add_epi32(xmm8, xmm10);
    }

    if (num_points & 1) {
        sq_dist = lv_creal(source[0]) * lv_creal(source[0]) +
                  lv_cimag(source[0]) * lv_cimag(source[0]);

        xmm2 = _mm_load1_ps(&sq_dist);

        xmm1 = xmm3;

        xmm3 = _mm_min_ss(xmm3, xmm2);

        xmm4.float_vec = _mm_cmpgt_ps(xmm1, xmm3);
        xmm5.float_vec = _mm_cmpeq_ps(xmm1, xmm3);

        xmm8 = _mm_shuffle_epi32(xmm8, 0x00);

        xmm11 = _mm_and_si128(xmm8, xmm4.int_vec);
        xmm12 = _mm_and_si128(xmm9, xmm5.int_vec);

        xmm9 = _mm_add_epi32(xmm11, xmm12);
    }

    _mm_store_ps((float*)&(holderf.f), xmm3);
    _mm_store_si128(&(holderi.int_vec), xmm9);

    target[0] = holderi.i[0];
    sq_dist = holderf.f[0];
    target[0] = (holderf.f[1] < sq_dist) ? holderi.i[1] : target[0];
    sq_dist = (holderf.f[1] < sq_dist) ? holderf.f[1] : sq_dist;
    target[0] = (holderf.f[2] < sq_dist) ? holderi.i[2] : target[0];
    sq_dist = (holderf.f[2] < sq_dist) ? holderf.f[2] : sq_dist;
    target[0] = (holderf.f[3] < sq_dist) ? holderi.i[3] : target[0];
    sq_dist = (holderf.f[3] < sq_dist) ? holderf.f[3] : sq_dist;
}

#endif /*LV_HAVE_SSE3*/

#ifdef LV_HAVE_GENERIC
static inline void volk_32fc_index_min_16u_generic(uint16_t* target,
                                                   const lv_32fc_t* source,
                                                   uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    float sq_dist = 0.0;
    float min = FLT_MAX;
    uint16_t index = 0;

    for (uint32_t i = 0; i < num_points; ++i) {
        sq_dist = lv_creal(source[i]) * lv_creal(source[i]) +
                  lv_cimag(source[i]) * lv_cimag(source[i]);

        if (sq_dist < min) {
            index = i;
            min = sq_dist;
        }
    }
    target[0] = index;
}

#endif /*LV_HAVE_GENERIC*/

#endif /*INCLUDED_volk_32fc_index_min_16u_a_H*/

#ifndef INCLUDED_volk_32fc_index_min_16u_u_H
#define INCLUDED_volk_32fc_index_min_16u_u_H

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <volk/volk_common.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void volk_32fc_index_min_16u_u_avx2_variant_0(uint16_t* target,
                                                            const lv_32fc_t* source,
                                                            uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    const __m256i indices_increment = _mm256_set1_epi32(8);
    /*
     * At the start of each loop iteration current_indices holds the indices of
     * the complex numbers loaded from memory. Explanation for odd order is given
     * in implementation of vector_32fc_index_min_variant0().
     */
    __m256i current_indices = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    __m256 min_values = _mm256_set1_ps(FLT_MAX);
    __m256i min_indices = _mm256_setzero_si256();

    for (unsigned i = 0; i < num_points / 8u; ++i) {
        __m256 in0 = _mm256_loadu_ps((float*)source);
        __m256 in1 = _mm256_loadu_ps((float*)(source + 4));
        vector_32fc_index_min_variant0(
            in0, in1, &min_values, &min_indices, &current_indices, indices_increment);
        source += 8;
    }

    // determine minimum value and index in the result of the vectorized loop
    __VOLK_ATTR_ALIGNED(32) float min_values_buffer[8];
    __VOLK_ATTR_ALIGNED(32) uint32_t min_indices_buffer[8];
    _mm256_store_ps(min_values_buffer, min_values);
    _mm256_store_si256((__m256i*)min_indices_buffer, min_indices);

    float min = FLT_MAX;
    uint32_t index = 0;
    for (unsigned i = 0; i < 8; i++) {
        if (min_values_buffer[i] < min) {
            min = min_values_buffer[i];
            index = min_indices_buffer[i];
        }
    }

    // handle tail not processed by the vectorized loop
    for (unsigned i = num_points & (~7u); i < num_points; ++i) {
        const float abs_squared =
            lv_creal(*source) * lv_creal(*source) + lv_cimag(*source) * lv_cimag(*source);
        if (abs_squared < min) {
            min = abs_squared;
            index = i;
        }
        ++source;
    }

    *target = index;
}

#endif /*LV_HAVE_AVX2*/

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void volk_32fc_index_min_16u_u_avx2_variant_1(uint16_t* target,
                                                            const lv_32fc_t* source,
                                                            uint32_t num_points)
{
    num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

    const __m256i indices_increment = _mm256_set1_epi32(8);
    /*
     * At the start of each loop iteration current_indices holds the indices of
     * the complex numbers loaded from memory. Explanation for odd order is given
     * in implementation of vector_32fc_index_min_variant0().
     */
    __m256i current_indices = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    __m256 min_values = _mm256_set1_ps(FLT_MAX);
    __m256i min_indices = _mm256_setzero_si256();

    for (unsigned i = 0; i < num_points / 8u; ++i) {
        __m256 in0 = _mm256_loadu_ps((float*)source);
        __m256 in1 = _mm256_loadu_ps((float*)(source + 4));
        vector_32fc_index_min_variant1(
            in0, in1, &min_values, &min_indices, &current_indices, indices_increment);
        source += 8;
    }

    // determine minimum value and index in the result of the vectorized loop
    __VOLK_ATTR_ALIGNED(32) float min_values_buffer[8];
    __VOLK_ATTR_ALIGNED(32) uint32_t min_indices_buffer[8];
    _mm256_store_ps(min_values_buffer, min_values);
    _mm256_store_si256((__m256i*)min_indices_buffer, min_indices);

    float min = FLT_MAX;
    uint32_t index = 0;
    for (unsigned i = 0; i < 8; i++) {
        if (min_values_buffer[i] < min) {
            min = min_values_buffer[i];
            index = min_indices_buffer[i];
        }
    }

    // handle tail not processed by the vectorized loop
    for (unsigned i = num_points & (~7u); i < num_points; ++i) {
        const float abs_squared =
            lv_creal(*source) * lv_creal(*source) + lv_cimag(*source) * lv_cimag(*source);
        if (abs_squared < min) {
            min = abs_squared;
            index = i;
        }
        ++source;
    }

    *target = index;
}

#endif /*LV_HAVE_AVX2*/

#endif /*INCLUDED_volk_32fc_index_min_16u_u_H*/
