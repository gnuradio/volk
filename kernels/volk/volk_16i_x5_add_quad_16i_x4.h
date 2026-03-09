/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_x5_add_quad_16i_x4
 *
 * \b Deprecation
 *
 * This kernel is deprecated, no replacement has been identified.
 *
 * \b Overview
 *
 * Adds a common 16-bit integer base vector (src0) to each of four other 16-bit integer
 * vectors (src1, src2, src3, src4), producing four output vectors. Computes target0 =
 * src0 + src1, target1 = src0 + src2, target2 = src0 + src3, and target3 = src0 + src4
 * element-wise.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_x5_add_quad_16i_x4(short* target0, short* target1, short* target2,
 * short* target3, short* src0, short* src1, short* src2, short* src3, short* src4,
 * unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li src0: Common base vector of 16-bit integers added to each of the other inputs.
 * \li src1: Input vector of 16-bit integers added to src0 to produce target0.
 * \li src2: Input vector of 16-bit integers added to src0 to produce target1.
 * \li src3: Input vector of 16-bit integers added to src0 to produce target2.
 * \li src4: Input vector of 16-bit integers added to src0 to produce target3.
 * \li num_points: The number of elements in each vector.
 *
 * \b Outputs
 * \li target0: Output vector of 16-bit integers: src0 + src1.
 * \li target1: Output vector of 16-bit integers: src0 + src2.
 * \li target2: Output vector of 16-bit integers: src0 + src3.
 * \li target3: Output vector of 16-bit integers: src0 + src4.
 *
 * \b Example
 * \code
 * #include <volk/volk.h>
 * #include <stdio.h>
 *
 * int main() {
 *     unsigned int N = 10;
 *     unsigned int alignment = volk_get_alignment();
 *
 *     short* src0 = (short*)volk_malloc(sizeof(short) * N, alignment);
 *     short* src1 = (short*)volk_malloc(sizeof(short) * N, alignment);
 *     short* src2 = (short*)volk_malloc(sizeof(short) * N, alignment);
 *     short* src3 = (short*)volk_malloc(sizeof(short) * N, alignment);
 *     short* src4 = (short*)volk_malloc(sizeof(short) * N, alignment);
 *     short* target0 = (short*)volk_malloc(sizeof(short) * N, alignment);
 *     short* target1 = (short*)volk_malloc(sizeof(short) * N, alignment);
 *     short* target2 = (short*)volk_malloc(sizeof(short) * N, alignment);
 *     short* target3 = (short*)volk_malloc(sizeof(short) * N, alignment);
 *
 *     for (unsigned int i = 0; i < N; i++) {
 *         src0[i] = (short)i;       // common base: 0, 1, 2, ...
 *         src1[i] = (short)(i * 2); // 0, 2, 4, ...
 *         src2[i] = (short)(i * 3); // 0, 3, 6, ...
 *         src3[i] = (short)(i * 4); // 0, 4, 8, ...
 *         src4[i] = (short)(i * 5); // 0, 5, 10, ...
 *     }
 *
 *     volk_16i_x5_add_quad_16i_x4(
 *         target0, target1, target2, target3,
 *         src0, src1, src2, src3, src4, N);
 *
 *     for (unsigned int i = 0; i < N; i++) {
 *         printf("i=%u: t0=%d t1=%d t2=%d t3=%d\n",
 *                i, target0[i], target1[i], target2[i], target3[i]);
 *     }
 *
 *     volk_free(src0);
 *     volk_free(src1);
 *     volk_free(src2);
 *     volk_free(src3);
 *     volk_free(src4);
 *     volk_free(target0);
 *     volk_free(target1);
 *     volk_free(target2);
 *     volk_free(target3);
 * }
 * \endcode
 */

#ifndef INCLUDED_volk_16i_x5_add_quad_16i_x4_a_H
#define INCLUDED_volk_16i_x5_add_quad_16i_x4_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>
#include <xmmintrin.h>

static inline void volk_16i_x5_add_quad_16i_x4_a_sse2(short* target0,
                                                      short* target1,
                                                      short* target2,
                                                      short* target3,
                                                      short* src0,
                                                      short* src1,
                                                      short* src2,
                                                      short* src3,
                                                      short* src4,
                                                      unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    __m128i xmm0, xmm1, xmm2, xmm3, xmm4;
    __m128i *p_target0, *p_target1, *p_target2, *p_target3, *p_src0, *p_src1, *p_src2,
        *p_src3, *p_src4;
    p_target0 = (__m128i*)target0;
    p_target1 = (__m128i*)target1;
    p_target2 = (__m128i*)target2;
    p_target3 = (__m128i*)target3;

    p_src0 = (__m128i*)src0;
    p_src1 = (__m128i*)src1;
    p_src2 = (__m128i*)src2;
    p_src3 = (__m128i*)src3;
    p_src4 = (__m128i*)src4;

    int i = 0;

    int bound = (num_bytes >> 4);
    int leftovers = (num_bytes >> 1) & 7;

    for (; i < bound; ++i) {
        xmm0 = _mm_load_si128(p_src0);
        xmm1 = _mm_load_si128(p_src1);
        xmm2 = _mm_load_si128(p_src2);
        xmm3 = _mm_load_si128(p_src3);
        xmm4 = _mm_load_si128(p_src4);

        p_src0 += 1;
        p_src1 += 1;

        xmm1 = _mm_add_epi16(xmm0, xmm1);
        xmm2 = _mm_add_epi16(xmm0, xmm2);
        xmm3 = _mm_add_epi16(xmm0, xmm3);
        xmm4 = _mm_add_epi16(xmm0, xmm4);


        p_src2 += 1;
        p_src3 += 1;
        p_src4 += 1;

        _mm_store_si128(p_target0, xmm1);
        _mm_store_si128(p_target1, xmm2);
        _mm_store_si128(p_target2, xmm3);
        _mm_store_si128(p_target3, xmm4);

        p_target0 += 1;
        p_target1 += 1;
        p_target2 += 1;
        p_target3 += 1;
    }

    for (i = bound * 8; i < (bound * 8) + leftovers; ++i) {
        target0[i] = src0[i] + src1[i];
        target1[i] = src0[i] + src2[i];
        target2[i] = src0[i] + src3[i];
        target3[i] = src0[i] + src4[i];
    }
}
#endif /*LV_HAVE_SSE2*/

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16i_x5_add_quad_16i_x4_neon(short* target0,
                                                    short* target1,
                                                    short* target2,
                                                    short* target3,
                                                    short* src0,
                                                    short* src1,
                                                    short* src2,
                                                    short* src3,
                                                    short* src4,
                                                    unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    unsigned int number = 0;

    int16x8_t src0_vec, src1_vec, src2_vec, src3_vec, src4_vec;
    int16x8_t target0_vec, target1_vec, target2_vec, target3_vec;
    for (number = 0; number < eighth_points; ++number) {
        src0_vec = vld1q_s16(src0);
        src1_vec = vld1q_s16(src1);
        src2_vec = vld1q_s16(src2);
        src3_vec = vld1q_s16(src3);
        src4_vec = vld1q_s16(src4);

        target0_vec = vaddq_s16(src0_vec, src1_vec);
        target1_vec = vaddq_s16(src0_vec, src2_vec);
        target2_vec = vaddq_s16(src0_vec, src3_vec);
        target3_vec = vaddq_s16(src0_vec, src4_vec);

        vst1q_s16(target0, target0_vec);
        vst1q_s16(target1, target1_vec);
        vst1q_s16(target2, target2_vec);
        vst1q_s16(target3, target3_vec);
        src0 += 8;
        src1 += 8;
        src2 += 8;
        src3 += 8;
        src4 += 8;
        target0 += 8;
        target1 += 8;
        target2 += 8;
        target3 += 8;
    }

    for (number = eighth_points * 8; number < num_points; ++number) {
        *target0++ = *src0 + *src1++;
        *target1++ = *src0 + *src2++;
        *target2++ = *src0 + *src3++;
        *target3++ = *src0++ + *src4++;
    }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_GENERIC

static inline void volk_16i_x5_add_quad_16i_x4_generic(short* target0,
                                                       short* target1,
                                                       short* target2,
                                                       short* target3,
                                                       short* src0,
                                                       short* src1,
                                                       short* src2,
                                                       short* src3,
                                                       short* src4,
                                                       unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    for (i = 0; i < bound; ++i) {
        target0[i] = src0[i] + src1[i];
        target1[i] = src0[i] + src2[i];
        target2[i] = src0[i] + src3[i];
        target3[i] = src0[i] + src4[i];
    }
}

#endif /* LV_HAVE_GENERIC */

#endif /*INCLUDED_volk_16i_x5_add_quad_16i_x4_a_H*/
