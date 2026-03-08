/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_x4_quad_max_star_16i
 *
 * \b Deprecation
 *
 * This kernel is deprecated.
 *
 * \b Overview
 *
 * <FIXME>
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_x4_quad_max_star_16i(short* target, short* src0, short* src1, short*
 * src2, short* src3, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li src0: The input vector 0.
 * \li src1: The input vector 1.
 * \li src2: The input vector 2.
 * \li src3: The input vector 3.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li target: The output value.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_16i_x4_quad_max_star_16i();
 *
 * volk_free(x);
 * \endcode
 */

#ifndef INCLUDED_volk_16i_x4_quad_max_star_16i_u_H
#define INCLUDED_volk_16i_x4_quad_max_star_16i_u_H

#include <inttypes.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_16i_x4_quad_max_star_16i_generic(short* target,
                                                         short* src0,
                                                         short* src1,
                                                         short* src2,
                                                         short* src3,
                                                         unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    short temp0 = 0;
    short temp1 = 0;
    for (i = 0; i < bound; ++i) {
        temp0 = (src0[i] > src1[i]) ? src0[i] : src1[i];
        temp1 = (src2[i] > src3[i]) ? src2[i] : src3[i];
        target[i] = (temp0 > temp1) ? temp0 : temp1;
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON

#include <arm_neon.h>

static inline void volk_16i_x4_quad_max_star_16i_neon(short* target,
                                                      short* src0,
                                                      short* src1,
                                                      short* src2,
                                                      short* src3,
                                                      unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    unsigned i;

    int16x8_t src0_vec, src1_vec, src2_vec, src3_vec;
    int16x8_t result1_vec, result2_vec;
    for (i = 0; i < eighth_points; ++i) {
        src0_vec = vld1q_s16(src0);
        src1_vec = vld1q_s16(src1);
        src2_vec = vld1q_s16(src2);
        src3_vec = vld1q_s16(src3);

        result1_vec = vmaxq_s16(src0_vec, src1_vec);
        result2_vec = vmaxq_s16(src2_vec, src3_vec);
        result1_vec = vmaxq_s16(result1_vec, result2_vec);

        vst1q_s16(target, result1_vec);
        src0 += 8;
        src1 += 8;
        src2 += 8;
        src3 += 8;
        target += 8;
    }

    short temp0 = 0;
    short temp1 = 0;
    for (i = eighth_points * 8; i < num_points; ++i) {
        temp0 = (*src0 > *src1) ? *src0 : *src1;
        temp1 = (*src2 > *src3) ? *src2 : *src3;
        *target++ = (temp0 > temp1) ? temp0 : temp1;
        src0++;
        src1++;
        src2++;
        src3++;
    }
}
#endif /* LV_HAVE_NEON */

#endif /* INCLUDED_volk_16i_x4_quad_max_star_16i_u_H */

#ifndef INCLUDED_volk_16i_x4_quad_max_star_16i_a_H
#define INCLUDED_volk_16i_x4_quad_max_star_16i_a_H

#include <inttypes.h>

#ifdef LV_HAVE_SSE2

#include <emmintrin.h>

static inline void volk_16i_x4_quad_max_star_16i_a_sse2(short* target,
                                                        short* src0,
                                                        short* src1,
                                                        short* src2,
                                                        short* src3,
                                                        unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = (num_bytes >> 4);
    int bound_copy = bound;
    int leftovers = (num_bytes >> 1) & 7;

    __m128i* p_target;
    const __m128i *p_src0, *p_src1, *p_src2, *p_src3;
    p_target = (__m128i*)target;
    p_src0 = (const __m128i*)src0;
    p_src1 = (const __m128i*)src1;
    p_src2 = (const __m128i*)src2;
    p_src3 = (const __m128i*)src3;

    __m128i xmm1, xmm2, xmm3, xmm4;

    while (bound_copy > 0) {
        xmm1 = _mm_load_si128(p_src0);
        xmm2 = _mm_load_si128(p_src1);
        xmm3 = _mm_load_si128(p_src2);
        xmm4 = _mm_load_si128(p_src3);

        xmm1 = _mm_max_epi16(xmm1, xmm2);
        xmm3 = _mm_max_epi16(xmm3, xmm4);
        xmm1 = _mm_max_epi16(xmm1, xmm3);

        _mm_store_si128(p_target, xmm1);

        p_src0 += 1;
        p_src1 += 1;
        p_src2 += 1;
        p_src3 += 1;
        p_target += 1;
        bound_copy -= 1;
    }

    short temp0 = 0;
    short temp1 = 0;
    for (i = bound * 8; i < (bound * 8) + leftovers; ++i) {
        temp0 = (src0[i] > src1[i]) ? src0[i] : src1[i];
        temp1 = (src2[i] > src3[i]) ? src2[i] : src3[i];
        target[i] = (temp0 > temp1) ? temp0 : temp1;
    }
    return;
}

#endif /* LV_HAVE_SSE2 */

#endif /* INCLUDED_volk_16i_x4_quad_max_star_16i_a_H */
