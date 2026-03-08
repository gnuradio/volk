/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 * Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_max_star_horizontal_16i
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
 * void volk_16i_max_star_horizontal_16i(short* target, short* src0, unsigned int
 * num_points); \endcode
 *
 * \b Inputs
 * \li src0: The input vector.
 * \li num_points: The number of complex data points.
 *
 * \b Outputs
 * \li target: The output value of the max* operation.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_16i_max_star_horizontal_16i();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_16i_max_star_horizontal_16i_u_H
#define INCLUDED_volk_16i_max_star_horizontal_16i_u_H

#include <volk/volk_common.h>

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_16i_max_star_horizontal_16i_generic(int16_t* target,
                                                            int16_t* src0,
                                                            unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    for (i = 0; i < bound; i += 2) {
        target[i >> 1] = (src0[i] > src0[i + 1]) ? src0[i] : src0[i + 1];
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON

#include <arm_neon.h>
static inline void volk_16i_max_star_horizontal_16i_neon(int16_t* target,
                                                         int16_t* src0,
                                                         unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 16;
    unsigned number;
    int16x8x2_t input_vec;
    int16x8_t max_vec;
    for (number = 0; number < eighth_points; ++number) {
        input_vec = vld2q_s16(src0);
        max_vec = vmaxq_s16(input_vec.val[0], input_vec.val[1]);
        vst1q_s16(target, max_vec);
        src0 += 16;
        target += 8;
    }
    for (number = 0; number < num_points % 16; number += 2) {
        target[number >> 1] = (src0[number] > src0[number + 1])
                                  ? src0[number]
                                  : src0[number + 1];
    }
}
#endif /* LV_HAVE_NEON */

#endif /* INCLUDED_volk_16i_max_star_horizontal_16i_u_H */

#ifndef INCLUDED_volk_16i_max_star_horizontal_16i_a_H
#define INCLUDED_volk_16i_max_star_horizontal_16i_a_H

#include <volk/volk_common.h>

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSSE3

#include <emmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

static inline void volk_16i_max_star_horizontal_16i_a_ssse3(int16_t* target,
                                                            int16_t* src0,
                                                            unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    /* Shuffle masks to deinterleave even/odd 16-bit elements */
    static const uint8_t shuf_even_lo[16] = {
        0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    };
    static const uint8_t shuf_even_hi[16] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d
    };
    static const uint8_t shuf_odd_lo[16] = {
        0x02, 0x03, 0x06, 0x07, 0x0a, 0x0b, 0x0e, 0x0f,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    };
    static const uint8_t shuf_odd_hi[16] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x02, 0x03, 0x06, 0x07, 0x0a, 0x0b, 0x0e, 0x0f
    };

    __m128i xmm_even_lo = _mm_load_si128((const __m128i*)shuf_even_lo);
    __m128i xmm_even_hi = _mm_load_si128((const __m128i*)shuf_even_hi);
    __m128i xmm_odd_lo = _mm_load_si128((const __m128i*)shuf_odd_lo);
    __m128i xmm_odd_hi = _mm_load_si128((const __m128i*)shuf_odd_hi);

    __m128i* p_target = (__m128i*)target;
    const __m128i* p_src0 = (const __m128i*)src0;

    int bound = num_bytes >> 5;
    int intermediate = (num_bytes >> 4) & 1;
    int leftovers = (num_bytes >> 1) & 7;

    int i = 0;

    for (i = 0; i < bound; ++i) {
        __m128i xmm0 = _mm_load_si128(&p_src0[0]);
        __m128i xmm1 = _mm_load_si128(&p_src0[1]);
        p_src0 += 2;

        __m128i evens = _mm_or_si128(
            _mm_shuffle_epi8(xmm0, xmm_even_lo),
            _mm_shuffle_epi8(xmm1, xmm_even_hi));
        __m128i odds = _mm_or_si128(
            _mm_shuffle_epi8(xmm0, xmm_odd_lo),
            _mm_shuffle_epi8(xmm1, xmm_odd_hi));

        _mm_store_si128(p_target, _mm_max_epi16(evens, odds));
        p_target += 1;
    }

    if (intermediate) {
        __m128i xmm0 = _mm_load_si128(p_src0);
        p_src0 += 1;

        __m128i evens = _mm_shuffle_epi8(xmm0, xmm_even_lo);
        __m128i odds = _mm_shuffle_epi8(xmm0, xmm_odd_lo);
        __m128i result = _mm_max_epi16(evens, odds);

        _mm_storel_pd((double*)p_target, bit128_p(&result)->double_vec);
        p_target = (__m128i*)((int8_t*)p_target + 8);
    }

    for (i = (bound << 4) + (intermediate << 3);
         i < (bound << 4) + (intermediate << 3) + leftovers;
         i += 2) {
        target[i >> 1] = (src0[i] > src0[i + 1]) ? src0[i] : src0[i + 1];
    }
}

#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_NEONV7
extern void volk_16i_max_star_horizontal_16i_a_neonasm(int16_t* target,
                                                       int16_t* src0,
                                                       unsigned int num_points);
#endif /* LV_HAVE_NEONV7 */

#endif /* INCLUDED_volk_16i_max_star_horizontal_16i_a_H */
