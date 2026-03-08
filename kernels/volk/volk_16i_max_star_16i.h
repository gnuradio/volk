/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_max_star_16i
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
 * void volk_16i_max_star_16i(short* target, short* src0, unsigned int num_points);
 * \endcode
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
 * volk_16i_max_star_16i();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_16i_max_star_16i_u_H
#define INCLUDED_volk_16i_max_star_16i_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_16i_max_star_16i_generic(short* target, short* src0, unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    short candidate = src0[0];
    for (i = 1; i < bound; ++i) {
        candidate = (candidate > src0[i]) ? candidate : src0[i];
    }
    target[0] = candidate;
}

#endif /*LV_HAVE_GENERIC*/

#endif /* INCLUDED_volk_16i_max_star_16i_u_H */

#ifndef INCLUDED_volk_16i_max_star_16i_a_H
#define INCLUDED_volk_16i_max_star_16i_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSSE3

#include <emmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

static inline void
volk_16i_max_star_16i_a_ssse3(short* target, short* src0, unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    short candidate = src0[0];
    short cands[8];
    __m128i xmm0, xmm1, xmm3, xmm4, xmm5, xmm6;

    const __m128i* p_src0;

    p_src0 = (const __m128i*)src0;

    int bound = num_bytes >> 4;
    int leftovers = (num_bytes >> 1) & 7;

    int i = 0;

    xmm0 = _mm_set1_epi16(candidate);

    for (i = 0; i < bound; ++i) {
        xmm1 = _mm_load_si128(p_src0);
        p_src0 += 1;
        // xmm2 = _mm_sub_epi16(xmm1, xmm0);

        xmm3 = _mm_cmpgt_epi16(xmm0, xmm1);
        xmm4 = _mm_cmpeq_epi16(xmm0, xmm1);
        xmm5 = _mm_cmpgt_epi16(xmm1, xmm0);

        xmm6 = _mm_xor_si128(xmm4, xmm5);

        xmm3 = _mm_and_si128(xmm3, xmm0);
        xmm4 = _mm_and_si128(xmm6, xmm1);

        xmm0 = _mm_add_epi16(xmm3, xmm4);
    }

    _mm_store_si128((__m128i*)cands, xmm0);

    for (i = 0; i < 8; ++i) {
        candidate = (candidate > cands[i]) ? candidate : cands[i];
    }

    for (i = 0; i < leftovers; ++i) {
        candidate = (candidate > src0[(bound << 3) + i])
                        ? candidate
                        : src0[(bound << 3) + i];
    }

    target[0] = candidate;
}

#endif /*LV_HAVE_SSSE3*/

#endif /* INCLUDED_volk_16i_max_star_16i_a_H */
