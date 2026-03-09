/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_permute_and_scalar_add
 *
 * \b Deprecation
 *
 * This kernel is deprecated, no replacement has been identified.
 *
 * \b Overview
 *
 * Gathers elements from a source vector using an index array, then adds
 * scalar-masked contributions from four control vectors. For each element i,
 * computes: target[i] = src0[permute_indexes[i]] + (cntl0[i] & scalars[0]) +
 * (cntl1[i] & scalars[1]) + (cntl2[i] & scalars[2]) + (cntl3[i] & scalars[3]).
 * The control vectors act as bitwise masks that selectively pass or block each scalar
 * value on a per-element basis.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_permute_and_scalar_add(short* target, short* src0, short*
 * permute_indexes, short* cntl0, short* cntl1, short* cntl2, short* cntl3, short*
 * scalars, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li src0: The source vector of short values to gather from.
 * \li permute_indexes: Vector of short indices used to look up elements in src0.
 * \li cntl0: First control mask vector of short values (num_points elements).
 * \li cntl1: Second control mask vector of short values (num_points elements).
 * \li cntl2: Third control mask vector of short values (num_points elements).
 * \li cntl3: Fourth control mask vector of short values (num_points elements).
 * \li scalars: Array of 4 short values, each bitwise ANDed with the corresponding
 * control vector.
 * \li num_points: The number of short values to process.
 *
 * \b Outputs
 * \li target: The output vector of short values (num_points elements).
 *
 * \b Example
 * \code
 * #include <volk/volk.h>
 * #include <stdio.h>
 *
 * int N = 8;
 * unsigned int alignment = volk_get_alignment();
 *
 * short* src0 = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* permute_indexes = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* cntl0 = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* cntl1 = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* cntl2 = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* cntl3 = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* scalars = (short*)volk_malloc(sizeof(short) * 4, alignment);
 * short* target = (short*)volk_malloc(sizeof(short) * N, alignment);
 *
 * for (unsigned int ii = 0; ii < N; ++ii) {
 *     src0[ii] = (short)(ii * 10);
 * }
 *
 * // Reverse permutation: read src0 in reverse order
 * for (unsigned int ii = 0; ii < N; ++ii) {
 *     permute_indexes[ii] = (short)(N - 1 - ii);
 * }
 *
 * // Control masks: -1 (all bits set) passes the scalar through, 0 blocks it
 * for (unsigned int ii = 0; ii < N; ++ii) {
 *     cntl0[ii] = -1;
 *     cntl1[ii] = 0;
 *     cntl2[ii] = 0;
 *     cntl3[ii] = 0;
 * }
 *
 * scalars[0] = 1;
 * scalars[1] = 2;
 * scalars[2] = 3;
 * scalars[3] = 4;
 *
 * // target[i] = src0[7-i] + (-1 & 1) + 0 + 0 + 0 = src0[7-i] + 1
 * volk_16i_permute_and_scalar_add(target, src0, permute_indexes, cntl0, cntl1, cntl2,
 *                                 cntl3, scalars, N);
 *
 * for (unsigned int ii = 0; ii < N; ++ii) {
 *     printf("target[%u] = %d\n", ii, target[ii]);
 * }
 *
 * volk_free(src0);
 * volk_free(permute_indexes);
 * volk_free(cntl0);
 * volk_free(cntl1);
 * volk_free(cntl2);
 * volk_free(cntl3);
 * volk_free(scalars);
 * volk_free(target);
 * \endcode
 */

#ifndef INCLUDED_volk_16i_permute_and_scalar_add_a_H
#define INCLUDED_volk_16i_permute_and_scalar_add_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2

#include <emmintrin.h>
#include <xmmintrin.h>

static inline void volk_16i_permute_and_scalar_add_a_sse2(short* target,
                                                          short* src0,
                                                          short* permute_indexes,
                                                          short* cntl0,
                                                          short* cntl1,
                                                          short* cntl2,
                                                          short* cntl3,
                                                          short* scalars,
                                                          unsigned int num_points)
{

    const unsigned int num_bytes = num_points * 2;

    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    __m128i *p_target, *p_cntl0, *p_cntl1, *p_cntl2, *p_cntl3, *p_scalars;

    short* p_permute_indexes = permute_indexes;

    p_target = (__m128i*)target;
    p_cntl0 = (__m128i*)cntl0;
    p_cntl1 = (__m128i*)cntl1;
    p_cntl2 = (__m128i*)cntl2;
    p_cntl3 = (__m128i*)cntl3;
    p_scalars = (__m128i*)scalars;

    int i = 0;

    int bound = (num_bytes >> 4);
    int leftovers = (num_bytes >> 1) & 7;

    xmm0 = _mm_load_si128(p_scalars);

    xmm1 = _mm_shufflelo_epi16(xmm0, 0);
    xmm2 = _mm_shufflelo_epi16(xmm0, 0x55);
    xmm3 = _mm_shufflelo_epi16(xmm0, 0xaa);
    xmm4 = _mm_shufflelo_epi16(xmm0, 0xff);

    xmm1 = _mm_shuffle_epi32(xmm1, 0x00);
    xmm2 = _mm_shuffle_epi32(xmm2, 0x00);
    xmm3 = _mm_shuffle_epi32(xmm3, 0x00);
    xmm4 = _mm_shuffle_epi32(xmm4, 0x00);


    for (; i < bound; ++i) {
        xmm0 = _mm_setzero_si128();
        xmm5 = _mm_setzero_si128();
        xmm6 = _mm_setzero_si128();
        xmm7 = _mm_setzero_si128();

        xmm0 = _mm_insert_epi16(xmm0, src0[p_permute_indexes[0]], 0);
        xmm5 = _mm_insert_epi16(xmm5, src0[p_permute_indexes[1]], 1);
        xmm6 = _mm_insert_epi16(xmm6, src0[p_permute_indexes[2]], 2);
        xmm7 = _mm_insert_epi16(xmm7, src0[p_permute_indexes[3]], 3);
        xmm0 = _mm_insert_epi16(xmm0, src0[p_permute_indexes[4]], 4);
        xmm5 = _mm_insert_epi16(xmm5, src0[p_permute_indexes[5]], 5);
        xmm6 = _mm_insert_epi16(xmm6, src0[p_permute_indexes[6]], 6);
        xmm7 = _mm_insert_epi16(xmm7, src0[p_permute_indexes[7]], 7);

        xmm0 = _mm_add_epi16(xmm0, xmm5);
        xmm6 = _mm_add_epi16(xmm6, xmm7);

        p_permute_indexes += 8;

        xmm0 = _mm_add_epi16(xmm0, xmm6);

        xmm5 = _mm_load_si128(p_cntl0);
        xmm6 = _mm_load_si128(p_cntl1);
        xmm7 = _mm_load_si128(p_cntl2);

        xmm5 = _mm_and_si128(xmm5, xmm1);
        xmm6 = _mm_and_si128(xmm6, xmm2);
        xmm7 = _mm_and_si128(xmm7, xmm3);

        xmm0 = _mm_add_epi16(xmm0, xmm5);

        xmm5 = _mm_load_si128(p_cntl3);

        xmm6 = _mm_add_epi16(xmm6, xmm7);

        p_cntl0 += 1;

        xmm5 = _mm_and_si128(xmm5, xmm4);

        xmm0 = _mm_add_epi16(xmm0, xmm6);

        p_cntl1 += 1;
        p_cntl2 += 1;

        xmm0 = _mm_add_epi16(xmm0, xmm5);

        p_cntl3 += 1;

        _mm_store_si128(p_target, xmm0);

        p_target += 1;
    }

    for (i = bound * 8; i < (bound * 8) + leftovers; ++i) {
        target[i] = src0[permute_indexes[i]] + (cntl0[i] & scalars[0]) +
                    (cntl1[i] & scalars[1]) + (cntl2[i] & scalars[2]) +
                    (cntl3[i] & scalars[3]);
    }
}
#endif /*LV_HAVE_SSE*/


#ifdef LV_HAVE_GENERIC
static inline void volk_16i_permute_and_scalar_add_generic(short* target,
                                                           short* src0,
                                                           short* permute_indexes,
                                                           short* cntl0,
                                                           short* cntl1,
                                                           short* cntl2,
                                                           short* cntl3,
                                                           short* scalars,
                                                           unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    for (i = 0; i < bound; ++i) {
        target[i] = src0[permute_indexes[i]] + (cntl0[i] & scalars[0]) +
                    (cntl1[i] & scalars[1]) + (cntl2[i] & scalars[2]) +
                    (cntl3[i] & scalars[3]);
    }
}

#endif /*LV_HAVE_GENERIC*/

#endif /*INCLUDED_volk_16i_permute_and_scalar_add_a_H*/
