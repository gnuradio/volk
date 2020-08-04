/* -*- c++ -*- */
/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

/*!
 * \page volk_4ic_deinterleave_8i_x2
 *
 * \b Overview
 *
 * Deinterleaves the complex 4-bit integer vector into I & Q vector data
 * and converts them to 8-bit integers.
 *
 * <b> Dispatcher Prototype</b>
 * \code
 * void volk_4ic_deinterleave_8i_x2(int8_t* iBuffer, int8t_t* qBuffer, const int8_t*
 * complexVector, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The I buffer output data.
 * \li qBuffer: The Q buffer output data.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_4ic_deinterleave_8i_x2();
 *
 * volk_free(x);
 * \endcode
 *
 */

#ifndef INCLUDED_volk_4ic_deinterleave_8i_x2_a_H
#define INCLUDED_volk_4ic_deinterleave_8i_x2_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_4ic_deinterleave_8i_x2_generic(int8_t* iBuffer,
                                                       int8_t* qBuffer,
                                                       const int8_t* complexVector,
                                                       unsigned int num_points)
{
    const int8_t* complexVectorPtr = complexVector;
    int8_t* iBufferPtr = iBuffer;
    int8_t* qBufferPtr = qBuffer;
    for (unsigned int i = 0; i < num_points; i++) {
        *iBufferPtr++ = (int8_t)(*complexVectorPtr) >> 4;
        *qBufferPtr++ = ((int8_t)(*complexVectorPtr++) << 4) >> 4;
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_4ic_deinterleave_8i_x2_u_sse2(int8_t* iBuffer,
                                                      int8_t* qBuffer,
                                                      const int8_t* complexVector,
                                                      unsigned int num_points)
{
    // SSE2 algorithm was written by Andrey Semashev, licensed as CC-BY-SA
    // https://stackoverflow.com/questions/63200053/deinterleve-vector-of-nibbles-using-simd
    const __m128i mask = _mm_set1_epi32(0x0F0F0F0F);
    const __m128i signed_max = _mm_set1_epi32(0x07070707);

    int8_t* complexVectorPtr = complexVector;
    int8_t* iBufferPtr = iBuffer;
    int8_t* qBufferPtr = qBuffer;

    const unsigned int num_blocks = num_points / 16;
    for (int block_index = 0; block_index < num_blocks; block_index++) {
        // Load and deinterleave input half-bytes
        __m128i input_even = _mm_loadu_si128((__m128i*)complexVectorPtr);
        __m128i input_odd = _mm_srli_epi32(input_even, 4);
        complexVectorPtr += 16;

        input_even = _mm_and_si128(input_even, mask);
        input_odd = _mm_and_si128(input_odd, mask);

        // Get the sign bits
        __m128i sign_even = _mm_cmpgt_epi8(input_even, signed_max);
        __m128i sign_odd = _mm_cmpgt_epi8(input_odd, signed_max);

        // Combine sign bits with deinterleaved input
        input_even = _mm_or_si128(input_even, _mm_andnot_si128(mask, sign_even));
        input_odd = _mm_or_si128(input_odd, _mm_andnot_si128(mask, sign_odd));

        // Store the results
        _mm_storeu_si128((__m128i*)iBufferPtr, input_even);
        _mm_storeu_si128((__m128i*)qBufferPtr, input_odd);
        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    const int remainder = num_points % 16;
    volk_4ic_deinterleave_8i_x2_generic(
        iBufferPtr, qBufferPtr, complexVectorPtr, remainder);
}

#endif /* LV_HAVE_SSE2 */
#endif /* INCLUDED_volk_4ic_deinterleave_8i_x2_u_H */
