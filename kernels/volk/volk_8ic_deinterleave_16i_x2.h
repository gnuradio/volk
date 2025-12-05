/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8ic_deinterleave_16i_x2
 *
 * \b Overview
 *
 * Deinterleaves the complex 8-bit char vector into I & Q vector data
 * and converts them to 16-bit shorts.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8ic_deinterleave_16i_x2(int16_t* iBuffer, int16_t* qBuffer, const lv_8sc_t*
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
 * volk_8ic_deinterleave_16i_x2();
 *
 * volk_free(x);
 * \endcode
 */

#ifndef INCLUDED_volk_8ic_deinterleave_16i_x2_a_H
#define INCLUDED_volk_8ic_deinterleave_16i_x2_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8ic_deinterleave_16i_x2_a_avx2(int16_t* iBuffer,
                                                       int16_t* qBuffer,
                                                       const lv_8sc_t* complexVector,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;
    __m256i MoveMask = _mm256_set_epi8(15,
                                       13,
                                       11,
                                       9,
                                       7,
                                       5,
                                       3,
                                       1,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0,
                                       15,
                                       13,
                                       11,
                                       9,
                                       7,
                                       5,
                                       3,
                                       1,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0);
    __m256i complexVal, iOutputVal, qOutputVal;
    __m128i iOutputVal0, qOutputVal0;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        complexVal = _mm256_shuffle_epi8(complexVal, MoveMask);
        complexVal = _mm256_permute4x64_epi64(complexVal, 0xd8);

        iOutputVal0 = _mm256_extracti128_si256(complexVal, 0);
        qOutputVal0 = _mm256_extracti128_si256(complexVal, 1);

        iOutputVal = _mm256_cvtepi8_epi16(iOutputVal0);
        iOutputVal = _mm256_slli_epi16(iOutputVal, 8);

        qOutputVal = _mm256_cvtepi8_epi16(qOutputVal0);
        qOutputVal = _mm256_slli_epi16(qOutputVal, 8);

        _mm256_store_si256((__m256i*)iBufferPtr, iOutputVal);
        _mm256_store_si256((__m256i*)qBufferPtr, qOutputVal);

        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ =
            ((int16_t)*complexVectorPtr++) *
            256; // load 8 bit Complexvector into 16 bit, shift left by 8 bits and store
        *qBufferPtr++ = ((int16_t)*complexVectorPtr++) * 256;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_8ic_deinterleave_16i_x2_a_sse4_1(int16_t* iBuffer,
                                                         int16_t* qBuffer,
                                                         const lv_8sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;
    __m128i iMoveMask = _mm_set_epi8(0x80,
                                     0x80,
                                     0x80,
                                     0x80,
                                     0x80,
                                     0x80,
                                     0x80,
                                     0x80,
                                     14,
                                     12,
                                     10,
                                     8,
                                     6,
                                     4,
                                     2,
                                     0); // set 16 byte values
    __m128i qMoveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 15, 13, 11, 9, 7, 5, 3, 1);
    __m128i complexVal, iOutputVal, qOutputVal;

    unsigned int eighthPoints = num_points / 8;

    for (number = 0; number < eighthPoints; number++) {
        complexVal = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 16; // aligned load

        iOutputVal = _mm_shuffle_epi8(complexVal,
                                      iMoveMask); // shuffle 16 bytes of 128bit complexVal
        qOutputVal = _mm_shuffle_epi8(complexVal, qMoveMask);

        iOutputVal = _mm_cvtepi8_epi16(iOutputVal); // fills 2-byte sign extended versions
                                                    // of lower 8 bytes of input to output
        iOutputVal =
            _mm_slli_epi16(iOutputVal, 8); // shift in left by 8 bits, each of the 8
                                           // 16-bit integers, shift in with zeros

        qOutputVal = _mm_cvtepi8_epi16(qOutputVal);
        qOutputVal = _mm_slli_epi16(qOutputVal, 8);

        _mm_store_si128((__m128i*)iBufferPtr, iOutputVal); // aligned store
        _mm_store_si128((__m128i*)qBufferPtr, qOutputVal);

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ =
            ((int16_t)*complexVectorPtr++) *
            256; // load 8 bit Complexvector into 16 bit, shift left by 8 bits and store
        *qBufferPtr++ = ((int16_t)*complexVectorPtr++) * 256;
    }
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_8ic_deinterleave_16i_x2_a_avx(int16_t* iBuffer,
                                                      int16_t* qBuffer,
                                                      const lv_8sc_t* complexVector,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;
    __m128i iMoveMask = _mm_set_epi8(0x80,
                                     0x80,
                                     0x80,
                                     0x80,
                                     0x80,
                                     0x80,
                                     0x80,
                                     0x80,
                                     14,
                                     12,
                                     10,
                                     8,
                                     6,
                                     4,
                                     2,
                                     0); // set 16 byte values
    __m128i qMoveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 15, 13, 11, 9, 7, 5, 3, 1);
    __m256i complexVal, iOutputVal, qOutputVal;
    __m128i complexVal1, complexVal0;
    __m128i iOutputVal1, iOutputVal0, qOutputVal1, qOutputVal0;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32; // aligned load

        // Extract from complexVal to iOutputVal and qOutputVal
        complexVal1 = _mm256_extractf128_si256(complexVal, 1);
        complexVal0 = _mm256_extractf128_si256(complexVal, 0);

        iOutputVal1 = _mm_shuffle_epi8(
            complexVal1, iMoveMask); // shuffle 16 bytes of 128bit complexVal
        iOutputVal0 = _mm_shuffle_epi8(complexVal0, iMoveMask);
        qOutputVal1 = _mm_shuffle_epi8(complexVal1, qMoveMask);
        qOutputVal0 = _mm_shuffle_epi8(complexVal0, qMoveMask);

        iOutputVal1 =
            _mm_cvtepi8_epi16(iOutputVal1); // fills 2-byte sign extended versions of
                                            // lower 8 bytes of input to output
        iOutputVal1 =
            _mm_slli_epi16(iOutputVal1, 8); // shift in left by 8 bits, each of the 8
                                            // 16-bit integers, shift in with zeros
        iOutputVal0 = _mm_cvtepi8_epi16(iOutputVal0);
        iOutputVal0 = _mm_slli_epi16(iOutputVal0, 8);

        qOutputVal1 = _mm_cvtepi8_epi16(qOutputVal1);
        qOutputVal1 = _mm_slli_epi16(qOutputVal1, 8);
        qOutputVal0 = _mm_cvtepi8_epi16(qOutputVal0);
        qOutputVal0 = _mm_slli_epi16(qOutputVal0, 8);

        // Pack iOutputVal0,1 to iOutputVal
        __m256i dummy = _mm256_setzero_si256();
        iOutputVal = _mm256_insertf128_si256(dummy, iOutputVal0, 0);
        iOutputVal = _mm256_insertf128_si256(iOutputVal, iOutputVal1, 1);
        qOutputVal = _mm256_insertf128_si256(dummy, qOutputVal0, 0);
        qOutputVal = _mm256_insertf128_si256(qOutputVal, qOutputVal1, 1);

        _mm256_store_si256((__m256i*)iBufferPtr, iOutputVal); // aligned store
        _mm256_store_si256((__m256i*)qBufferPtr, qOutputVal);

        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ =
            ((int16_t)*complexVectorPtr++) *
            256; // load 8 bit Complexvector into 16 bit, shift left by 8 bits and store
        *qBufferPtr++ = ((int16_t)*complexVectorPtr++) * 256;
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_GENERIC

static inline void volk_8ic_deinterleave_16i_x2_generic(int16_t* iBuffer,
                                                        int16_t* qBuffer,
                                                        const lv_8sc_t* complexVector,
                                                        unsigned int num_points)
{
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;
    unsigned int number;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = (int16_t)(*complexVectorPtr++) * 256;
        *qBufferPtr++ = (int16_t)(*complexVectorPtr++) * 256;
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_8ic_deinterleave_16i_x2_a_H */

#ifndef INCLUDED_volk_8ic_deinterleave_16i_x2_u_H
#define INCLUDED_volk_8ic_deinterleave_16i_x2_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8ic_deinterleave_16i_x2_u_avx2(int16_t* iBuffer,
                                                       int16_t* qBuffer,
                                                       const lv_8sc_t* complexVector,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;
    __m256i MoveMask = _mm256_set_epi8(15,
                                       13,
                                       11,
                                       9,
                                       7,
                                       5,
                                       3,
                                       1,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0,
                                       15,
                                       13,
                                       11,
                                       9,
                                       7,
                                       5,
                                       3,
                                       1,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0);
    __m256i complexVal, iOutputVal, qOutputVal;
    __m128i iOutputVal0, qOutputVal0;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        complexVal = _mm256_shuffle_epi8(complexVal, MoveMask);
        complexVal = _mm256_permute4x64_epi64(complexVal, 0xd8);

        iOutputVal0 = _mm256_extracti128_si256(complexVal, 0);
        qOutputVal0 = _mm256_extracti128_si256(complexVal, 1);

        iOutputVal = _mm256_cvtepi8_epi16(iOutputVal0);
        iOutputVal = _mm256_slli_epi16(iOutputVal, 8);

        qOutputVal = _mm256_cvtepi8_epi16(qOutputVal0);
        qOutputVal = _mm256_slli_epi16(qOutputVal, 8);

        _mm256_storeu_si256((__m256i*)iBufferPtr, iOutputVal);
        _mm256_storeu_si256((__m256i*)qBufferPtr, qOutputVal);

        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ =
            ((int16_t)*complexVectorPtr++) *
            256; // load 8 bit Complexvector into 16 bit, shift left by 8 bits and store
        *qBufferPtr++ = ((int16_t)*complexVectorPtr++) * 256;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_8ic_deinterleave_16i_x2_neon(int16_t* iBuffer,
                                                     int16_t* qBuffer,
                                                     const lv_8sc_t* complexVector,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    for (; number < eighth_points; number++) {
        int8x8x2_t input = vld2_s8(complexVectorPtr);
        complexVectorPtr += 16;

        int16x8_t iVal = vshll_n_s8(input.val[0], 8);
        int16x8_t qVal = vshll_n_s8(input.val[1], 8);

        vst1q_s16(iBufferPtr, iVal);
        vst1q_s16(qBufferPtr, qVal);
        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((int16_t)*complexVectorPtr++) * 256;
        *qBufferPtr++ = ((int16_t)*complexVectorPtr++) * 256;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_8ic_deinterleave_16i_x2_rvv(int16_t* iBuffer,
                                                    int16_t* qBuffer,
                                                    const lv_8sc_t* complexVector,
                                                    unsigned int num_points)
{
    const uint16_t* in = (const uint16_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e16m8(n);
        vuint16m8_t vc = __riscv_vle16_v_u16m8(in, vl);
        vuint16m8_t vr = __riscv_vsll(vc, 8, vl);
        vuint16m8_t vi = __riscv_vand(vc, 0xFF00, vl);
        __riscv_vse16((uint16_t*)iBuffer, vr, vl);
        __riscv_vse16((uint16_t*)qBuffer, vi, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_8ic_deinterleave_16i_x2_u_H */
