/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8ic_deinterleave_real_8i
 *
 * \b Overview
 *
 * Deinterleaves the complex 8-bit char vector into just the I (real)
 * vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8ic_deinterleave_real_8i(int8_t* iBuffer, const lv_8sc_t* complexVector,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The I buffer output data.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_8ic_deinterleave_real_8i();
 *
 * volk_free(x);
 * \endcode
 */

#ifndef INCLUDED_VOLK_8sc_DEINTERLEAVE_REAL_8s_ALIGNED8_H
#define INCLUDED_VOLK_8sc_DEINTERLEAVE_REAL_8s_ALIGNED8_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_8i_a_avx2(int8_t* iBuffer,
                                                        const lv_8sc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    __m256i moveMask1 = _mm256_set_epi8(0x80,
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
                                        0,
                                        0x80,
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
                                        0);
    __m256i moveMask2 = _mm256_set_epi8(14,
                                        12,
                                        10,
                                        8,
                                        6,
                                        4,
                                        2,
                                        0,
                                        0x80,
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
                                        0,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80);
    __m256i complexVal1, complexVal2, outputVal;

    unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {

        complexVal1 = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal2 = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        complexVal1 = _mm256_shuffle_epi8(complexVal1, moveMask1);
        complexVal2 = _mm256_shuffle_epi8(complexVal2, moveMask2);
        outputVal = _mm256_or_si256(complexVal1, complexVal2);
        outputVal = _mm256_permute4x64_epi64(outputVal, 0xd8);

        _mm256_store_si256((__m256i*)iBufferPtr, outputVal);
        iBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_8ic_deinterleave_real_8i_a_ssse3(int8_t* iBuffer,
                                                         const lv_8sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    __m128i moveMask1 = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i moveMask2 = _mm_set_epi8(
        14, 12, 10, 8, 6, 4, 2, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
    __m128i complexVal1, complexVal2, outputVal;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal1 = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal2 = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        complexVal1 = _mm_shuffle_epi8(complexVal1, moveMask1);
        complexVal2 = _mm_shuffle_epi8(complexVal2, moveMask2);

        outputVal = _mm_or_si128(complexVal1, complexVal2);

        _mm_store_si128((__m128i*)iBufferPtr, outputVal);
        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSSE3 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_8i_a_avx(int8_t* iBuffer,
                                                       const lv_8sc_t* complexVector,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    __m128i moveMaskL = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i moveMaskH = _mm_set_epi8(
        14, 12, 10, 8, 6, 4, 2, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
    __m256i complexVal1, complexVal2, outputVal;
    __m128i complexVal1H, complexVal1L, complexVal2H, complexVal2L, outputVal1,
        outputVal2;

    unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {

        complexVal1 = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal2 = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        complexVal1H = _mm256_extractf128_si256(complexVal1, 1);
        complexVal1L = _mm256_extractf128_si256(complexVal1, 0);
        complexVal2H = _mm256_extractf128_si256(complexVal2, 1);
        complexVal2L = _mm256_extractf128_si256(complexVal2, 0);

        complexVal1H = _mm_shuffle_epi8(complexVal1H, moveMaskH);
        complexVal1L = _mm_shuffle_epi8(complexVal1L, moveMaskL);
        outputVal1 = _mm_or_si128(complexVal1H, complexVal1L);


        complexVal2H = _mm_shuffle_epi8(complexVal2H, moveMaskH);
        complexVal2L = _mm_shuffle_epi8(complexVal2L, moveMaskL);
        outputVal2 = _mm_or_si128(complexVal2H, complexVal2L);

        __m256i dummy = _mm256_setzero_si256();
        outputVal = _mm256_insertf128_si256(dummy, outputVal1, 0);
        outputVal = _mm256_insertf128_si256(outputVal, outputVal2, 1);


        _mm256_store_si256((__m256i*)iBufferPtr, outputVal);
        iBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_GENERIC

static inline void volk_8ic_deinterleave_real_8i_generic(int8_t* iBuffer,
                                                         const lv_8sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_8ic_deinterleave_real_8i_neon(int8_t* iBuffer,
                                                      const lv_8sc_t* complexVector,
                                                      unsigned int num_points)
{
    unsigned int number;
    unsigned int sixteenth_points = num_points / 16;

    int8x16x2_t input_vector;
    for (number = 0; number < sixteenth_points; ++number) {
        input_vector = vld2q_s8((int8_t*)complexVector);
        vst1q_s8(iBuffer, input_vector.val[0]);
        iBuffer += 16;
        complexVector += 16;
    }

    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    for (number = sixteenth_points * 16; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_8ic_deinterleave_real_8i_neonv8(int8_t* iBuffer,
                                                        const lv_8sc_t* complexVector,
                                                        unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    for (unsigned int number = 0; number < thirtysecondPoints; number++) {
        int8x16x2_t cplx0 = vld2q_s8((const int8_t*)complexVector);
        int8x16x2_t cplx1 = vld2q_s8((const int8_t*)complexVector + 32);
        __VOLK_PREFETCH((const int8_t*)complexVector + 64);

        vst1q_s8(iBuffer, cplx0.val[0]);
        vst1q_s8(iBuffer + 16, cplx1.val[0]);

        iBuffer += 32;
        complexVector += 32;
    }

    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    for (unsigned int number = thirtysecondPoints * 32; number < num_points; number++) {
        *iBuffer++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEONV8 */


#endif /* INCLUDED_VOLK_8sc_DEINTERLEAVE_REAL_8s_ALIGNED8_H */

#ifndef INCLUDED_VOLK_8sc_DEINTERLEAVE_REAL_8s_UNALIGNED8_H
#define INCLUDED_VOLK_8sc_DEINTERLEAVE_REAL_8s_UNALIGNED8_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_8i_u_avx2(int8_t* iBuffer,
                                                        const lv_8sc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    __m256i moveMask1 = _mm256_set_epi8(0x80,
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
                                        0,
                                        0x80,
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
                                        0);
    __m256i moveMask2 = _mm256_set_epi8(14,
                                        12,
                                        10,
                                        8,
                                        6,
                                        4,
                                        2,
                                        0,
                                        0x80,
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
                                        0,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80);
    __m256i complexVal1, complexVal2, outputVal;

    unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {

        complexVal1 = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal2 = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        complexVal1 = _mm256_shuffle_epi8(complexVal1, moveMask1);
        complexVal2 = _mm256_shuffle_epi8(complexVal2, moveMask2);
        outputVal = _mm256_or_si256(complexVal1, complexVal2);
        outputVal = _mm256_permute4x64_epi64(outputVal, 0xd8);

        _mm256_storeu_si256((__m256i*)iBufferPtr, outputVal);
        iBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_8ic_deinterleave_real_8i_rvv(int8_t* iBuffer,
                                                     const lv_8sc_t* complexVector,
                                                     unsigned int num_points)
{
    const uint16_t* in = (const uint16_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in += vl, iBuffer += vl) {
        vl = __riscv_vsetvl_e16m8(n);
        vuint16m8_t vc = __riscv_vle16_v_u16m8(in, vl);
        __riscv_vse8((uint8_t*)iBuffer, __riscv_vnsrl(vc, 0, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_VOLK_8sc_DEINTERLEAVE_REAL_8s_UNALIGNED8_H */
