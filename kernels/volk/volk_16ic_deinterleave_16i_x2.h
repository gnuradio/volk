/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_deinterleave_16i_x2
 *
 * \b Overview
 *
 * Deinterleaves the complex 16 bit vector into I & Q vector data.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_deinterleave_16i_x2(int16_t* iBuffer, int16_t* qBuffer, const lv_16sc_t*
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
 * volk_16ic_deinterleave_16i_x2();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_deinterleave_16i_x2_a_H
#define INCLUDED_volk_16ic_deinterleave_16i_x2_a_H

#include <inttypes.h>
#include <stdio.h>
#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_a_avx2(int16_t* iBuffer,
                                                        int16_t* qBuffer,
                                                        const lv_16sc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    __m256i MoveMask = _mm256_set_epi8(15,
                                       14,
                                       11,
                                       10,
                                       7,
                                       6,
                                       3,
                                       2,
                                       13,
                                       12,
                                       9,
                                       8,
                                       5,
                                       4,
                                       1,
                                       0,
                                       15,
                                       14,
                                       11,
                                       10,
                                       7,
                                       6,
                                       3,
                                       2,
                                       13,
                                       12,
                                       9,
                                       8,
                                       5,
                                       4,
                                       1,
                                       0);

    __m256i iMove2, iMove1;
    __m256i complexVal1, complexVal2, iOutputVal, qOutputVal;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal1 = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal2 = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        iMove2 = _mm256_shuffle_epi8(complexVal2, MoveMask);
        iMove1 = _mm256_shuffle_epi8(complexVal1, MoveMask);

        iOutputVal = _mm256_permute2x128_si256(_mm256_permute4x64_epi64(iMove1, 0x08),
                                               _mm256_permute4x64_epi64(iMove2, 0x80),
                                               0x30);
        qOutputVal = _mm256_permute2x128_si256(_mm256_permute4x64_epi64(iMove1, 0x0d),
                                               _mm256_permute4x64_epi64(iMove2, 0xd0),
                                               0x30);

        _mm256_store_si256((__m256i*)iBufferPtr, iOutputVal);
        _mm256_store_si256((__m256i*)qBufferPtr, qOutputVal);

        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    int16_t* int16ComplexVectorPtr = (int16_t*)complexVectorPtr;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *int16ComplexVectorPtr++;
        *qBufferPtr++ = *int16ComplexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_a_ssse3(int16_t* iBuffer,
                                                         int16_t* qBuffer,
                                                         const lv_16sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    __m128i iMoveMask1 = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 13, 12, 9, 8, 5, 4, 1, 0);
    __m128i iMoveMask2 = _mm_set_epi8(
        13, 12, 9, 8, 5, 4, 1, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

    __m128i qMoveMask1 = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 15, 14, 11, 10, 7, 6, 3, 2);
    __m128i qMoveMask2 = _mm_set_epi8(
        15, 14, 11, 10, 7, 6, 3, 2, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

    __m128i complexVal1, complexVal2, iOutputVal, qOutputVal;

    unsigned int eighthPoints = num_points / 8;

    for (number = 0; number < eighthPoints; number++) {
        complexVal1 = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal2 = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        iOutputVal = _mm_or_si128(_mm_shuffle_epi8(complexVal1, iMoveMask1),
                                  _mm_shuffle_epi8(complexVal2, iMoveMask2));
        qOutputVal = _mm_or_si128(_mm_shuffle_epi8(complexVal1, qMoveMask1),
                                  _mm_shuffle_epi8(complexVal2, qMoveMask2));

        _mm_store_si128((__m128i*)iBufferPtr, iOutputVal);
        _mm_store_si128((__m128i*)qBufferPtr, qOutputVal);

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    int16_t* int16ComplexVectorPtr = (int16_t*)complexVectorPtr;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *int16ComplexVectorPtr++;
        *qBufferPtr++ = *int16ComplexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_a_sse2(int16_t* iBuffer,
                                                        int16_t* qBuffer,
                                                        const lv_16sc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;
    __m128i complexVal1, complexVal2, iComplexVal1, iComplexVal2, qComplexVal1,
        qComplexVal2, iOutputVal, qOutputVal;
    __m128i lowMask = _mm_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF);
    __m128i highMask = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0);

    unsigned int eighthPoints = num_points / 8;

    for (number = 0; number < eighthPoints; number++) {
        complexVal1 = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 8;
        complexVal2 = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 8;

        iComplexVal1 = _mm_shufflelo_epi16(complexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        iComplexVal1 = _mm_shufflehi_epi16(iComplexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        iComplexVal1 = _mm_shuffle_epi32(iComplexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        iComplexVal2 = _mm_shufflelo_epi16(complexVal2, _MM_SHUFFLE(3, 1, 2, 0));

        iComplexVal2 = _mm_shufflehi_epi16(iComplexVal2, _MM_SHUFFLE(3, 1, 2, 0));

        iComplexVal2 = _mm_shuffle_epi32(iComplexVal2, _MM_SHUFFLE(2, 0, 3, 1));

        iOutputVal = _mm_or_si128(_mm_and_si128(iComplexVal1, lowMask),
                                  _mm_and_si128(iComplexVal2, highMask));

        _mm_store_si128((__m128i*)iBufferPtr, iOutputVal);

        qComplexVal1 = _mm_shufflelo_epi16(complexVal1, _MM_SHUFFLE(2, 0, 3, 1));

        qComplexVal1 = _mm_shufflehi_epi16(qComplexVal1, _MM_SHUFFLE(2, 0, 3, 1));

        qComplexVal1 = _mm_shuffle_epi32(qComplexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        qComplexVal2 = _mm_shufflelo_epi16(complexVal2, _MM_SHUFFLE(2, 0, 3, 1));

        qComplexVal2 = _mm_shufflehi_epi16(qComplexVal2, _MM_SHUFFLE(2, 0, 3, 1));

        qComplexVal2 = _mm_shuffle_epi32(qComplexVal2, _MM_SHUFFLE(2, 0, 3, 1));

        qOutputVal = _mm_or_si128(_mm_and_si128(qComplexVal1, lowMask),
                                  _mm_and_si128(qComplexVal2, highMask));

        _mm_store_si128((__m128i*)qBufferPtr, qOutputVal);

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_deinterleave_16i_x2_generic(int16_t* iBuffer,
                                                         int16_t* qBuffer,
                                                         const lv_16sc_t* complexVector,
                                                         unsigned int num_points)
{
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;
    unsigned int number;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_deinterleave_16i_x2_neon(int16_t* iBuffer,
                                                      int16_t* qBuffer,
                                                      const lv_16sc_t* complexVector,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    int16x8x2_t complexVal;

    for (; number < eighthPoints; number++) {
        complexVal = vld2q_s16(complexVectorPtr);
        vst1q_s16(iBufferPtr, complexVal.val[0]);
        vst1q_s16(qBufferPtr, complexVal.val[1]);
        complexVectorPtr += 16;
        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16ic_deinterleave_16i_x2_neonv8(int16_t* iBuffer,
                                                        int16_t* qBuffer,
                                                        const lv_16sc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    int16x8x2_t complexVal0, complexVal1;

    for (; number < sixteenthPoints; number++) {
        complexVal0 = vld2q_s16(complexVectorPtr);
        complexVal1 = vld2q_s16(complexVectorPtr + 16);
        __VOLK_PREFETCH(complexVectorPtr + 32);

        vst1q_s16(iBufferPtr, complexVal0.val[0]);
        vst1q_s16(iBufferPtr + 8, complexVal1.val[0]);
        vst1q_s16(qBufferPtr, complexVal0.val[1]);
        vst1q_s16(qBufferPtr + 8, complexVal1.val[1]);

        complexVectorPtr += 32;
        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_ORC

extern void volk_16ic_deinterleave_16i_x2_a_orc_impl(int16_t* iBuffer,
                                                     int16_t* qBuffer,
                                                     const lv_16sc_t* complexVector,
                                                     int num_points);
static inline void volk_16ic_deinterleave_16i_x2_u_orc(int16_t* iBuffer,
                                                       int16_t* qBuffer,
                                                       const lv_16sc_t* complexVector,
                                                       unsigned int num_points)
{
    volk_16ic_deinterleave_16i_x2_a_orc_impl(iBuffer, qBuffer, complexVector, num_points);
}
#endif /* LV_HAVE_ORC */

#endif /* INCLUDED_volk_16ic_deinterleave_16i_x2_a_H */


#ifndef INCLUDED_volk_16ic_deinterleave_16i_x2_u_H
#define INCLUDED_volk_16ic_deinterleave_16i_x2_u_H

#include <inttypes.h>
#include <stdio.h>
#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_u_avx2(int16_t* iBuffer,
                                                        int16_t* qBuffer,
                                                        const lv_16sc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    __m256i MoveMask = _mm256_set_epi8(15,
                                       14,
                                       11,
                                       10,
                                       7,
                                       6,
                                       3,
                                       2,
                                       13,
                                       12,
                                       9,
                                       8,
                                       5,
                                       4,
                                       1,
                                       0,
                                       15,
                                       14,
                                       11,
                                       10,
                                       7,
                                       6,
                                       3,
                                       2,
                                       13,
                                       12,
                                       9,
                                       8,
                                       5,
                                       4,
                                       1,
                                       0);

    __m256i iMove2, iMove1;
    __m256i complexVal1, complexVal2, iOutputVal, qOutputVal;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal1 = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal2 = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        iMove2 = _mm256_shuffle_epi8(complexVal2, MoveMask);
        iMove1 = _mm256_shuffle_epi8(complexVal1, MoveMask);

        iOutputVal = _mm256_permute2x128_si256(_mm256_permute4x64_epi64(iMove1, 0x08),
                                               _mm256_permute4x64_epi64(iMove2, 0x80),
                                               0x30);
        qOutputVal = _mm256_permute2x128_si256(_mm256_permute4x64_epi64(iMove1, 0x0d),
                                               _mm256_permute4x64_epi64(iMove2, 0xd0),
                                               0x30);

        _mm256_storeu_si256((__m256i*)iBufferPtr, iOutputVal);
        _mm256_storeu_si256((__m256i*)qBufferPtr, qOutputVal);

        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    int16_t* int16ComplexVectorPtr = (int16_t*)complexVectorPtr;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *int16ComplexVectorPtr++;
        *qBufferPtr++ = *int16ComplexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16ic_deinterleave_16i_x2_rvv(int16_t* iBuffer,
                                                     int16_t* qBuffer,
                                                     const lv_16sc_t* complexVector,
                                                     unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vuint32m8_t vc = __riscv_vle32_v_u32m8((const uint32_t*)complexVector, vl);
        vuint16m4_t vr = __riscv_vnsrl(vc, 0, vl);
        vuint16m4_t vi = __riscv_vnsrl(vc, 16, vl);
        __riscv_vse16((uint16_t*)iBuffer, vr, vl);
        __riscv_vse16((uint16_t*)qBuffer, vi, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_16ic_deinterleave_16i_x2_rvvseg(int16_t* iBuffer,
                                                        int16_t* qBuffer,
                                                        const lv_16sc_t* complexVector,
                                                        unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vuint16m4x2_t vc =
            __riscv_vlseg2e16_v_u16m4x2((const uint16_t*)complexVector, vl);
        vuint16m4_t vr = __riscv_vget_u16m4(vc, 0);
        vuint16m4_t vi = __riscv_vget_u16m4(vc, 1);
        __riscv_vse16((uint16_t*)iBuffer, vr, vl);
        __riscv_vse16((uint16_t*)qBuffer, vi, vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_16ic_deinterleave_16i_x2_u_H */
