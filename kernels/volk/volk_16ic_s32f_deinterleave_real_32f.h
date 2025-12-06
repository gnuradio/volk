/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_s32f_deinterleave_real_32f
 *
 * \b Overview
 *
 * Deinterleaves the complex 16 bit vector and returns just the real
 * part (inphase) of the data as a vector of floats that have been
 * scaled.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 *  void volk_16ic_s32f_deinterleave_real_32f(float* iBuffer, const lv_16sc_t*
 * complexVector, const float scalar, unsigned int num_points){ \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of 16-bit shorts.
 * \li scalar: The value to be divided against each sample of the input complex vector.
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The floating point I buffer output data.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_16ic_s32f_deinterleave_real_32f();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_s32f_deinterleave_real_32f_a_H
#define INCLUDED_volk_16ic_s32f_deinterleave_real_32f_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_16ic_s32f_deinterleave_real_32f_a_avx2(float* iBuffer,
                                            const lv_16sc_t* complexVector,
                                            const float scalar,
                                            unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    __m256 iFloatValue;

    const float iScalar = 1.0 / scalar;
    __m256 invScalar = _mm256_set1_ps(iScalar);
    __m256i complexVal, iIntVal;
    __m128i complexVal128;
    int8_t* complexVectorPtr = (int8_t*)complexVector;

    __m256i moveMask = _mm256_set_epi8(0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       13,
                                       12,
                                       9,
                                       8,
                                       5,
                                       4,
                                       1,
                                       0,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       13,
                                       12,
                                       9,
                                       8,
                                       5,
                                       4,
                                       1,
                                       0);

    for (; number < eighthPoints; number++) {
        complexVal = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal = _mm256_shuffle_epi8(complexVal, moveMask);
        complexVal = _mm256_permute4x64_epi64(complexVal, 0xd8);
        complexVal128 = _mm256_extracti128_si256(complexVal, 0);

        iIntVal = _mm256_cvtepi16_epi32(complexVal128);
        iFloatValue = _mm256_cvtepi32_ps(iIntVal);

        iFloatValue = _mm256_mul_ps(iFloatValue, invScalar);

        _mm256_store_ps(iBufferPtr, iFloatValue);

        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    int16_t* sixteenTComplexVectorPtr = (int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((float)(*sixteenTComplexVectorPtr++)) * iScalar;
        sixteenTComplexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_16ic_s32f_deinterleave_real_32f_a_sse4_1(float* iBuffer,
                                              const lv_16sc_t* complexVector,
                                              const float scalar,
                                              unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m128 iFloatValue;

    const float iScalar = 1.0 / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    __m128i complexVal, iIntVal;
    int8_t* complexVectorPtr = (int8_t*)complexVector;

    __m128i moveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 13, 12, 9, 8, 5, 4, 1, 0);

    for (; number < quarterPoints; number++) {
        complexVal = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal = _mm_shuffle_epi8(complexVal, moveMask);

        iIntVal = _mm_cvtepi16_epi32(complexVal);
        iFloatValue = _mm_cvtepi32_ps(iIntVal);

        iFloatValue = _mm_mul_ps(iFloatValue, invScalar);

        _mm_store_ps(iBufferPtr, iFloatValue);

        iBufferPtr += 4;
    }

    number = quarterPoints * 4;
    int16_t* sixteenTComplexVectorPtr = (int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((float)(*sixteenTComplexVectorPtr++)) * iScalar;
        sixteenTComplexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_16ic_s32f_deinterleave_real_32f_a_sse(float* iBuffer,
                                           const lv_16sc_t* complexVector,
                                           const float scalar,
                                           unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;
    __m128 iValue;

    const float iScalar = 1.0 / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    int16_t* complexVectorPtr = (int16_t*)complexVector;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

    for (; number < quarterPoints; number++) {
        floatBuffer[0] = (float)(*complexVectorPtr);
        complexVectorPtr += 2;
        floatBuffer[1] = (float)(*complexVectorPtr);
        complexVectorPtr += 2;
        floatBuffer[2] = (float)(*complexVectorPtr);
        complexVectorPtr += 2;
        floatBuffer[3] = (float)(*complexVectorPtr);
        complexVectorPtr += 2;

        iValue = _mm_load_ps(floatBuffer);

        iValue = _mm_mul_ps(iValue, invScalar);

        _mm_store_ps(iBufferPtr, iValue);

        iBufferPtr += 4;
    }

    number = quarterPoints * 4;
    complexVectorPtr = (int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((float)(*complexVectorPtr++)) * iScalar;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC
static inline void
volk_16ic_s32f_deinterleave_real_32f_generic(float* iBuffer,
                                             const lv_16sc_t* complexVector,
                                             const float scalar,
                                             unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* iBufferPtr = iBuffer;
    const float invScalar = 1.0 / scalar;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = ((float)(*complexVectorPtr++)) * invScalar;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_16ic_s32f_deinterleave_real_32f_a_H */

#ifndef INCLUDED_volk_16ic_s32f_deinterleave_real_32f_u_H
#define INCLUDED_volk_16ic_s32f_deinterleave_real_32f_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_16ic_s32f_deinterleave_real_32f_u_avx2(float* iBuffer,
                                            const lv_16sc_t* complexVector,
                                            const float scalar,
                                            unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    __m256 iFloatValue;

    const float iScalar = 1.0 / scalar;
    __m256 invScalar = _mm256_set1_ps(iScalar);
    __m256i complexVal, iIntVal;
    __m128i complexVal128;
    int8_t* complexVectorPtr = (int8_t*)complexVector;

    __m256i moveMask = _mm256_set_epi8(0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       13,
                                       12,
                                       9,
                                       8,
                                       5,
                                       4,
                                       1,
                                       0,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       13,
                                       12,
                                       9,
                                       8,
                                       5,
                                       4,
                                       1,
                                       0);

    for (; number < eighthPoints; number++) {
        complexVal = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal = _mm256_shuffle_epi8(complexVal, moveMask);
        complexVal = _mm256_permute4x64_epi64(complexVal, 0xd8);
        complexVal128 = _mm256_extracti128_si256(complexVal, 0);

        iIntVal = _mm256_cvtepi16_epi32(complexVal128);
        iFloatValue = _mm256_cvtepi32_ps(iIntVal);

        iFloatValue = _mm256_mul_ps(iFloatValue, invScalar);

        _mm256_storeu_ps(iBufferPtr, iFloatValue);

        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    int16_t* sixteenTComplexVectorPtr = (int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((float)(*sixteenTComplexVectorPtr++)) * iScalar;
        sixteenTComplexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_16ic_s32f_deinterleave_real_32f_neon(float* iBuffer,
                                          const lv_16sc_t* complexVector,
                                          const float scalar,
                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    float* iBufferPtr = iBuffer;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    const float invScalar = 1.0f / scalar;
    float32x4_t vInvScalar = vdupq_n_f32(invScalar);

    for (; number < quarter_points; number++) {
        int16x4x2_t input = vld2_s16(complexVectorPtr);
        complexVectorPtr += 8;

        int32x4_t iInt = vmovl_s16(input.val[0]);
        float32x4_t iFloat = vcvtq_f32_s32(iInt);
        iFloat = vmulq_f32(iFloat, vInvScalar);

        vst1q_f32(iBufferPtr, iFloat);
        iBufferPtr += 4;
    }

    number = quarter_points * 4;
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((float)(*complexVectorPtr++)) * invScalar;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_16ic_s32f_deinterleave_real_32f_rvv(float* iBuffer,
                                         const lv_16sc_t* complexVector,
                                         const float scalar,
                                         unsigned int num_points)
{
    const int32_t* in = (const int32_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in += vl, iBuffer += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t vc = __riscv_vle32_v_i32m8(in, vl);
        vfloat32m8_t vr = __riscv_vfwcvt_f(__riscv_vncvt_x(vc, vl), vl);
        __riscv_vse32(iBuffer, __riscv_vfmul(vr, 1.0f / scalar, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_16ic_s32f_deinterleave_real_32f_u_H */
