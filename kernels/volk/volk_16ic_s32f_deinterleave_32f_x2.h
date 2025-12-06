/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_s32f_deinterleave_32f_x2
 *
 * \b Overview
 *
 * Deinterleaves the complex 16 bit vector into I & Q vector data and
 * returns the result as two vectors of floats that have been scaled.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 *  void volk_16ic_s32f_deinterleave_32f_x2(float* iBuffer, float* qBuffer, const
 * lv_16sc_t* complexVector, const float scalar, unsigned int num_points){ \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of 16-bit shorts.
 * \li scalar: The value to be divided against each sample of the input complex vector.
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The floating point I buffer output data.
 * \li qBuffer: The floating point Q buffer output data.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_16ic_s32f_deinterleave_32f_x2();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_s32f_deinterleave_32f_x2_a_H
#define INCLUDED_volk_16ic_s32f_deinterleave_32f_x2_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_16ic_s32f_deinterleave_32f_x2_a_avx2(float* iBuffer,
                                          float* qBuffer,
                                          const lv_16sc_t* complexVector,
                                          const float scalar,
                                          unsigned int num_points)
{
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    uint64_t number = 0;
    const uint64_t eighthPoints = num_points / 8;
    __m256 cplxValue1, cplxValue2, iValue, qValue;
    __m256i cplxValueA, cplxValueB;
    __m128i cplxValue128;

    __m256 invScalar = _mm256_set1_ps(1.0 / scalar);
    int16_t* complexVectorPtr = (int16_t*)complexVector;
    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    for (; number < eighthPoints; number++) {

        cplxValueA = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 16;

        // cvt
        cplxValue128 = _mm256_extracti128_si256(cplxValueA, 0);
        cplxValueB = _mm256_cvtepi16_epi32(cplxValue128);
        cplxValue1 = _mm256_cvtepi32_ps(cplxValueB);
        cplxValue128 = _mm256_extracti128_si256(cplxValueA, 1);
        cplxValueB = _mm256_cvtepi16_epi32(cplxValue128);
        cplxValue2 = _mm256_cvtepi32_ps(cplxValueB);

        cplxValue1 = _mm256_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm256_mul_ps(cplxValue2, invScalar);

        // Arrange in i1i2i3i4 format
        iValue = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        iValue = _mm256_permutevar8x32_ps(iValue, idx);
        // Arrange in q1q2q3q4 format
        qValue = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));
        qValue = _mm256_permutevar8x32_ps(qValue, idx);

        _mm256_store_ps(iBufferPtr, iValue);
        _mm256_store_ps(qBufferPtr, qValue);

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    complexVectorPtr = (int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
        *qBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_16ic_s32f_deinterleave_32f_x2_a_sse(float* iBuffer,
                                         float* qBuffer,
                                         const lv_16sc_t* complexVector,
                                         const float scalar,
                                         unsigned int num_points)
{
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    uint64_t number = 0;
    const uint64_t quarterPoints = num_points / 4;
    __m128 cplxValue1, cplxValue2, iValue, qValue;

    __m128 invScalar = _mm_set_ps1(1.0 / scalar);
    int16_t* complexVectorPtr = (int16_t*)complexVector;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[8];

    for (; number < quarterPoints; number++) {

        floatBuffer[0] = (float)(complexVectorPtr[0]);
        floatBuffer[1] = (float)(complexVectorPtr[1]);
        floatBuffer[2] = (float)(complexVectorPtr[2]);
        floatBuffer[3] = (float)(complexVectorPtr[3]);

        floatBuffer[4] = (float)(complexVectorPtr[4]);
        floatBuffer[5] = (float)(complexVectorPtr[5]);
        floatBuffer[6] = (float)(complexVectorPtr[6]);
        floatBuffer[7] = (float)(complexVectorPtr[7]);

        cplxValue1 = _mm_load_ps(&floatBuffer[0]);
        cplxValue2 = _mm_load_ps(&floatBuffer[4]);

        complexVectorPtr += 8;

        cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

        // Arrange in i1i2i3i4 format
        iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        // Arrange in q1q2q3q4 format
        qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        _mm_store_ps(iBufferPtr, iValue);
        _mm_store_ps(qBufferPtr, qValue);

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    complexVectorPtr = (int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
        *qBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void
volk_16ic_s32f_deinterleave_32f_x2_generic(float* iBuffer,
                                           float* qBuffer,
                                           const lv_16sc_t* complexVector,
                                           const float scalar,
                                           unsigned int num_points)
{
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;
    unsigned int number;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
        *qBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
static inline void volk_16ic_s32f_deinterleave_32f_x2_neon(float* iBuffer,
                                                           float* qBuffer,
                                                           const lv_16sc_t* complexVector,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;
    unsigned int eighth_points = num_points / 4;
    unsigned int number;
    float iScalar = 1.f / scalar;
    float32x4_t invScalar;
    invScalar = vld1q_dup_f32(&iScalar);

    int16x4x2_t complexInput_s16;
    int32x4x2_t complexInput_s32;
    float32x4x2_t complexFloat;

    for (number = 0; number < eighth_points; number++) {
        complexInput_s16 = vld2_s16(complexVectorPtr);
        complexInput_s32.val[0] = vmovl_s16(complexInput_s16.val[0]);
        complexInput_s32.val[1] = vmovl_s16(complexInput_s16.val[1]);
        complexFloat.val[0] = vcvtq_f32_s32(complexInput_s32.val[0]);
        complexFloat.val[1] = vcvtq_f32_s32(complexInput_s32.val[1]);
        complexFloat.val[0] = vmulq_f32(complexFloat.val[0], invScalar);
        complexFloat.val[1] = vmulq_f32(complexFloat.val[1], invScalar);
        vst1q_f32(iBufferPtr, complexFloat.val[0]);
        vst1q_f32(qBufferPtr, complexFloat.val[1]);
        complexVectorPtr += 8;
        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    for (number = eighth_points * 4; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
        *qBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_16ic_s32f_deinterleave_32f_x2_neonv8(float* iBuffer,
                                          float* qBuffer,
                                          const lv_16sc_t* complexVector,
                                          const float scalar,
                                          unsigned int num_points)
{
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;
    const unsigned int eighthPoints = num_points / 8;
    const float iScalar = 1.f / scalar;
    const float32x4_t invScalar = vdupq_n_f32(iScalar);

    for (unsigned int number = 0; number < eighthPoints; number++) {
        int16x8x2_t cplx0 = vld2q_s16(complexVectorPtr);
        __VOLK_PREFETCH(complexVectorPtr + 32);

        /* Convert lower 4 of each to float */
        int32x4_t i_lo = vmovl_s16(vget_low_s16(cplx0.val[0]));
        int32x4_t q_lo = vmovl_s16(vget_low_s16(cplx0.val[1]));
        int32x4_t i_hi = vmovl_s16(vget_high_s16(cplx0.val[0]));
        int32x4_t q_hi = vmovl_s16(vget_high_s16(cplx0.val[1]));

        float32x4_t iFloat_lo = vmulq_f32(vcvtq_f32_s32(i_lo), invScalar);
        float32x4_t qFloat_lo = vmulq_f32(vcvtq_f32_s32(q_lo), invScalar);
        float32x4_t iFloat_hi = vmulq_f32(vcvtq_f32_s32(i_hi), invScalar);
        float32x4_t qFloat_hi = vmulq_f32(vcvtq_f32_s32(q_hi), invScalar);

        vst1q_f32(iBufferPtr, iFloat_lo);
        vst1q_f32(iBufferPtr + 4, iFloat_hi);
        vst1q_f32(qBufferPtr, qFloat_lo);
        vst1q_f32(qBufferPtr + 4, qFloat_hi);

        complexVectorPtr += 16;
        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
        *qBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_ORC
extern void volk_16ic_s32f_deinterleave_32f_x2_a_orc_impl(float* iBuffer,
                                                          float* qBuffer,
                                                          const lv_16sc_t* complexVector,
                                                          const float scalar,
                                                          int num_points);

static inline void
volk_16ic_s32f_deinterleave_32f_x2_u_orc(float* iBuffer,
                                         float* qBuffer,
                                         const lv_16sc_t* complexVector,
                                         const float scalar,
                                         unsigned int num_points)
{
    volk_16ic_s32f_deinterleave_32f_x2_a_orc_impl(
        iBuffer, qBuffer, complexVector, scalar, num_points);
}
#endif /* LV_HAVE_ORC */


#endif /* INCLUDED_volk_16ic_s32f_deinterleave_32f_x2_a_H */


#ifndef INCLUDED_volk_16ic_s32f_deinterleave_32f_x2_u_H
#define INCLUDED_volk_16ic_s32f_deinterleave_32f_x2_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_16ic_s32f_deinterleave_32f_x2_u_avx2(float* iBuffer,
                                          float* qBuffer,
                                          const lv_16sc_t* complexVector,
                                          const float scalar,
                                          unsigned int num_points)
{
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    uint64_t number = 0;
    const uint64_t eighthPoints = num_points / 8;
    __m256 cplxValue1, cplxValue2, iValue, qValue;
    __m256i cplxValueA, cplxValueB;
    __m128i cplxValue128;

    __m256 invScalar = _mm256_set1_ps(1.0 / scalar);
    int16_t* complexVectorPtr = (int16_t*)complexVector;
    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    for (; number < eighthPoints; number++) {

        cplxValueA = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 16;

        // cvt
        cplxValue128 = _mm256_extracti128_si256(cplxValueA, 0);
        cplxValueB = _mm256_cvtepi16_epi32(cplxValue128);
        cplxValue1 = _mm256_cvtepi32_ps(cplxValueB);
        cplxValue128 = _mm256_extracti128_si256(cplxValueA, 1);
        cplxValueB = _mm256_cvtepi16_epi32(cplxValue128);
        cplxValue2 = _mm256_cvtepi32_ps(cplxValueB);

        cplxValue1 = _mm256_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm256_mul_ps(cplxValue2, invScalar);

        // Arrange in i1i2i3i4 format
        iValue = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        iValue = _mm256_permutevar8x32_ps(iValue, idx);
        // Arrange in q1q2q3q4 format
        qValue = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));
        qValue = _mm256_permutevar8x32_ps(qValue, idx);

        _mm256_storeu_ps(iBufferPtr, iValue);
        _mm256_storeu_ps(qBufferPtr, qValue);

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    complexVectorPtr = (int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
        *qBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16ic_s32f_deinterleave_32f_x2_rvv(float* iBuffer,
                                                          float* qBuffer,
                                                          const lv_16sc_t* complexVector,
                                                          const float scalar,
                                                          unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint32m8_t vc = __riscv_vle32_v_i32m8((const int32_t*)complexVector, vl);
        vint16m4_t vr = __riscv_vnsra(vc, 0, vl);
        vint16m4_t vi = __riscv_vnsra(vc, 16, vl);
        vfloat32m8_t vrf = __riscv_vfwcvt_f(vr, vl);
        vfloat32m8_t vif = __riscv_vfwcvt_f(vi, vl);
        __riscv_vse32(iBuffer, __riscv_vfmul(vrf, 1.0f / scalar, vl), vl);
        __riscv_vse32(qBuffer, __riscv_vfmul(vif, 1.0f / scalar, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void
volk_16ic_s32f_deinterleave_32f_x2_rvvseg(float* iBuffer,
                                          float* qBuffer,
                                          const lv_16sc_t* complexVector,
                                          const float scalar,
                                          unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint16m4x2_t vc = __riscv_vlseg2e16_v_i16m4x2((const int16_t*)complexVector, vl);
        vint16m4_t vr = __riscv_vget_i16m4(vc, 0);
        vint16m4_t vi = __riscv_vget_i16m4(vc, 1);
        vfloat32m8_t vrf = __riscv_vfwcvt_f(vr, vl);
        vfloat32m8_t vif = __riscv_vfwcvt_f(vi, vl);
        __riscv_vse32(iBuffer, __riscv_vfmul(vrf, 1.0f / scalar, vl), vl);
        __riscv_vse32(qBuffer, __riscv_vfmul(vif, 1.0f / scalar, vl), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_16ic_s32f_deinterleave_32f_x2_u_H */
