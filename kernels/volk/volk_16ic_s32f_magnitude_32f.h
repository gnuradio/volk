/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_s32f_magnitude_32f
 *
 * \b Overview
 *
 * Computes the magnitude of the complexVector and stores the results
 * in the magnitudeVector as a scaled floating point number.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_s32f_magnitude_32f(float* magnitudeVector, const lv_16sc_t*
 * complexVector, const float scalar, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of complex 16-bit shorts.
 * \li scalar: The value to be divided against each sample of the input complex vector.
 * \li num_points: The number of samples.
 *
 * \b Outputs
 * \li magnitudeVector: The magnitude of the complex values.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_16ic_s32f_magnitude_32f();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_s32f_magnitude_32f_a_H
#define INCLUDED_volk_16ic_s32f_magnitude_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_a_avx2(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m256 invScalar = _mm256_set1_ps(1.0 / scalar);

    __m256 cplxValue1, cplxValue2, result;
    __m256i int1, int2;
    __m128i short1, short2;
    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    for (; number < eighthPoints; number++) {

        int1 = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 16;
        short1 = _mm256_extracti128_si256(int1, 0);
        short2 = _mm256_extracti128_si256(int1, 1);

        int1 = _mm256_cvtepi16_epi32(short1);
        int2 = _mm256_cvtepi16_epi32(short2);
        cplxValue1 = _mm256_cvtepi32_ps(int1);
        cplxValue2 = _mm256_cvtepi32_ps(int2);

        cplxValue1 = _mm256_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm256_mul_ps(cplxValue2, invScalar);

        cplxValue1 = _mm256_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm256_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm256_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values
        result = _mm256_permutevar8x32_ps(result, idx);

        result = _mm256_sqrt_ps(result); // Square root the values

        _mm256_store_ps(magnitudeVectorPtr, result);

        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float val1Real = (float)(*complexVectorPtr++) / scalar;
        float val1Imag = (float)(*complexVectorPtr++) / scalar;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_a_sse3(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m128 invScalar = _mm_set_ps1(1.0 / scalar);

    __m128 cplxValue1, cplxValue2, result;

    __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[8];

    for (; number < quarterPoints; number++) {

        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        inputFloatBuffer[4] = (float)(complexVectorPtr[4]);
        inputFloatBuffer[5] = (float)(complexVectorPtr[5]);
        inputFloatBuffer[6] = (float)(complexVectorPtr[6]);
        inputFloatBuffer[7] = (float)(complexVectorPtr[7]);

        cplxValue1 = _mm_load_ps(&inputFloatBuffer[0]);
        cplxValue2 = _mm_load_ps(&inputFloatBuffer[4]);

        complexVectorPtr += 8;

        cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

        cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result); // Square root the values

        _mm_store_ps(magnitudeVectorPtr, result);

        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float val1Real = (float)(*complexVectorPtr++) / scalar;
        float val1Imag = (float)(*complexVectorPtr++) / scalar;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_a_sse(float* magnitudeVector,
                                                      const lv_16sc_t* complexVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float iScalar = 1.0 / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);

    __m128 cplxValue1, cplxValue2, result, re, im;

    __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[8];

    for (; number < quarterPoints; number++) {
        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        inputFloatBuffer[4] = (float)(complexVectorPtr[4]);
        inputFloatBuffer[5] = (float)(complexVectorPtr[5]);
        inputFloatBuffer[6] = (float)(complexVectorPtr[6]);
        inputFloatBuffer[7] = (float)(complexVectorPtr[7]);

        cplxValue1 = _mm_load_ps(&inputFloatBuffer[0]);
        cplxValue2 = _mm_load_ps(&inputFloatBuffer[4]);

        re = _mm_shuffle_ps(cplxValue1, cplxValue2, 0x88);
        im = _mm_shuffle_ps(cplxValue1, cplxValue2, 0xdd);

        complexVectorPtr += 8;

        cplxValue1 = _mm_mul_ps(re, invScalar);
        cplxValue2 = _mm_mul_ps(im, invScalar);

        cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm_add_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result); // Square root the values

        _mm_store_ps(magnitudeVectorPtr, result);

        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float val1Real = (float)(*complexVectorPtr++) * iScalar;
        float val1Imag = (float)(*complexVectorPtr++) * iScalar;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}


#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_s32f_magnitude_32f_generic(float* magnitudeVector,
                                                        const lv_16sc_t* complexVector,
                                                        const float scalar,
                                                        unsigned int num_points)
{
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;
    unsigned int number = 0;
    const float invScalar = 1.0 / scalar;
    for (number = 0; number < num_points; number++) {
        float real = ((float)(*complexVectorPtr++)) * invScalar;
        float imag = ((float)(*complexVectorPtr++)) * invScalar;
        *magnitudeVectorPtr++ = sqrtf((real * real) + (imag * imag));
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_16ic_s32f_magnitude_32f_a_H */

#ifndef INCLUDED_volk_16ic_s32f_magnitude_32f_u_H
#define INCLUDED_volk_16ic_s32f_magnitude_32f_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_u_avx2(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m256 invScalar = _mm256_set1_ps(1.0 / scalar);

    __m256 cplxValue1, cplxValue2, result;
    __m256i int1, int2;
    __m128i short1, short2;
    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    for (; number < eighthPoints; number++) {

        int1 = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 16;
        short1 = _mm256_extracti128_si256(int1, 0);
        short2 = _mm256_extracti128_si256(int1, 1);

        int1 = _mm256_cvtepi16_epi32(short1);
        int2 = _mm256_cvtepi16_epi32(short2);
        cplxValue1 = _mm256_cvtepi32_ps(int1);
        cplxValue2 = _mm256_cvtepi32_ps(int2);

        cplxValue1 = _mm256_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm256_mul_ps(cplxValue2, invScalar);

        cplxValue1 = _mm256_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm256_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm256_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values
        result = _mm256_permutevar8x32_ps(result, idx);

        result = _mm256_sqrt_ps(result); // Square root the values

        _mm256_storeu_ps(magnitudeVectorPtr, result);

        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float val1Real = (float)(*complexVectorPtr++) / scalar;
        float val1Imag = (float)(*complexVectorPtr++) / scalar;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_s32f_magnitude_32f_neon(float* magnitudeVector,
                                                     const lv_16sc_t* complexVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;
    const float invScalar = 1.0f / scalar;
    float32x4_t vInvScalar = vdupq_n_f32(invScalar);

    for (; number < quarter_points; number++) {
        int16x4x2_t input = vld2_s16(complexVectorPtr);
        complexVectorPtr += 8;

        int32x4_t realInt = vmovl_s16(input.val[0]);
        int32x4_t imagInt = vmovl_s16(input.val[1]);

        float32x4_t realFloat = vcvtq_f32_s32(realInt);
        float32x4_t imagFloat = vcvtq_f32_s32(imagInt);

        realFloat = vmulq_f32(realFloat, vInvScalar);
        imagFloat = vmulq_f32(imagFloat, vInvScalar);

        float32x4_t realSquared = vmulq_f32(realFloat, realFloat);
        float32x4_t imagSquared = vmulq_f32(imagFloat, imagFloat);
        float32x4_t sumSquared = vaddq_f32(realSquared, imagSquared);

        /* Use reciprocal square root estimate with Newton-Raphson refinement */
        float32x4_t rsqrt = vrsqrteq_f32(sumSquared);
        rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(sumSquared, rsqrt), rsqrt));
        rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(sumSquared, rsqrt), rsqrt));
        float32x4_t result = vmulq_f32(sumSquared, rsqrt);

        /* Handle zero case - if sumSquared is 0, result should be 0 */
        uint32x4_t zero_mask = vceqq_f32(sumSquared, vdupq_n_f32(0.0f));
        result = vbslq_f32(zero_mask, sumSquared, result);

        vst1q_f32(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarter_points * 4;
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float real = ((float)(*complexVectorPtr++)) * invScalar;
        float imag = ((float)(*complexVectorPtr++)) * invScalar;
        *magnitudeVectorPtr++ = sqrtf((real * real) + (imag * imag));
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16ic_s32f_magnitude_32f_neonv8(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;
    const float invScalar = 1.0f / scalar;
    float32x4_t vInvScalar = vdupq_n_f32(invScalar);

    for (; number < eighth_points; number++) {
        int16x8x2_t input = vld2q_s16(complexVectorPtr);
        complexVectorPtr += 16;
        __VOLK_PREFETCH(complexVectorPtr + 16);

        /* First 4 elements */
        int32x4_t realInt0 = vmovl_s16(vget_low_s16(input.val[0]));
        int32x4_t imagInt0 = vmovl_s16(vget_low_s16(input.val[1]));

        float32x4_t realFloat0 = vcvtq_f32_s32(realInt0);
        float32x4_t imagFloat0 = vcvtq_f32_s32(imagInt0);

        realFloat0 = vmulq_f32(realFloat0, vInvScalar);
        imagFloat0 = vmulq_f32(imagFloat0, vInvScalar);

        float32x4_t sumSquared0 =
            vfmaq_f32(vmulq_f32(imagFloat0, imagFloat0), realFloat0, realFloat0);
        float32x4_t result0 = vsqrtq_f32(sumSquared0);

        /* Second 4 elements */
        int32x4_t realInt1 = vmovl_s16(vget_high_s16(input.val[0]));
        int32x4_t imagInt1 = vmovl_s16(vget_high_s16(input.val[1]));

        float32x4_t realFloat1 = vcvtq_f32_s32(realInt1);
        float32x4_t imagFloat1 = vcvtq_f32_s32(imagInt1);

        realFloat1 = vmulq_f32(realFloat1, vInvScalar);
        imagFloat1 = vmulq_f32(imagFloat1, vInvScalar);

        float32x4_t sumSquared1 =
            vfmaq_f32(vmulq_f32(imagFloat1, imagFloat1), realFloat1, realFloat1);
        float32x4_t result1 = vsqrtq_f32(sumSquared1);

        vst1q_f32(magnitudeVectorPtr, result0);
        vst1q_f32(magnitudeVectorPtr + 4, result1);
        magnitudeVectorPtr += 8;
    }

    number = eighth_points * 8;
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float real = ((float)(*complexVectorPtr++)) * invScalar;
        float imag = ((float)(*complexVectorPtr++)) * invScalar;
        *magnitudeVectorPtr++ = sqrtf((real * real) + (imag * imag));
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16ic_s32f_magnitude_32f_rvv(float* magnitudeVector,
                                                    const lv_16sc_t* complexVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint32m8_t vc = __riscv_vle32_v_i32m8((const int32_t*)complexVector, vl);
        vint16m4_t vr = __riscv_vnsra(vc, 0, vl);
        vint16m4_t vi = __riscv_vnsra(vc, 16, vl);
        vfloat32m8_t vrf = __riscv_vfmul(__riscv_vfwcvt_f(vr, vl), 1.0f / scalar, vl);
        vfloat32m8_t vif = __riscv_vfmul(__riscv_vfwcvt_f(vi, vl), 1.0f / scalar, vl);
        vfloat32m8_t vf = __riscv_vfmacc(__riscv_vfmul(vif, vif, vl), vrf, vrf, vl);
        __riscv_vse32(magnitudeVector, __riscv_vfsqrt(vf, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_16ic_s32f_magnitude_32f_rvvseg(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint16m4x2_t vc = __riscv_vlseg2e16_v_i16m4x2((const int16_t*)complexVector, vl);
        vint16m4_t vr = __riscv_vget_i16m4(vc, 0);
        vint16m4_t vi = __riscv_vget_i16m4(vc, 1);
        vfloat32m8_t vrf = __riscv_vfmul(__riscv_vfwcvt_f(vr, vl), 1.0f / scalar, vl);
        vfloat32m8_t vif = __riscv_vfmul(__riscv_vfwcvt_f(vi, vl), 1.0f / scalar, vl);
        vfloat32m8_t vf = __riscv_vfmacc(__riscv_vfmul(vif, vif, vl), vrf, vrf, vl);
        __riscv_vse32(magnitudeVector, __riscv_vfsqrt(vf, vl), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_16ic_s32f_magnitude_32f_u_H */
