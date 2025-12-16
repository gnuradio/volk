/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_magnitude_16i
 *
 * \b Overview
 *
 * Computes the magnitude of the complexVector and stores the results
 * in the magnitudeVector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_magnitude_16i(int16_t* magnitudeVector, const lv_16sc_t* complexVector,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li num_points: The number of samples.
 *
 * \b Outputs
 * \li magnitudeVector: The magnitude of the complex values.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_16ic_magnitude_16i();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_magnitude_16i_a_H
#define INCLUDED_volk_16ic_magnitude_16i_a_H

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_magnitude_16i_a_avx2(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m256 vScalar = _mm256_set1_ps(SHRT_MAX);
    __m256 invScalar = _mm256_set1_ps(1.0f / SHRT_MAX);
    __m256i int1, int2;
    __m128i short1, short2;
    __m256 cplxValue1, cplxValue2, result;
    __m256i idx = _mm256_set_epi32(0, 0, 0, 0, 5, 1, 4, 0);

    for (; number < eighthPoints; number++) {

        int1 = _mm256_load_si256((__m256i*)complexVectorPtr);
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

        result = _mm256_sqrt_ps(result); // Square root the values

        result = _mm256_mul_ps(result, vScalar); // Scale the results

        int1 = _mm256_cvtps_epi32(result);
        int1 = _mm256_packs_epi32(int1, int1);
        int1 = _mm256_permutevar8x32_epi32(
            int1, idx); // permute to compensate for shuffling in hadd and packs
        short1 = _mm256_extracti128_si256(int1, 0);
        _mm_store_si128((__m128i*)magnitudeVectorPtr, short1);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        const float val1Real = (float)(*complexVectorPtr++) / SHRT_MAX;
        const float val1Imag = (float)(*complexVectorPtr++) / SHRT_MAX;
        const float val1Result =
            sqrtf((val1Real * val1Real) + (val1Imag * val1Imag)) * SHRT_MAX;
        *magnitudeVectorPtr++ = (int16_t)rintf(val1Result);
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_16ic_magnitude_16i_a_sse3(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m128 vScalar = _mm_set_ps1(SHRT_MAX);
    __m128 invScalar = _mm_set_ps1(1.0f / SHRT_MAX);

    __m128 cplxValue1, cplxValue2, result;

    __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[8];
    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

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

        result = _mm_mul_ps(result, vScalar); // Scale the results

        _mm_store_ps(outputFloatBuffer, result);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[0]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[1]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[2]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[3]);
    }

    number = quarterPoints * 4;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        const float val1Real = (float)(*complexVectorPtr++) / SHRT_MAX;
        const float val1Imag = (float)(*complexVectorPtr++) / SHRT_MAX;
        const float val1Result =
            sqrtf((val1Real * val1Real) + (val1Imag * val1Imag)) * SHRT_MAX;
        *magnitudeVectorPtr++ = (int16_t)rintf(val1Result);
    }
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_16ic_magnitude_16i_a_sse(int16_t* magnitudeVector,
                                                 const lv_16sc_t* complexVector,
                                                 unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m128 vScalar = _mm_set_ps1(SHRT_MAX);
    __m128 invScalar = _mm_set_ps1(1.0f / SHRT_MAX);

    __m128 cplxValue1, cplxValue2, iValue, qValue, result;

    __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[4];
    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (; number < quarterPoints; number++) {

        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        cplxValue1 = _mm_load_ps(inputFloatBuffer);
        complexVectorPtr += 4;

        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        cplxValue2 = _mm_load_ps(inputFloatBuffer);
        complexVectorPtr += 4;

        cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

        // Arrange in i1i2i3i4 format
        iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        // Arrange in q1q2q3q4 format
        qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        iValue = _mm_mul_ps(iValue, iValue); // Square the I values
        qValue = _mm_mul_ps(qValue, qValue); // Square the Q Values

        result = _mm_add_ps(iValue, qValue); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result); // Square root the values

        result = _mm_mul_ps(result, vScalar); // Scale the results

        _mm_store_ps(outputFloatBuffer, result);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[0]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[1]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[2]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[3]);
    }

    number = quarterPoints * 4;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        const float val1Real = (float)(*complexVectorPtr++) / SHRT_MAX;
        const float val1Imag = (float)(*complexVectorPtr++) / SHRT_MAX;
        const float val1Result =
            sqrtf((val1Real * val1Real) + (val1Imag * val1Imag)) * SHRT_MAX;
        *magnitudeVectorPtr++ = (int16_t)rintf(val1Result);
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_magnitude_16i_generic(int16_t* magnitudeVector,
                                                   const lv_16sc_t* complexVector,
                                                   unsigned int num_points)
{
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;
    unsigned int number = 0;
    const float scalar = SHRT_MAX;
    for (number = 0; number < num_points; number++) {
        float real = ((float)(*complexVectorPtr++)) / scalar;
        float imag = ((float)(*complexVectorPtr++)) / scalar;
        *magnitudeVectorPtr++ =
            (int16_t)rintf(sqrtf((real * real) + (imag * imag)) * scalar);
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_16ic_magnitude_16i_a_H */


#ifndef INCLUDED_volk_16ic_magnitude_16i_u_H
#define INCLUDED_volk_16ic_magnitude_16i_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_magnitude_16i_u_avx2(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m256 vScalar = _mm256_set1_ps(SHRT_MAX);
    __m256 invScalar = _mm256_set1_ps(1.0f / SHRT_MAX);
    __m256i int1, int2;
    __m128i short1, short2;
    __m256 cplxValue1, cplxValue2, result;
    __m256i idx = _mm256_set_epi32(0, 0, 0, 0, 5, 1, 4, 0);

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

        result = _mm256_sqrt_ps(result); // Square root the values

        result = _mm256_mul_ps(result, vScalar); // Scale the results

        int1 = _mm256_cvtps_epi32(result);
        int1 = _mm256_packs_epi32(int1, int1);
        int1 = _mm256_permutevar8x32_epi32(
            int1, idx); // permute to compensate for shuffling in hadd and packs
        short1 = _mm256_extracti128_si256(int1, 0);
        _mm_storeu_si128((__m128i*)magnitudeVectorPtr, short1);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        const float val1Real = (float)(*complexVectorPtr++) / SHRT_MAX;
        const float val1Imag = (float)(*complexVectorPtr++) / SHRT_MAX;
        const float val1Result =
            sqrtf((val1Real * val1Real) + (val1Imag * val1Imag)) * SHRT_MAX;
        *magnitudeVectorPtr++ = (int16_t)rintf(val1Result);
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_NEONV7
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void volk_16ic_magnitude_16i_neonv7(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;

    const float scalar = SHRT_MAX;
    const float inv_scalar = 1.0f / scalar;

    int16_t* magnitudeVectorPtr = magnitudeVector;
    const lv_16sc_t* complexVectorPtr = complexVector;

    float32x4_t mag_vec;
    float32x4x2_t c_vec;

    for (number = 0; number < quarter_points; number++) {
        const int16x4x2_t c16_vec = vld2_s16((int16_t*)complexVectorPtr);
        __VOLK_PREFETCH(complexVectorPtr + 4);
        c_vec.val[0] = vcvtq_f32_s32(vmovl_s16(c16_vec.val[0]));
        c_vec.val[1] = vcvtq_f32_s32(vmovl_s16(c16_vec.val[1]));
        // Scale to close to 0-1
        c_vec.val[0] = vmulq_n_f32(c_vec.val[0], inv_scalar);
        c_vec.val[1] = vmulq_n_f32(c_vec.val[1], inv_scalar);
        // vsqrtq_f32 is armv8
        const float32x4_t mag_vec_squared = _vmagnitudesquaredq_f32(c_vec);
        mag_vec = vmulq_f32(mag_vec_squared, _vinvsqrtq_f32(mag_vec_squared));
        // Reconstruct
        mag_vec = vmulq_n_f32(mag_vec, scalar);
        // Add 0.5 for correct rounding because vcvtq_s32_f32 truncates.
        // This works because the magnitude is always positive.
        mag_vec = vaddq_f32(mag_vec, vdupq_n_f32(0.5));
        const int16x4_t mag16_vec = vmovn_s32(vcvtq_s32_f32(mag_vec));
        vst1_s16(magnitudeVectorPtr, mag16_vec);
        // Advance pointers
        magnitudeVectorPtr += 4;
        complexVectorPtr += 4;
    }

    // Deal with the rest
    for (number = quarter_points * 4; number < num_points; number++) {
        const float real = lv_creal(*complexVectorPtr) * inv_scalar;
        const float imag = lv_cimag(*complexVectorPtr) * inv_scalar;
        *magnitudeVectorPtr =
            (int16_t)rintf(sqrtf((real * real) + (imag * imag)) * scalar);
        complexVectorPtr++;
        magnitudeVectorPtr++;
    }
}
#endif /* LV_HAVE_NEONV7 */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16ic_magnitude_16i_neonv8(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;

    const float scalar = SHRT_MAX;
    const float inv_scalar = 1.0f / scalar;

    int16_t* magnitudeVectorPtr = magnitudeVector;
    const lv_16sc_t* complexVectorPtr = complexVector;

    float32x4_t mag_vec, mag_sq;
    float32x4x2_t c_vec;

    for (number = 0; number < quarter_points; number++) {
        const int16x4x2_t c16_vec = vld2_s16((int16_t*)complexVectorPtr);
        __VOLK_PREFETCH(complexVectorPtr + 4);
        c_vec.val[0] = vcvtq_f32_s32(vmovl_s16(c16_vec.val[0]));
        c_vec.val[1] = vcvtq_f32_s32(vmovl_s16(c16_vec.val[1]));
        // Scale to close to 0-1
        c_vec.val[0] = vmulq_n_f32(c_vec.val[0], inv_scalar);
        c_vec.val[1] = vmulq_n_f32(c_vec.val[1], inv_scalar);
        // ARMv8: Use FMA for magnitude squared and native sqrt
        mag_sq =
            vfmaq_f32(vmulq_f32(c_vec.val[0], c_vec.val[0]), c_vec.val[1], c_vec.val[1]);
        mag_vec = vsqrtq_f32(mag_sq);
        // Reconstruct
        mag_vec = vmulq_n_f32(mag_vec, scalar);
        // Add 0.5 for correct rounding because vcvtq_s32_f32 truncates.
        mag_vec = vaddq_f32(mag_vec, vdupq_n_f32(0.5f));
        const int16x4_t mag16_vec = vmovn_s32(vcvtq_s32_f32(mag_vec));
        vst1_s16(magnitudeVectorPtr, mag16_vec);
        // Advance pointers
        magnitudeVectorPtr += 4;
        complexVectorPtr += 4;
    }

    // Deal with the rest
    for (number = quarter_points * 4; number < num_points; number++) {
        const float real = lv_creal(*complexVectorPtr) * inv_scalar;
        const float imag = lv_cimag(*complexVectorPtr) * inv_scalar;
        *magnitudeVectorPtr =
            (int16_t)rintf(sqrtf((real * real) + (imag * imag)) * scalar);
        complexVectorPtr++;
        magnitudeVectorPtr++;
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16ic_magnitude_16i_rvv(int16_t* magnitudeVector,
                                               const lv_16sc_t* complexVector,
                                               unsigned int num_points)
{
    const float scale = SHRT_MAX, iscale = 1.0f / scale;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint32m8_t vc = __riscv_vle32_v_i32m8((const int32_t*)complexVector, vl);
        vint16m4_t vr = __riscv_vnsra(vc, 0, vl);
        vint16m4_t vi = __riscv_vnsra(vc, 16, vl);
        vfloat32m8_t vrf = __riscv_vfmul(__riscv_vfwcvt_f(vr, vl), iscale, vl);
        vfloat32m8_t vif = __riscv_vfmul(__riscv_vfwcvt_f(vi, vl), iscale, vl);
        vfloat32m8_t vf = __riscv_vfmacc(__riscv_vfmul(vif, vif, vl), vrf, vrf, vl);
        vf = __riscv_vfmul(__riscv_vfsqrt(vf, vl), scale, vl);
        __riscv_vse16(magnitudeVector, __riscv_vfncvt_x(vf, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_16ic_magnitude_16i_rvvseg(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    const float scale = SHRT_MAX, iscale = 1.0f / scale;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint16m4x2_t vc = __riscv_vlseg2e16_v_i16m4x2((const int16_t*)complexVector, vl);
        vint16m4_t vr = __riscv_vget_i16m4(vc, 0);
        vint16m4_t vi = __riscv_vget_i16m4(vc, 1);
        vfloat32m8_t vrf = __riscv_vfmul(__riscv_vfwcvt_f(vr, vl), iscale, vl);
        vfloat32m8_t vif = __riscv_vfmul(__riscv_vfwcvt_f(vi, vl), iscale, vl);
        vfloat32m8_t vf = __riscv_vfmacc(__riscv_vfmul(vif, vif, vl), vrf, vrf, vl);
        vf = __riscv_vfmul(__riscv_vfsqrt(vf, vl), scale, vl);
        __riscv_vse16(magnitudeVector, __riscv_vfncvt_x(vf, vl), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_16ic_magnitude_16i_u_H */
