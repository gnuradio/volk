/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32i_s32f_convert_32f
 *
 * \b Overview
 *
 * Converts the samples in the inputVector from 32-bit integers into
 * floating point values and then divides them by the input scalar.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32i_s32f_convert_32f(float* outputVector, const int32_t* inputVector, const
 * float scalar, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: The vector of 32-bit integers.
 * \li scalar: The value that the output is divided by after being converted to a float.
 * \li num_points: The number of values.
 *
 * \b Outputs
 * \li complexVector: The output vector of floats.
 *
 * \b Example
 * Convert full-range integers to floats in range [0,1].
 * \code
 *   int N = 1<<8;
 *   unsigned int alignment = volk_get_alignment();
 *
 *   int32_t* x = (int32_t*)volk_malloc(N*sizeof(int32_t), alignment);
 *   float* z = (float*)volk_malloc(N*sizeof(float), alignment);
 *   float scale = (float)N;
 *   for(unsigned int ii=0; ii<N; ++ii){
 *       x[ii] = ii;
 *   }
 *
 *   volk_32i_s32f_convert_32f(z, x, scale, N);
 *
 *   volk_free(x);
 *   volk_free(z);
 * \endcode
 */

#ifndef INCLUDED_volk_32i_s32f_convert_32f_u_H
#define INCLUDED_volk_32i_s32f_convert_32f_u_H

#include <inttypes.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32i_s32f_convert_32f_generic(float* outputVector,
                                                     const int32_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    const float invScalar = 1.0f / scalar;
    for (unsigned int number = 0; number < num_points; ++number) {
        *outputVector++ = (float)*inputVector++ * invScalar;
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32i_s32f_convert_32f_u_avx512f(float* outputVector,
                                                       const int32_t* inputVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    const unsigned int onesixteenthPoints = num_points / 16;
    const __m512 invScalar = _mm512_set1_ps(1.0f / scalar);

    for (unsigned int number = 0; number < onesixteenthPoints; ++number) {
        const __m512i inputVal = _mm512_loadu_si512((const __m512i*)inputVector);
        const __m512 output = _mm512_mul_ps(_mm512_cvtepi32_ps(inputVal), invScalar);
        _mm512_storeu_ps(outputVector, output);
        inputVector += 16;
        outputVector += 16;
    }

    volk_32i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - onesixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32i_s32f_convert_32f_u_avx2(float* outputVector,
                                                    const int32_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int oneEightPoints = num_points / 8;
    const __m256 invScalar = _mm256_set1_ps(1.0f / scalar);

    for (unsigned int number = 0; number < oneEightPoints; ++number) {
        const __m256i inputVal = _mm256_loadu_si256((const __m256i*)inputVector);
        const __m256 output = _mm256_mul_ps(_mm256_cvtepi32_ps(inputVal), invScalar);
        _mm256_storeu_ps(outputVector, output);
        inputVector += 8;
        outputVector += 8;
    }

    volk_32i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - oneEightPoints * 8);
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32i_s32f_convert_32f_u_sse2(float* outputVector,
                                                    const int32_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;
    const __m128 invScalar = _mm_set1_ps(1.0f / scalar);

    for (unsigned int number = 0; number < quarterPoints; ++number) {
        const __m128i inputVal = _mm_loadu_si128((const __m128i*)inputVector);
        const __m128 output = _mm_mul_ps(_mm_cvtepi32_ps(inputVal), invScalar);
        _mm_storeu_ps(outputVector, output);
        inputVector += 4;
        outputVector += 4;
    }

    volk_32i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE2 */


#endif /* INCLUDED_volk_32i_s32f_convert_32f_u_H */


#ifndef INCLUDED_volk_32i_s32f_convert_32f_a_H
#define INCLUDED_volk_32i_s32f_convert_32f_a_H

#include <inttypes.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32i_s32f_convert_32f_a_avx512f(float* outputVector,
                                                       const int32_t* inputVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    const unsigned int onesixteenthPoints = num_points / 16;
    const __m512 invScalar = _mm512_set1_ps(1.0f / scalar);

    for (unsigned int number = 0; number < onesixteenthPoints; ++number) {
        const __m512i inputVal = _mm512_load_si512((const __m512i*)inputVector);
        const __m512 output = _mm512_mul_ps(_mm512_cvtepi32_ps(inputVal), invScalar);
        _mm512_store_ps(outputVector, output);
        inputVector += 16;
        outputVector += 16;
    }

    volk_32i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - onesixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32i_s32f_convert_32f_a_avx2(float* outputVector,
                                                    const int32_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int oneEightPoints = num_points / 8;
    const __m256 invScalar = _mm256_set1_ps(1.0f / scalar);

    for (unsigned int number = 0; number < oneEightPoints; ++number) {
        const __m256i inputVal = _mm256_load_si256((const __m256i*)inputVector);
        const __m256 output = _mm256_mul_ps(_mm256_cvtepi32_ps(inputVal), invScalar);
        _mm256_store_ps(outputVector, output);
        inputVector += 8;
        outputVector += 8;
    }

    volk_32i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - oneEightPoints * 8);
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32i_s32f_convert_32f_a_sse2(float* outputVector,
                                                    const int32_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;
    const __m128 invScalar = _mm_set1_ps(1.0f / scalar);

    for (unsigned int number = 0; number < quarterPoints; ++number) {
        const __m128i inputVal = _mm_load_si128((const __m128i*)inputVector);
        const __m128 output = _mm_mul_ps(_mm_cvtepi32_ps(inputVal), invScalar);
        _mm_store_ps(outputVector, output);
        inputVector += 4;
        outputVector += 4;
    }

    volk_32i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32i_s32f_convert_32f_neon(float* outputVector,
                                                  const int32_t* inputVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;
    const float32x4_t invScalar = vdupq_n_f32(1.0f / scalar);

    for (unsigned int number = 0; number < quarterPoints; ++number) {
        const int32x4_t inputVal = vld1q_s32(inputVector);
        const float32x4_t output = vmulq_f32(vcvtq_f32_s32(inputVal), invScalar);
        vst1q_f32(outputVector, output);
        inputVector += 4;
        outputVector += 4;
    }

    volk_32i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32i_s32f_convert_32f_neonv8(float* outputVector,
                                                    const int32_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    const float32x4_t invScalar = vdupq_n_f32(1.0f / scalar);

    for (unsigned int number = 0; number < eighthPoints; ++number) {
        const int32x4_t inputVal0 = vld1q_s32(inputVector);
        const int32x4_t inputVal1 = vld1q_s32(inputVector + 4);
        __VOLK_PREFETCH(inputVector + 8);
        const float32x4_t output0 = vmulq_f32(vcvtq_f32_s32(inputVal0), invScalar);
        const float32x4_t output1 = vmulq_f32(vcvtq_f32_s32(inputVal1), invScalar);
        vst1q_f32(outputVector, output0);
        vst1q_f32(outputVector + 4, output1);
        inputVector += 8;
        outputVector += 8;
    }

    volk_32i_s32f_convert_32f_generic(
        outputVector, inputVector, scalar, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32i_s32f_convert_32f_rvv(float* outputVector,
                                                 const int32_t* inputVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vfcvt_f(__riscv_vle32_v_i32m8(inputVector, vl), vl);
        __riscv_vse32(outputVector, __riscv_vfmul(v, 1.0f / scalar, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32i_s32f_convert_32f_a_H */
