/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_normalize
 *
 * \b Overview
 *
 * Normalizes all points in the buffer by the scalar value (divides
 * each data point by the scalar value).
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_normalize(float* vecBuffer, const float scalar, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li vecBuffer: The buffer of values to be vectorized.
 * \li scalar: The scale value to be applied to each buffer value.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li vecBuffer: returns as an in-place calculation.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = 2.f * ((float)ii / (float)N) - 1.f;
 *   }
 *
 *   // Normalize by the smallest delta (0.2 in this example)
 *   float scale = 5.0f;
 *
 *   volk_32f_s32f_normalize(increasing, scale, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("increasing[%u] = %f\n", ii, increasing[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_normalize_a_H
#define INCLUDED_volk_32f_s32f_normalize_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_normalize_a_avx(float* vecBuffer,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    unsigned int number = 0;
    float* inputPtr = vecBuffer;

    const float invScalar = 1.0 / scalar;
    __m256 vecScalar = _mm256_set1_ps(invScalar);

    __m256 input1;

    const uint64_t eighthPoints = num_points / 8;
    for (; number < eighthPoints; number++) {

        input1 = _mm256_load_ps(inputPtr);

        input1 = _mm256_mul_ps(input1, vecScalar);

        _mm256_store_ps(inputPtr, input1);

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *inputPtr *= invScalar;
        inputPtr++;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_normalize_a_sse(float* vecBuffer,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    unsigned int number = 0;
    float* inputPtr = vecBuffer;

    const float invScalar = 1.0 / scalar;
    __m128 vecScalar = _mm_set_ps1(invScalar);

    __m128 input1;

    const uint64_t quarterPoints = num_points / 4;
    for (; number < quarterPoints; number++) {

        input1 = _mm_load_ps(inputPtr);

        input1 = _mm_mul_ps(input1, vecScalar);

        _mm_store_ps(inputPtr, input1);

        inputPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *inputPtr *= invScalar;
        inputPtr++;
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_normalize_generic(float* vecBuffer,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    float* inputPtr = vecBuffer;
    const float invScalar = 1.0 / scalar;
    for (number = 0; number < num_points; number++) {
        *inputPtr *= invScalar;
        inputPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_ORC

extern void volk_32f_s32f_normalize_a_orc_impl(float* dst,
                                               float* src,
                                               const float scalar,
                                               int num_points);
static inline void volk_32f_s32f_normalize_u_orc(float* vecBuffer,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    float invscalar = 1.0 / scalar;
    volk_32f_s32f_normalize_a_orc_impl(vecBuffer, vecBuffer, invscalar, num_points);
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32f_s32f_normalize_a_H */

#ifndef INCLUDED_volk_32f_s32f_normalize_u_H
#define INCLUDED_volk_32f_s32f_normalize_u_H

#include <inttypes.h>
#include <stdio.h>
#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_normalize_u_avx(float* vecBuffer,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    unsigned int number = 0;
    float* inputPtr = vecBuffer;

    const float invScalar = 1.0 / scalar;
    __m256 vecScalar = _mm256_set1_ps(invScalar);

    __m256 input1;

    const uint64_t eighthPoints = num_points / 8;
    for (; number < eighthPoints; number++) {

        input1 = _mm256_loadu_ps(inputPtr);

        input1 = _mm256_mul_ps(input1, vecScalar);

        _mm256_storeu_ps(inputPtr, input1);

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *inputPtr *= invScalar;
        inputPtr++;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_normalize_neon(float* vecBuffer,
                                                const float scalar,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    float* inputPtr = vecBuffer;
    const float invScalar = 1.0f / scalar;
    float32x4_t vInvScalar = vdupq_n_f32(invScalar);
    const unsigned int quarter_points = num_points / 4;

    for (; number < quarter_points; number++) {
        float32x4_t input = vld1q_f32(inputPtr);
        input = vmulq_f32(input, vInvScalar);
        vst1q_f32(inputPtr, input);
        inputPtr += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        *inputPtr++ *= invScalar;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_normalize_neonv8(float* vecBuffer,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    float* inputPtr = vecBuffer;
    const float invScalar = 1.0f / scalar;
    float32x4_t vInvScalar = vdupq_n_f32(invScalar);
    const unsigned int eighth_points = num_points / 8;

    for (; number < eighth_points; number++) {
        float32x4_t input0 = vld1q_f32(inputPtr);
        float32x4_t input1 = vld1q_f32(inputPtr + 4);
        __VOLK_PREFETCH(inputPtr + 8);

        input0 = vmulq_f32(input0, vInvScalar);
        input1 = vmulq_f32(input1, vInvScalar);

        vst1q_f32(inputPtr, input0);
        vst1q_f32(inputPtr + 4, input1);
        inputPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *inputPtr++ *= invScalar;
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_s32f_normalize_rvv(float* vecBuffer, const float scalar, unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, vecBuffer += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(vecBuffer, vl);
        __riscv_vse32(vecBuffer, __riscv_vfmul(v, 1.0f / scalar, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_s32f_normalize_u_H */
