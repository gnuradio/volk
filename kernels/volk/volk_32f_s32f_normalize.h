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

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_normalize_generic(float* vecBuffer,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    const float invScalar = 1.0f / scalar;
    for (unsigned int number = 0; number < num_points; number++) {
        *vecBuffer++ *= invScalar;
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_normalize_a_avx(float* vecBuffer,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    const float invScalar = 1.0f / scalar;

    const __m256 vecScalar = _mm256_set1_ps(invScalar);
    for (unsigned int number = 0; number < eighthPoints; number++) {
        const __m256 input = _mm256_load_ps(vecBuffer);
        const __m256 result = _mm256_mul_ps(input, vecScalar);
        _mm256_store_ps(vecBuffer, result);
        vecBuffer += 8;
    }

    volk_32f_s32f_normalize_generic(vecBuffer, scalar, num_points - 8 * eighthPoints);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_normalize_a_sse(float* vecBuffer,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    const unsigned int blocks = num_points / 4;
    const float invScalar = 1.0f / scalar;
    const __m128 vecScalar = _mm_set_ps1(invScalar);
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m128 input = _mm_load_ps(vecBuffer);
        const __m128 result = _mm_mul_ps(input, vecScalar);
        _mm_store_ps(vecBuffer, result);
        vecBuffer += 4;
    }
    volk_32f_s32f_normalize_generic(vecBuffer, scalar, num_points - 4 * blocks);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_ORC

extern void volk_32f_s32f_normalize_a_orc_impl(float* dst,
                                               float* src,
                                               const float scalar,
                                               int num_points);
static inline void volk_32f_s32f_normalize_u_orc(float* vecBuffer,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    float invscalar = 1.0f / scalar;
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
    const unsigned int blocks = num_points / 8;
    const float invScalar = 1.0f / scalar;
    const __m256 vecScalar = _mm256_set1_ps(invScalar);
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m256 input = _mm256_loadu_ps(vecBuffer);
        const __m256 result = _mm256_mul_ps(input, vecScalar);
        _mm256_storeu_ps(vecBuffer, result);
        vecBuffer += 8;
    }
    volk_32f_s32f_normalize_generic(vecBuffer, scalar, num_points - 8 * blocks);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_normalize_neon(float* vecBuffer,
                                                const float scalar,
                                                unsigned int num_points)
{
    const unsigned int blocks = num_points / 4;
    const float invScalar = 1.0f / scalar;
    const float32x4_t vecScalar = vdupq_n_f32(invScalar);
    for (unsigned int number = 0; number < blocks; ++number) {
        const float32x4_t input = vld1q_f32(vecBuffer);
        const float32x4_t result = vmulq_f32(input, vecScalar);
        vst1q_f32(vecBuffer, result);
        vecBuffer += 4;
    }
    volk_32f_s32f_normalize_generic(vecBuffer, scalar, num_points - 4 * blocks);
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_normalize_neonv8(float* vecBuffer,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    const unsigned int blocks = num_points / 8;
    const float invScalar = 1.0f / scalar;
    const float32x4_t vecScalar = vdupq_n_f32(invScalar);
    for (unsigned int number = 0; number < blocks; ++number) {
        const float32x4_t input = vld1q_f32(vecBuffer);
        const float32x4_t input1 = vld1q_f32(vecBuffer + 4);
        __VOLK_PREFETCH(vecBuffer + 8);
        const float32x4_t result = vmulq_f32(input, vecScalar);
        const float32x4_t result1 = vmulq_f32(input1, vecScalar);
        vst1q_f32(vecBuffer, result);
        vst1q_f32(vecBuffer + 4, result1);
        vecBuffer += 8;
    }
    volk_32f_s32f_normalize_generic(vecBuffer, scalar, num_points - 8 * blocks);
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
