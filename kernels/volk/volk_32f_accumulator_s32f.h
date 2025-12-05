/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_accumulator_s32f
 *
 * \b Overview
 *
 * Accumulates the values in the input buffer.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_accumulator_s32f(float* result, const float* inputBuffer, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li inputBuffer The buffer of data to be accumulated
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li result The accumulated result.
 *
 * \b Example
 * Calculate the sum of numbers  0 through 99
 * \code
 *   int N = 100;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float), alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (float)ii;
 *   }
 *
 *   volk_32f_accumulator_s32f(out, increasing, N);
 *
 *   printf("sum(1..100) = %1.2f\n", out[0]);
 *
 *   volk_free(increasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_accumulator_s32f_a_H
#define INCLUDED_volk_32f_accumulator_s32f_a_H

#include <inttypes.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_accumulator_s32f_a_avx512f(float* result,
                                                       const float* inputBuffer,
                                                       unsigned int num_points)
{
    float returnValue = 0;
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const float* aPtr = inputBuffer;

    __m512 accumulator = _mm512_setzero_ps();
    __m512 aVal = _mm512_setzero_ps();

    for (; number < sixteenthPoints; number++) {
        aVal = _mm512_load_ps(aPtr);
        accumulator = _mm512_add_ps(accumulator, aVal);
        aPtr += 16;
    }

    // Horizontal sum using AVX512 reduce instruction
    returnValue = _mm512_reduce_add_ps(accumulator);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_accumulator_s32f_a_avx(float* result,
                                                   const float* inputBuffer,
                                                   unsigned int num_points)
{
    float returnValue = 0;
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(32) float tempBuffer[8];

    __m256 accumulator = _mm256_setzero_ps();
    __m256 aVal = _mm256_setzero_ps();

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        accumulator = _mm256_add_ps(accumulator, aVal);
        aPtr += 8;
    }

    _mm256_store_ps(tempBuffer, accumulator);

    returnValue = tempBuffer[0];
    returnValue += tempBuffer[1];
    returnValue += tempBuffer[2];
    returnValue += tempBuffer[3];
    returnValue += tempBuffer[4];
    returnValue += tempBuffer[5];
    returnValue += tempBuffer[6];
    returnValue += tempBuffer[7];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_accumulator_s32f_u_avx512f(float* result,
                                                       const float* inputBuffer,
                                                       unsigned int num_points)
{
    float returnValue = 0;
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const float* aPtr = inputBuffer;

    __m512 accumulator = _mm512_setzero_ps();
    __m512 aVal = _mm512_setzero_ps();

    for (; number < sixteenthPoints; number++) {
        aVal = _mm512_loadu_ps(aPtr);
        accumulator = _mm512_add_ps(accumulator, aVal);
        aPtr += 16;
    }

    // Horizontal sum using AVX512 reduce instruction
    returnValue = _mm512_reduce_add_ps(accumulator);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_accumulator_s32f_u_avx(float* result,
                                                   const float* inputBuffer,
                                                   unsigned int num_points)
{
    float returnValue = 0;
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(32) float tempBuffer[8];

    __m256 accumulator = _mm256_setzero_ps();
    __m256 aVal = _mm256_setzero_ps();

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        accumulator = _mm256_add_ps(accumulator, aVal);
        aPtr += 8;
    }

    _mm256_store_ps(tempBuffer, accumulator);

    returnValue = tempBuffer[0];
    returnValue += tempBuffer[1];
    returnValue += tempBuffer[2];
    returnValue += tempBuffer[3];
    returnValue += tempBuffer[4];
    returnValue += tempBuffer[5];
    returnValue += tempBuffer[6];
    returnValue += tempBuffer[7];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_accumulator_s32f_a_sse(float* result,
                                                   const float* inputBuffer,
                                                   unsigned int num_points)
{
    float returnValue = 0;
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(16) float tempBuffer[4];

    __m128 accumulator = _mm_setzero_ps();
    __m128 aVal = _mm_setzero_ps();

    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);
        accumulator = _mm_add_ps(accumulator, aVal);
        aPtr += 4;
    }

    _mm_store_ps(tempBuffer, accumulator);

    returnValue = tempBuffer[0];
    returnValue += tempBuffer[1];
    returnValue += tempBuffer[2];
    returnValue += tempBuffer[3];

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_accumulator_s32f_u_sse(float* result,
                                                   const float* inputBuffer,
                                                   unsigned int num_points)
{
    float returnValue = 0;
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(16) float tempBuffer[4];

    __m128 accumulator = _mm_setzero_ps();
    __m128 aVal = _mm_setzero_ps();

    for (; number < quarterPoints; number++) {
        aVal = _mm_loadu_ps(aPtr);
        accumulator = _mm_add_ps(accumulator, aVal);
        aPtr += 4;
    }

    _mm_store_ps(tempBuffer, accumulator);

    returnValue = tempBuffer[0];
    returnValue += tempBuffer[1];
    returnValue += tempBuffer[2];
    returnValue += tempBuffer[3];

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_accumulator_s32f_neon(float* result,
                                                  const float* inputBuffer,
                                                  unsigned int num_points)
{
    float returnValue = 0;
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* aPtr = inputBuffer;
    float32x4_t accumulator = vdupq_n_f32(0.0f);
    float32x4_t aVal;

    for (; number < quarterPoints; number++) {
        aVal = vld1q_f32(aPtr);
        accumulator = vaddq_f32(accumulator, aVal);
        aPtr += 4;
    }

    // Horizontal sum - manual for NEON (ARMv7 compatible)
    float32x2_t sum_pair =
        vadd_f32(vget_low_f32(accumulator), vget_high_f32(accumulator));
    sum_pair = vpadd_f32(sum_pair, sum_pair);
    returnValue = vget_lane_f32(sum_pair, 0);

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_accumulator_s32f_neonv8(float* result,
                                                    const float* inputBuffer,
                                                    unsigned int num_points)
{
    float returnValue = 0;
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* aPtr = inputBuffer;
    float32x4_t accumulator0 = vdupq_n_f32(0.0f);
    float32x4_t accumulator1 = vdupq_n_f32(0.0f);

    // 2x unrolled loop for better instruction-level parallelism
    for (; number < eighthPoints; number++) {
        float32x4_t aVal0 = vld1q_f32(aPtr);
        float32x4_t aVal1 = vld1q_f32(aPtr + 4);
        __VOLK_PREFETCH(aPtr + 8);
        accumulator0 = vaddq_f32(accumulator0, aVal0);
        accumulator1 = vaddq_f32(accumulator1, aVal1);
        aPtr += 8;
    }

    // Combine accumulators
    accumulator0 = vaddq_f32(accumulator0, accumulator1);

    // ARMv8 horizontal sum using vaddvq_f32
    returnValue = vaddvq_f32(accumulator0);

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_GENERIC
static inline void volk_32f_accumulator_s32f_generic(float* result,
                                                     const float* inputBuffer,
                                                     unsigned int num_points)
{
    const float* aPtr = inputBuffer;
    unsigned int number = 0;
    float returnValue = 0;

    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_32f_accumulator_s32f_rvv(float* result,
                                                 const float* inputBuffer,
                                                 unsigned int num_points)
{
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0, __riscv_vsetvlmax_e32m8());
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputBuffer += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(inputBuffer, vl);
        vsum = __riscv_vfadd_tu(vsum, vsum, v, vl);
    }
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t v = RISCV_SHRINK8(vfadd, f, 32, vsum);
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0, vl);
    *result = __riscv_vfmv_f(__riscv_vfredusum(v, z, vl));
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_accumulator_s32f_a_H */
