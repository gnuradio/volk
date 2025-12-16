/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_stddev_32f
 *
 * \b Overview
 *
 * Computes the standard deviation of the input buffer using the supplied mean.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_stddev_32f(float* stddev, const float* inputBuffer, const float
 * mean, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputBuffer: The input vector of floats.
 * \li mean: The mean of the input buffer.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li stddev: The output vector.
 *
 * \b Example
 * Calculate the standard deviation from numbers generated with c++11's normal generator
 * \code
 *   int N = 1000;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float mean = 0.0f;
 *   float* stddev = (float*)volk_malloc(sizeof(float), alignment);
 *
 *   // Use a normal generator with 0 mean, stddev = 1
 *   std::default_random_engine generator;
 *   std::normal_distribution<float> distribution(mean,1);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] =  distribution(generator);
 *   }
 *
 *   volk_32f_s32f_power_32f(stddev, increasing, mean, N);
 *
 *   printf("std. dev. = %f\n", *stddev);
 *
 *   volk_free(increasing);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_stddev_32f_a_H
#define INCLUDED_volk_32f_s32f_stddev_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_32f_s32f_stddev_32f_a_sse4_1(float* stddev,
                                                     const float* inputBuffer,
                                                     const float mean,
                                                     unsigned int num_points)
{
    float returnValue = 0;
    if (num_points > 0) {
        unsigned int number = 0;
        const unsigned int sixteenthPoints = num_points / 16;

        const float* aPtr = inputBuffer;

        __VOLK_ATTR_ALIGNED(16) float squareBuffer[4];

        __m128 squareAccumulator = _mm_setzero_ps();
        __m128 aVal1, aVal2, aVal3, aVal4;
        __m128 cVal1, cVal2, cVal3, cVal4;
        for (; number < sixteenthPoints; number++) {
            aVal1 = _mm_load_ps(aPtr);
            aPtr += 4;
            cVal1 = _mm_dp_ps(aVal1, aVal1, 0xF1);

            aVal2 = _mm_load_ps(aPtr);
            aPtr += 4;
            cVal2 = _mm_dp_ps(aVal2, aVal2, 0xF2);

            aVal3 = _mm_load_ps(aPtr);
            aPtr += 4;
            cVal3 = _mm_dp_ps(aVal3, aVal3, 0xF4);

            aVal4 = _mm_load_ps(aPtr);
            aPtr += 4;
            cVal4 = _mm_dp_ps(aVal4, aVal4, 0xF8);

            cVal1 = _mm_or_ps(cVal1, cVal2);
            cVal3 = _mm_or_ps(cVal3, cVal4);
            cVal1 = _mm_or_ps(cVal1, cVal3);

            squareAccumulator =
                _mm_add_ps(squareAccumulator, cVal1); // squareAccumulator += x^2
        }
        _mm_store_ps(squareBuffer,
                     squareAccumulator); // Store the results back into the C container
        returnValue = squareBuffer[0];
        returnValue += squareBuffer[1];
        returnValue += squareBuffer[2];
        returnValue += squareBuffer[3];

        number = sixteenthPoints * 16;
        for (; number < num_points; number++) {
            returnValue += (*aPtr) * (*aPtr);
            aPtr++;
        }
        returnValue /= num_points;
        returnValue -= (mean * mean);
        returnValue = sqrtf(returnValue);
    }
    *stddev = returnValue;
}

#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_stddev_32f_a_sse(float* stddev,
                                                  const float* inputBuffer,
                                                  const float mean,
                                                  unsigned int num_points)
{
    float returnValue = 0;
    if (num_points > 0) {
        unsigned int number = 0;
        const unsigned int quarterPoints = num_points / 4;

        const float* aPtr = inputBuffer;

        __VOLK_ATTR_ALIGNED(16) float squareBuffer[4];

        __m128 squareAccumulator = _mm_setzero_ps();
        __m128 aVal = _mm_setzero_ps();
        for (; number < quarterPoints; number++) {
            aVal = _mm_load_ps(aPtr);      // aVal = x
            aVal = _mm_mul_ps(aVal, aVal); // squareAccumulator += x^2
            squareAccumulator = _mm_add_ps(squareAccumulator, aVal);
            aPtr += 4;
        }
        _mm_store_ps(squareBuffer,
                     squareAccumulator); // Store the results back into the C container
        returnValue = squareBuffer[0];
        returnValue += squareBuffer[1];
        returnValue += squareBuffer[2];
        returnValue += squareBuffer[3];

        number = quarterPoints * 4;
        for (; number < num_points; number++) {
            returnValue += (*aPtr) * (*aPtr);
            aPtr++;
        }
        returnValue /= num_points;
        returnValue -= (mean * mean);
        returnValue = sqrtf(returnValue);
    }
    *stddev = returnValue;
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_stddev_32f_a_avx(float* stddev,
                                                  const float* inputBuffer,
                                                  const float mean,
                                                  unsigned int num_points)
{
    float stdDev = 0;
    if (num_points > 0) {
        unsigned int number = 0;
        const unsigned int thirtySecondthPoints = num_points / 32;

        const float* aPtr = inputBuffer;
        __VOLK_ATTR_ALIGNED(32) float squareBuffer[8];

        __m256 squareAccumulator = _mm256_setzero_ps();
        __m256 aVal1, aVal2, aVal3, aVal4;
        __m256 cVal1, cVal2, cVal3, cVal4;
        for (; number < thirtySecondthPoints; number++) {
            aVal1 = _mm256_load_ps(aPtr);
            aPtr += 8;
            cVal1 = _mm256_dp_ps(aVal1, aVal1, 0xF1);

            aVal2 = _mm256_load_ps(aPtr);
            aPtr += 8;
            cVal2 = _mm256_dp_ps(aVal2, aVal2, 0xF2);

            aVal3 = _mm256_load_ps(aPtr);
            aPtr += 8;
            cVal3 = _mm256_dp_ps(aVal3, aVal3, 0xF4);

            aVal4 = _mm256_load_ps(aPtr);
            aPtr += 8;
            cVal4 = _mm256_dp_ps(aVal4, aVal4, 0xF8);

            cVal1 = _mm256_or_ps(cVal1, cVal2);
            cVal3 = _mm256_or_ps(cVal3, cVal4);
            cVal1 = _mm256_or_ps(cVal1, cVal3);

            squareAccumulator =
                _mm256_add_ps(squareAccumulator, cVal1); // squareAccumulator += x^2
        }
        _mm256_store_ps(squareBuffer,
                        squareAccumulator); // Store the results back into the C container
        stdDev = squareBuffer[0];
        stdDev += squareBuffer[1];
        stdDev += squareBuffer[2];
        stdDev += squareBuffer[3];
        stdDev += squareBuffer[4];
        stdDev += squareBuffer[5];
        stdDev += squareBuffer[6];
        stdDev += squareBuffer[7];

        number = thirtySecondthPoints * 32;
        for (; number < num_points; number++) {
            stdDev += (*aPtr) * (*aPtr);
            aPtr++;
        }
        stdDev /= num_points;
        stdDev -= (mean * mean);
        stdDev = sqrtf(stdDev);
    }
    *stddev = stdDev;
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_stddev_32f_generic(float* stddev,
                                                    const float* inputBuffer,
                                                    const float mean,
                                                    unsigned int num_points)
{
    float returnValue = 0;
    if (num_points > 0) {
        const float* aPtr = inputBuffer;
        unsigned int number = 0;

        for (number = 0; number < num_points; number++) {
            returnValue += (*aPtr) * (*aPtr);
            aPtr++;
        }

        returnValue /= num_points;
        returnValue -= (mean * mean);
        returnValue = sqrtf(returnValue);
    }
    *stddev = returnValue;
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_stddev_32f_neon(float* stddev,
                                                 const float* inputBuffer,
                                                 const float mean,
                                                 unsigned int num_points)
{
    float returnValue = 0;
    if (num_points > 0) {
        unsigned int number = 0;
        const unsigned int quarterPoints = num_points / 4;

        const float* aPtr = inputBuffer;

        float32x4_t squareAccumulator = vdupq_n_f32(0.0f);

        for (; number < quarterPoints; number++) {
            float32x4_t aVal = vld1q_f32(aPtr);
            squareAccumulator = vmlaq_f32(squareAccumulator, aVal, aVal);
            aPtr += 4;
        }

        // Reduce the accumulator
        float32x2_t sum =
            vadd_f32(vget_low_f32(squareAccumulator), vget_high_f32(squareAccumulator));
        sum = vpadd_f32(sum, sum);
        returnValue = vget_lane_f32(sum, 0);

        number = quarterPoints * 4;
        for (; number < num_points; number++) {
            returnValue += (*aPtr) * (*aPtr);
            aPtr++;
        }
        returnValue /= num_points;
        returnValue -= (mean * mean);
        returnValue = sqrtf(returnValue);
    }
    *stddev = returnValue;
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_stddev_32f_neonv8(float* stddev,
                                                   const float* inputBuffer,
                                                   const float mean,
                                                   unsigned int num_points)
{
    float returnValue = 0;
    if (num_points > 0) {
        unsigned int number = 0;
        const unsigned int eighthPoints = num_points / 8;

        const float* aPtr = inputBuffer;

        float32x4_t squareAccumulator0 = vdupq_n_f32(0.0f);
        float32x4_t squareAccumulator1 = vdupq_n_f32(0.0f);

        for (; number < eighthPoints; number++) {
            __VOLK_PREFETCH(aPtr + 16);
            float32x4_t aVal0 = vld1q_f32(aPtr);
            float32x4_t aVal1 = vld1q_f32(aPtr + 4);
            squareAccumulator0 = vfmaq_f32(squareAccumulator0, aVal0, aVal0);
            squareAccumulator1 = vfmaq_f32(squareAccumulator1, aVal1, aVal1);
            aPtr += 8;
        }

        // Combine and reduce the accumulators
        float32x4_t squareAccumulator = vaddq_f32(squareAccumulator0, squareAccumulator1);
        returnValue = vaddvq_f32(squareAccumulator);

        number = eighthPoints * 8;
        for (; number < num_points; number++) {
            returnValue += (*aPtr) * (*aPtr);
            aPtr++;
        }
        returnValue /= num_points;
        returnValue -= (mean * mean);
        returnValue = sqrtf(returnValue);
    }
    *stddev = returnValue;
}

#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32f_s32f_stddev_32f_a_H */

#ifndef INCLUDED_volk_32f_s32f_stddev_32f_u_H
#define INCLUDED_volk_32f_s32f_stddev_32f_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_stddev_32f_u_avx(float* stddev,
                                                  const float* inputBuffer,
                                                  const float mean,
                                                  unsigned int num_points)
{
    float stdDev = 0;
    if (num_points > 0) {
        unsigned int number = 0;
        const unsigned int thirtySecondthPoints = num_points / 32;

        const float* aPtr = inputBuffer;
        __VOLK_ATTR_ALIGNED(32) float squareBuffer[8];

        __m256 squareAccumulator = _mm256_setzero_ps();
        __m256 aVal1, aVal2, aVal3, aVal4;
        __m256 cVal1, cVal2, cVal3, cVal4;
        for (; number < thirtySecondthPoints; number++) {
            aVal1 = _mm256_loadu_ps(aPtr);
            aPtr += 8;
            cVal1 = _mm256_dp_ps(aVal1, aVal1, 0xF1);

            aVal2 = _mm256_loadu_ps(aPtr);
            aPtr += 8;
            cVal2 = _mm256_dp_ps(aVal2, aVal2, 0xF2);

            aVal3 = _mm256_loadu_ps(aPtr);
            aPtr += 8;
            cVal3 = _mm256_dp_ps(aVal3, aVal3, 0xF4);

            aVal4 = _mm256_loadu_ps(aPtr);
            aPtr += 8;
            cVal4 = _mm256_dp_ps(aVal4, aVal4, 0xF8);

            cVal1 = _mm256_or_ps(cVal1, cVal2);
            cVal3 = _mm256_or_ps(cVal3, cVal4);
            cVal1 = _mm256_or_ps(cVal1, cVal3);

            squareAccumulator =
                _mm256_add_ps(squareAccumulator, cVal1); // squareAccumulator += x^2
        }
        _mm256_storeu_ps(
            squareBuffer,
            squareAccumulator); // Store the results back into the C container
        stdDev = squareBuffer[0];
        stdDev += squareBuffer[1];
        stdDev += squareBuffer[2];
        stdDev += squareBuffer[3];
        stdDev += squareBuffer[4];
        stdDev += squareBuffer[5];
        stdDev += squareBuffer[6];
        stdDev += squareBuffer[7];

        number = thirtySecondthPoints * 32;
        for (; number < num_points; number++) {
            stdDev += (*aPtr) * (*aPtr);
            aPtr++;
        }
        stdDev /= num_points;
        stdDev -= (mean * mean);
        stdDev = sqrtf(stdDev);
    }
    *stddev = stdDev;
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_32f_s32f_stddev_32f_rvv(float* stddev,
                                                const float* inputBuffer,
                                                const float mean,
                                                unsigned int num_points)
{
    if (num_points == 0) {
        *stddev = 0;
        return;
    }
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0, __riscv_vsetvlmax_e32m8());
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputBuffer += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(inputBuffer, vl);
        vsum = __riscv_vfmacc_tu(vsum, v, v, vl);
    }
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t v = RISCV_SHRINK8(vfadd, f, 32, vsum);
    v = __riscv_vfredusum(v, __riscv_vfmv_s_f_f32m1(0, vl), vl);
    float sum = __riscv_vfmv_f(v);
    *stddev = sqrtf((sum / num_points) - (mean * mean));
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_s32f_stddev_32f_u_H */
