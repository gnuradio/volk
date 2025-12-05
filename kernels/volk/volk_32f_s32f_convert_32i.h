/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_convert_32i
 *
 * \b Overview
 *
 * Converts a floating point number to a 32-bit integer after applying a
 * scaling factor.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_convert_32i(int32_t* outputVector, const float* inputVector, const
 * float scalar, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: the input vector of floats.
 * \li scalar: The value multiplied against each point in the input buffer.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: The output vector.
 *
 * \b Example
 * Convert floats from [-1,1] to integers with a scale of 5 to maintain smallest delta
 * \code
 *  int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   int32_t* out = (int32_t*)volk_malloc(sizeof(int32_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = 2.f * ((float)ii / (float)N) - 1.f;
 *   }
 *
 *   // Normalize by the smallest delta (0.2 in this example)
 *   float scale = 5.f;
 *
 *   volk_32f_s32f_convert_32i(out, increasing, scale, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %i\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_convert_32i_u_H
#define INCLUDED_volk_32f_s32f_convert_32i_u_H

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_convert_32i_u_avx(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int eighthPoints = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m256 vScalar = _mm256_set1_ps(scalar);
    __m256 inputVal1;
    __m256i intInputVal1;
    __m256 vmin_val = _mm256_set1_ps(min_val);
    __m256 vmax_val = _mm256_set1_ps(max_val);

    for (; number < eighthPoints; number++) {
        inputVal1 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;

        inputVal1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        intInputVal1 = _mm256_cvtps_epi32(inputVal1);

        _mm256_storeu_si256((__m256i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r > max_val)
            r = max_val;
        else if (r < min_val)
            r = min_val;
        outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_s32f_convert_32i_u_sse2(int32_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m128 vScalar = _mm_set_ps1(scalar);
    __m128 inputVal1;
    __m128i intInputVal1;
    __m128 vmin_val = _mm_set_ps1(min_val);
    __m128 vmax_val = _mm_set_ps1(max_val);

    for (; number < quarterPoints; number++) {
        inputVal1 = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;

        inputVal1 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        intInputVal1 = _mm_cvtps_epi32(inputVal1);

        _mm_storeu_si128((__m128i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r > max_val)
            r = max_val;
        else if (r < min_val)
            r = min_val;
        outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_convert_32i_u_sse(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m128 vScalar = _mm_set_ps1(scalar);
    __m128 ret;
    __m128 vmin_val = _mm_set_ps1(min_val);
    __m128 vmax_val = _mm_set_ps1(max_val);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (; number < quarterPoints; number++) {
        ret = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;

        ret = _mm_max_ps(_mm_min_ps(_mm_mul_ps(ret, vScalar), vmax_val), vmin_val);

        _mm_store_ps(outputFloatBuffer, ret);
        *outputVectorPtr++ = (int32_t)rintf(outputFloatBuffer[0]);
        *outputVectorPtr++ = (int32_t)rintf(outputFloatBuffer[1]);
        *outputVectorPtr++ = (int32_t)rintf(outputFloatBuffer[2]);
        *outputVectorPtr++ = (int32_t)rintf(outputFloatBuffer[3]);
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r > max_val)
            r = max_val;
        else if (r < min_val)
            r = min_val;
        outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_convert_32i_generic(int32_t* outputVector,
                                                     const float* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    int32_t* outputVectorPtr = outputVector;
    const float* inputVectorPtr = inputVector;
    const float min_val = (float)INT_MIN;
    const float max_val = (float)((uint32_t)INT_MAX + 1);

    for (unsigned int number = 0; number < num_points; number++) {
        const float r = *inputVectorPtr++ * scalar;
        int s;
        if (r >= max_val)
            s = INT_MAX;
        else if (r < min_val)
            s = INT_MIN;
        else
            s = (int32_t)rintf(r);
        *outputVectorPtr++ = s;
    }
}

#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32f_s32f_convert_32i_u_H */
#ifndef INCLUDED_volk_32f_s32f_convert_32i_a_H
#define INCLUDED_volk_32f_s32f_convert_32i_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_convert_32i_a_avx(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int eighthPoints = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m256 vScalar = _mm256_set1_ps(scalar);
    __m256 inputVal1;
    __m256i intInputVal1;
    __m256 vmin_val = _mm256_set1_ps(min_val);
    __m256 vmax_val = _mm256_set1_ps(max_val);

    for (; number < eighthPoints; number++) {
        inputVal1 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;

        inputVal1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        intInputVal1 = _mm256_cvtps_epi32(inputVal1);

        _mm256_store_si256((__m256i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r > max_val)
            r = max_val;
        else if (r < min_val)
            r = min_val;
        outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_s32f_convert_32i_a_sse2(int32_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m128 vScalar = _mm_set_ps1(scalar);
    __m128 inputVal1;
    __m128i intInputVal1;
    __m128 vmin_val = _mm_set_ps1(min_val);
    __m128 vmax_val = _mm_set_ps1(max_val);

    for (; number < quarterPoints; number++) {
        inputVal1 = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;

        inputVal1 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        intInputVal1 = _mm_cvtps_epi32(inputVal1);

        _mm_store_si128((__m128i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r > max_val)
            r = max_val;
        else if (r < min_val)
            r = min_val;
        outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_convert_32i_a_sse(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m128 vScalar = _mm_set_ps1(scalar);
    __m128 ret;
    __m128 vmin_val = _mm_set_ps1(min_val);
    __m128 vmax_val = _mm_set_ps1(max_val);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (; number < quarterPoints; number++) {
        ret = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;

        ret = _mm_max_ps(_mm_min_ps(_mm_mul_ps(ret, vScalar), vmax_val), vmin_val);

        _mm_store_ps(outputFloatBuffer, ret);
        *outputVectorPtr++ = (int32_t)rintf(outputFloatBuffer[0]);
        *outputVectorPtr++ = (int32_t)rintf(outputFloatBuffer[1]);
        *outputVectorPtr++ = (int32_t)rintf(outputFloatBuffer[2]);
        *outputVectorPtr++ = (int32_t)rintf(outputFloatBuffer[3]);
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r > max_val)
            r = max_val;
        else if (r < min_val)
            r = min_val;
        outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_convert_32i_neon(int32_t* outputVector,
                                                  const float* inputVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    const float* inputPtr = inputVector;
    int32_t* outputPtr = outputVector;

    const float min_val = (float)INT_MIN;
    const float max_val = (float)((uint32_t)INT_MAX + 1);

    float32x4_t vScalar = vdupq_n_f32(scalar);
    float32x4_t vmin_val = vdupq_n_f32(min_val);
    float32x4_t vmax_val = vdupq_n_f32(max_val);

    for (; number < quarter_points; number++) {
        float32x4_t inputVal = vld1q_f32(inputPtr);
        inputVal = vmulq_f32(inputVal, vScalar);
        inputVal = vmaxq_f32(vminq_f32(inputVal, vmax_val), vmin_val);
        int32x4_t intVal = vcvtq_s32_f32(inputVal);
        vst1q_s32(outputPtr, intVal);
        inputPtr += 4;
        outputPtr += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        float r = *inputPtr++ * scalar;
        if (r >= max_val)
            *outputPtr++ = INT_MAX;
        else if (r < min_val)
            *outputPtr++ = INT_MIN;
        else
            *outputPtr++ = (int32_t)rintf(r);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_convert_32i_neonv8(int32_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    const float* inputPtr = inputVector;
    int32_t* outputPtr = outputVector;

    const float min_val = (float)INT_MIN;
    const float max_val = (float)((uint32_t)INT_MAX + 1);

    float32x4_t vScalar = vdupq_n_f32(scalar);
    float32x4_t vmin_val = vdupq_n_f32(min_val);
    float32x4_t vmax_val = vdupq_n_f32(max_val);

    for (; number < eighth_points; number++) {
        float32x4_t inputVal0 = vld1q_f32(inputPtr);
        float32x4_t inputVal1 = vld1q_f32(inputPtr + 4);
        __VOLK_PREFETCH(inputPtr + 8);

        inputVal0 = vmulq_f32(inputVal0, vScalar);
        inputVal1 = vmulq_f32(inputVal1, vScalar);
        inputVal0 = vmaxq_f32(vminq_f32(inputVal0, vmax_val), vmin_val);
        inputVal1 = vmaxq_f32(vminq_f32(inputVal1, vmax_val), vmin_val);

        int32x4_t intVal0 = vcvtnq_s32_f32(inputVal0);
        int32x4_t intVal1 = vcvtnq_s32_f32(inputVal1);

        vst1q_s32(outputPtr, intVal0);
        vst1q_s32(outputPtr + 4, intVal1);
        inputPtr += 8;
        outputPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        float r = *inputPtr++ * scalar;
        if (r >= max_val)
            *outputPtr++ = INT_MAX;
        else if (r < min_val)
            *outputPtr++ = INT_MIN;
        else
            *outputPtr++ = (int32_t)rintf(r);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_s32f_convert_32i_rvv(int32_t* outputVector,
                                                 const float* inputVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(inputVector, vl);
        v = __riscv_vfmul(v, scalar, vl);
        __riscv_vse32(outputVector, __riscv_vfcvt_x(v, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_s32f_convert_32i_a_H */
