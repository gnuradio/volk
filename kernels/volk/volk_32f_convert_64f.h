/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_convert_64f
 *
 * \b Overview
 *
 * Converts float values into doubles.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_convert_64f(double* outputVector, const float* inputVector, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: The vector of floats to convert to doubles.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: returns the converted doubles.
 *
 * \b Example
 * Generate floats and convert them to doubles.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   double* out = (double*)volk_malloc(sizeof(double)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       in[ii] = (float)ii;
 *   }
 *
 *   volk_32f_convert_64f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %g\n", ii, out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */


#ifndef INCLUDED_volk_32f_convert_64f_u_H
#define INCLUDED_volk_32f_convert_64f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_convert_64f_u_avx(double* outputVector,
                                              const float* inputVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    double* outputVectorPtr = outputVector;
    __m256d ret;
    __m128 inputVal;

    for (; number < quarterPoints; number++) {
        inputVal = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;

        ret = _mm256_cvtps_pd(inputVal);
        _mm256_storeu_pd(outputVectorPtr, ret);

        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = (double)(inputVector[number]);
    }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_convert_64f_u_sse2(double* outputVector,
                                               const float* inputVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    double* outputVectorPtr = outputVector;
    __m128d ret;
    __m128 inputVal;

    for (; number < quarterPoints; number++) {
        inputVal = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;

        ret = _mm_cvtps_pd(inputVal);

        _mm_storeu_pd(outputVectorPtr, ret);
        outputVectorPtr += 2;

        inputVal = _mm_movehl_ps(inputVal, inputVal);

        ret = _mm_cvtps_pd(inputVal);

        _mm_storeu_pd(outputVectorPtr, ret);
        outputVectorPtr += 2;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = (double)(inputVector[number]);
    }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_convert_64f_generic(double* outputVector,
                                                const float* inputVector,
                                                unsigned int num_points)
{
    double* outputVectorPtr = outputVector;
    const float* inputVectorPtr = inputVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *outputVectorPtr++ = ((double)(*inputVectorPtr++));
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32f_convert_64f_u_H */


#ifndef INCLUDED_volk_32f_convert_64f_a_H
#define INCLUDED_volk_32f_convert_64f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_convert_64f_a_avx(double* outputVector,
                                              const float* inputVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    double* outputVectorPtr = outputVector;
    __m256d ret;
    __m128 inputVal;

    for (; number < quarterPoints; number++) {
        inputVal = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;

        ret = _mm256_cvtps_pd(inputVal);
        _mm256_store_pd(outputVectorPtr, ret);

        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = (double)(inputVector[number]);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_convert_64f_a_sse2(double* outputVector,
                                               const float* inputVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    double* outputVectorPtr = outputVector;
    __m128d ret;
    __m128 inputVal;

    for (; number < quarterPoints; number++) {
        inputVal = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;

        ret = _mm_cvtps_pd(inputVal);

        _mm_store_pd(outputVectorPtr, ret);
        outputVectorPtr += 2;

        inputVal = _mm_movehl_ps(inputVal, inputVal);

        ret = _mm_cvtps_pd(inputVal);

        _mm_store_pd(outputVectorPtr, ret);
        outputVectorPtr += 2;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = (double)(inputVector[number]);
    }
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_convert_64f_neonv8(double* outputVector,
                                               const float* inputVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    const float* inputPtr = inputVector;
    double* outputPtr = outputVector;

    for (; number < eighth_points; number++) {
        float32x4_t in0 = vld1q_f32(inputPtr);
        float32x4_t in1 = vld1q_f32(inputPtr + 4);
        __VOLK_PREFETCH(inputPtr + 8);

        float64x2_t out0 = vcvt_f64_f32(vget_low_f32(in0));
        float64x2_t out1 = vcvt_f64_f32(vget_high_f32(in0));
        float64x2_t out2 = vcvt_f64_f32(vget_low_f32(in1));
        float64x2_t out3 = vcvt_f64_f32(vget_high_f32(in1));

        vst1q_f64(outputPtr, out0);
        vst1q_f64(outputPtr + 2, out1);
        vst1q_f64(outputPtr + 4, out2);
        vst1q_f64(outputPtr + 6, out3);

        inputPtr += 8;
        outputPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *outputPtr++ = (double)(*inputPtr++);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_convert_64f_rvv(double* outputVector,
                                            const float* inputVector,
                                            unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(inputVector, vl);
        __riscv_vse64(outputVector, __riscv_vfwcvt_f(v, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_convert_64f_a_H */
