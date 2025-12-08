/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_64f_convert_32f
 *
 * \b Overview
 *
 * Converts doubles into floats.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_64f_convert_32f(float* outputVector, const double* inputVector, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: The vector of doubles to convert to floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: returns the converted floats.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   double* increasing = (double*)volk_malloc(sizeof(double)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (double)ii;
 *   }
 *
 *   volk_64f_convert_32f(out, increasing, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %1.2f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_64f_convert_32f_u_H
#define INCLUDED_volk_64f_convert_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_64f_convert_32f_u_avx512f(float* outputVector,
                                                  const double* inputVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int oneSixteenthPoints = num_points / 16;

    const double* inputVectorPtr = (const double*)inputVector;
    float* outputVectorPtr = outputVector;
    __m256 ret1, ret2;
    __m512d inputVal1, inputVal2;

    for (; number < oneSixteenthPoints; number++) {
        inputVal1 = _mm512_loadu_pd(inputVectorPtr);
        inputVectorPtr += 8;
        inputVal2 = _mm512_loadu_pd(inputVectorPtr);
        inputVectorPtr += 8;

        ret1 = _mm512_cvtpd_ps(inputVal1);
        ret2 = _mm512_cvtpd_ps(inputVal2);

        _mm256_storeu_ps(outputVectorPtr, ret1);
        outputVectorPtr += 8;

        _mm256_storeu_ps(outputVectorPtr, ret2);
        outputVectorPtr += 8;
    }

    number = oneSixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]);
    }
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_64f_convert_32f_u_avx(float* outputVector,
                                              const double* inputVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int oneEightPoints = num_points / 8;

    const double* inputVectorPtr = (const double*)inputVector;
    float* outputVectorPtr = outputVector;
    __m128 ret1, ret2;
    __m256d inputVal1, inputVal2;

    for (; number < oneEightPoints; number++) {
        inputVal1 = _mm256_loadu_pd(inputVectorPtr);
        inputVectorPtr += 4;
        inputVal2 = _mm256_loadu_pd(inputVectorPtr);
        inputVectorPtr += 4;

        ret1 = _mm256_cvtpd_ps(inputVal1);
        ret2 = _mm256_cvtpd_ps(inputVal2);

        _mm_storeu_ps(outputVectorPtr, ret1);
        outputVectorPtr += 4;

        _mm_storeu_ps(outputVectorPtr, ret2);
        outputVectorPtr += 4;
    }

    number = oneEightPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]);
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_64f_convert_32f_u_sse2(float* outputVector,
                                               const double* inputVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const double* inputVectorPtr = (const double*)inputVector;
    float* outputVectorPtr = outputVector;
    __m128 ret, ret2;
    __m128d inputVal1, inputVal2;

    for (; number < quarterPoints; number++) {
        inputVal1 = _mm_loadu_pd(inputVectorPtr);
        inputVectorPtr += 2;
        inputVal2 = _mm_loadu_pd(inputVectorPtr);
        inputVectorPtr += 2;

        ret = _mm_cvtpd_ps(inputVal1);
        ret2 = _mm_cvtpd_ps(inputVal2);

        ret = _mm_movelh_ps(ret, ret2);

        _mm_storeu_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]);
    }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void volk_64f_convert_32f_generic(float* outputVector,
                                                const double* inputVector,
                                                unsigned int num_points)
{
    float* outputVectorPtr = outputVector;
    const double* inputVectorPtr = inputVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *outputVectorPtr++ = ((float)(*inputVectorPtr++));
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_64f_convert_32f_u_H */
#ifndef INCLUDED_volk_64f_convert_32f_a_H
#define INCLUDED_volk_64f_convert_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_64f_convert_32f_a_avx512f(float* outputVector,
                                                  const double* inputVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int oneSixteenthPoints = num_points / 16;

    const double* inputVectorPtr = (const double*)inputVector;
    float* outputVectorPtr = outputVector;
    __m256 ret1, ret2;
    __m512d inputVal1, inputVal2;

    for (; number < oneSixteenthPoints; number++) {
        inputVal1 = _mm512_load_pd(inputVectorPtr);
        inputVectorPtr += 8;
        inputVal2 = _mm512_load_pd(inputVectorPtr);
        inputVectorPtr += 8;

        ret1 = _mm512_cvtpd_ps(inputVal1);
        ret2 = _mm512_cvtpd_ps(inputVal2);

        _mm256_store_ps(outputVectorPtr, ret1);
        outputVectorPtr += 8;

        _mm256_store_ps(outputVectorPtr, ret2);
        outputVectorPtr += 8;
    }

    number = oneSixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]);
    }
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_64f_convert_32f_a_avx(float* outputVector,
                                              const double* inputVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int oneEightPoints = num_points / 8;

    const double* inputVectorPtr = (const double*)inputVector;
    float* outputVectorPtr = outputVector;
    __m128 ret1, ret2;
    __m256d inputVal1, inputVal2;

    for (; number < oneEightPoints; number++) {
        inputVal1 = _mm256_load_pd(inputVectorPtr);
        inputVectorPtr += 4;
        inputVal2 = _mm256_load_pd(inputVectorPtr);
        inputVectorPtr += 4;

        ret1 = _mm256_cvtpd_ps(inputVal1);
        ret2 = _mm256_cvtpd_ps(inputVal2);

        _mm_store_ps(outputVectorPtr, ret1);
        outputVectorPtr += 4;

        _mm_store_ps(outputVectorPtr, ret2);
        outputVectorPtr += 4;
    }

    number = oneEightPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]);
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_64f_convert_32f_a_sse2(float* outputVector,
                                               const double* inputVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const double* inputVectorPtr = (const double*)inputVector;
    float* outputVectorPtr = outputVector;
    __m128 ret, ret2;
    __m128d inputVal1, inputVal2;

    for (; number < quarterPoints; number++) {
        inputVal1 = _mm_load_pd(inputVectorPtr);
        inputVectorPtr += 2;
        inputVal2 = _mm_load_pd(inputVectorPtr);
        inputVectorPtr += 2;

        ret = _mm_cvtpd_ps(inputVal1);
        ret2 = _mm_cvtpd_ps(inputVal2);

        ret = _mm_movelh_ps(ret, ret2);

        _mm_store_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]);
    }
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_64f_convert_32f_neonv8(float* outputVector,
                                               const double* inputVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    const double* inputPtr = inputVector;
    float* outputPtr = outputVector;

    for (; number < eighth_points; number++) {
        float64x2_t in0 = vld1q_f64(inputPtr);
        float64x2_t in1 = vld1q_f64(inputPtr + 2);
        float64x2_t in2 = vld1q_f64(inputPtr + 4);
        float64x2_t in3 = vld1q_f64(inputPtr + 6);
        __VOLK_PREFETCH(inputPtr + 8);

        float32x2_t out0 = vcvt_f32_f64(in0);
        float32x2_t out1 = vcvt_f32_f64(in1);
        float32x2_t out2 = vcvt_f32_f64(in2);
        float32x2_t out3 = vcvt_f32_f64(in3);

        vst1q_f32(outputPtr, vcombine_f32(out0, out1));
        vst1q_f32(outputPtr + 4, vcombine_f32(out2, out3));

        inputPtr += 8;
        outputPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *outputPtr++ = (float)(*inputPtr++);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_64f_convert_32f_rvv(float* outputVector,
                                            const double* inputVector,
                                            unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vfloat64m8_t v = __riscv_vle64_v_f64m8(inputVector, vl);
        __riscv_vse32(outputVector, __riscv_vfncvt_f(v, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_64f_convert_32f_a_H */
