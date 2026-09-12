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

#ifdef LV_HAVE_GENERIC

static inline void volk_64f_convert_32f_generic(float* outputVector,
                                                const double* inputVector,
                                                unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; ++number) {
        *outputVector++ = (float)*inputVector++;
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_64f_convert_32f_u_avx512f(float* outputVector,
                                                  const double* inputVector,
                                                  unsigned int num_points)
{
    const unsigned int oneSixteenthPoints = num_points / 16;
    for (unsigned int number = 0; number < oneSixteenthPoints; ++number) {
        const __m512d inputVal1 = _mm512_loadu_pd(inputVector);
        const __m512d inputVal2 = _mm512_loadu_pd(inputVector + 8);
        _mm256_storeu_ps(outputVector, _mm512_cvtpd_ps(inputVal1));
        _mm256_storeu_ps(outputVector + 8, _mm512_cvtpd_ps(inputVal2));
        inputVector += 16;
        outputVector += 16;
    }

    volk_64f_convert_32f_generic(
        outputVector, inputVector, num_points - oneSixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_64f_convert_32f_u_avx(float* outputVector,
                                              const double* inputVector,
                                              unsigned int num_points)
{
    const unsigned int oneEightPoints = num_points / 8;
    for (unsigned int number = 0; number < oneEightPoints; ++number) {
        const __m256d inputVal1 = _mm256_loadu_pd(inputVector);
        const __m256d inputVal2 = _mm256_loadu_pd(inputVector + 4);
        _mm_storeu_ps(outputVector, _mm256_cvtpd_ps(inputVal1));
        _mm_storeu_ps(outputVector + 4, _mm256_cvtpd_ps(inputVal2));
        inputVector += 8;
        outputVector += 8;
    }

    volk_64f_convert_32f_generic(
        outputVector, inputVector, num_points - oneEightPoints * 8);
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_64f_convert_32f_u_sse2(float* outputVector,
                                               const double* inputVector,
                                               unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;
    for (unsigned int number = 0; number < quarterPoints; ++number) {
        const __m128d inputVal1 = _mm_loadu_pd(inputVector);
        const __m128d inputVal2 = _mm_loadu_pd(inputVector + 2);
        const __m128 output =
            _mm_movelh_ps(_mm_cvtpd_ps(inputVal1), _mm_cvtpd_ps(inputVal2));
        _mm_storeu_ps(outputVector, output);
        inputVector += 4;
        outputVector += 4;
    }

    volk_64f_convert_32f_generic(
        outputVector, inputVector, num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE2 */


#endif /* INCLUDED_volk_64f_convert_32f_u_H */
#ifndef INCLUDED_volk_64f_convert_32f_a_H
#define INCLUDED_volk_64f_convert_32f_a_H

#include <inttypes.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_64f_convert_32f_a_avx512f(float* outputVector,
                                                  const double* inputVector,
                                                  unsigned int num_points)
{
    const unsigned int oneSixteenthPoints = num_points / 16;
    for (unsigned int number = 0; number < oneSixteenthPoints; ++number) {
        const __m512d inputVal1 = _mm512_load_pd(inputVector);
        const __m512d inputVal2 = _mm512_load_pd(inputVector + 8);
        _mm256_store_ps(outputVector, _mm512_cvtpd_ps(inputVal1));
        _mm256_store_ps(outputVector + 8, _mm512_cvtpd_ps(inputVal2));
        inputVector += 16;
        outputVector += 16;
    }

    volk_64f_convert_32f_generic(
        outputVector, inputVector, num_points - oneSixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_64f_convert_32f_a_avx(float* outputVector,
                                              const double* inputVector,
                                              unsigned int num_points)
{
    const unsigned int oneEightPoints = num_points / 8;
    for (unsigned int number = 0; number < oneEightPoints; ++number) {
        const __m256d inputVal1 = _mm256_load_pd(inputVector);
        const __m256d inputVal2 = _mm256_load_pd(inputVector + 4);
        _mm_store_ps(outputVector, _mm256_cvtpd_ps(inputVal1));
        _mm_store_ps(outputVector + 4, _mm256_cvtpd_ps(inputVal2));
        inputVector += 8;
        outputVector += 8;
    }

    volk_64f_convert_32f_generic(
        outputVector, inputVector, num_points - oneEightPoints * 8);
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_64f_convert_32f_a_sse2(float* outputVector,
                                               const double* inputVector,
                                               unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;
    for (unsigned int number = 0; number < quarterPoints; ++number) {
        const __m128d inputVal1 = _mm_load_pd(inputVector);
        const __m128d inputVal2 = _mm_load_pd(inputVector + 2);
        const __m128 output =
            _mm_movelh_ps(_mm_cvtpd_ps(inputVal1), _mm_cvtpd_ps(inputVal2));
        _mm_store_ps(outputVector, output);
        inputVector += 4;
        outputVector += 4;
    }

    volk_64f_convert_32f_generic(
        outputVector, inputVector, num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_64f_convert_32f_neonv8(float* outputVector,
                                               const double* inputVector,
                                               unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    for (unsigned int number = 0; number < eighth_points; ++number) {
        const float64x2_t input0 = vld1q_f64(inputVector);
        const float64x2_t input1 = vld1q_f64(inputVector + 2);
        const float64x2_t input2 = vld1q_f64(inputVector + 4);
        const float64x2_t input3 = vld1q_f64(inputVector + 6);
        __VOLK_PREFETCH(inputVector + 8);
        vst1q_f32(outputVector, vcombine_f32(vcvt_f32_f64(input0), vcvt_f32_f64(input1)));
        vst1q_f32(outputVector + 4,
                  vcombine_f32(vcvt_f32_f64(input2), vcvt_f32_f64(input3)));
        inputVector += 8;
        outputVector += 8;
    }

    volk_64f_convert_32f_generic(
        outputVector, inputVector, num_points - eighth_points * 8);
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
