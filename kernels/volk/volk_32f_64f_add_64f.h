/* -*- c++ -*- */
/*
 * Copyright 2018 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_64f_add_64f
 *
 * \b Overview
 *
 * Adds two input vectors and store result as a double-precision vectors. One
 * of the input vector is defined as a single precision floating point, so
 * upcasting is performed before the addition
 *
 * c[i] = a[i] + b[i]
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_64f_add_64f(double* cVector, const double* aVector, const
 * double* bVector, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li aVector: First input vector.
 * \li bVector: Second input vector.
 * \li num_points: The number of values in both input vectors.
 *
 * \b Outputs
 * \li cVector: The output vector.
 *
 * \b Example
 * add elements of an increasing vector by those of a decreasing vector.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   double* decreasing = (double*)volk_malloc(sizeof(double)*N, alignment);
 *   double* out = (double*)volk_malloc(sizeof(double)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (double)ii;
 *       decreasing[ii] = 10.f - (double)ii;
 *   }
 *
 *   volk_32f_64f_add_64f(out, increasing, decreasing, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %1.2F\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(decreasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_64f_add_64f_H
#define INCLUDED_volk_32f_64f_add_64f_H

#include <inttypes.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_64f_add_64f_generic(double* cVector,
                                                const float* aVector,
                                                const double* bVector,
                                                unsigned int num_points)
{
    double* cPtr = cVector;
    const float* aPtr = aVector;
    const double* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = ((double)(*aPtr++)) + (*bPtr++);
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_64f_add_64f_neon(double* cVector,
                                             const float* aVector,
                                             const double* bVector,
                                             unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int half_points = num_points / 2;

    double* cPtr = cVector;
    const float* aPtr = aVector;
    const double* bPtr = bVector;

    float64x2_t aVal, bVal, cVal;
    float32x2_t aVal1;
    for (number = 0; number < half_points; number++) {
        // Load in to NEON registers
        aVal1 = vld1_f32(aPtr);
        bVal = vld1q_f64(bPtr);
        __VOLK_PREFETCH(aPtr + 2);
        __VOLK_PREFETCH(bPtr + 2);
        aPtr += 2; // q uses quadwords, 4 floats per vadd
        bPtr += 2;

        // Vector conversion
        aVal = vcvt_f64_f32(aVal1);
        // vector add
        cVal = vaddq_f64(aVal, bVal);
        // Store the results back into the C container
        vst1q_f64(cPtr, cVal);

        cPtr += 2;
    }

    number = half_points * 2; // should be = num_points
    for (; number < num_points; number++) {
        *cPtr++ = ((double)(*aPtr++)) + (*bPtr++);
    }
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_AVX

#include <immintrin.h>
#include <xmmintrin.h>

static inline void volk_32f_64f_add_64f_u_avx(double* cVector,
                                              const float* aVector,
                                              const double* bVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    double* cPtr = cVector;
    const float* aPtr = aVector;
    const double* bPtr = bVector;

    __m256 aVal;
    __m128 aVal1, aVal2;
    __m256d aDbl1, aDbl2, bVal1, bVal2, cVal1, cVal2;
    for (; number < eighth_points; number++) {

        aVal = _mm256_loadu_ps(aPtr);
        bVal1 = _mm256_loadu_pd(bPtr);
        bVal2 = _mm256_loadu_pd(bPtr + 4);

        aVal1 = _mm256_extractf128_ps(aVal, 0);
        aVal2 = _mm256_extractf128_ps(aVal, 1);

        aDbl1 = _mm256_cvtps_pd(aVal1);
        aDbl2 = _mm256_cvtps_pd(aVal2);

        cVal1 = _mm256_add_pd(aDbl1, bVal1);
        cVal2 = _mm256_add_pd(aDbl2, bVal2);

        _mm256_storeu_pd(cPtr,
                         cVal1); // Store the results back into the C container
        _mm256_storeu_pd(cPtr + 4,
                         cVal2); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *cPtr++ = ((double)(*aPtr++)) + (*bPtr++);
    }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX

#include <immintrin.h>
#include <xmmintrin.h>

static inline void volk_32f_64f_add_64f_a_avx(double* cVector,
                                              const float* aVector,
                                              const double* bVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    double* cPtr = cVector;
    const float* aPtr = aVector;
    const double* bPtr = bVector;

    __m256 aVal;
    __m128 aVal1, aVal2;
    __m256d aDbl1, aDbl2, bVal1, bVal2, cVal1, cVal2;
    for (; number < eighth_points; number++) {

        aVal = _mm256_load_ps(aPtr);
        bVal1 = _mm256_load_pd(bPtr);
        bVal2 = _mm256_load_pd(bPtr + 4);

        aVal1 = _mm256_extractf128_ps(aVal, 0);
        aVal2 = _mm256_extractf128_ps(aVal, 1);

        aDbl1 = _mm256_cvtps_pd(aVal1);
        aDbl2 = _mm256_cvtps_pd(aVal2);

        cVal1 = _mm256_add_pd(aDbl1, bVal1);
        cVal2 = _mm256_add_pd(aDbl2, bVal2);

        _mm256_store_pd(cPtr, cVal1); // Store the results back into the C container
        _mm256_store_pd(cPtr + 4,
                        cVal2); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *cPtr++ = ((double)(*aPtr++)) + (*bPtr++);
    }
}

#endif /* LV_HAVE_AVX */

#endif /* INCLUDED_volk_32f_64f_add_64f_u_H */
