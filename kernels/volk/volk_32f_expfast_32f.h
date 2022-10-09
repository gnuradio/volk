/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_expfast_32f
 *
 * \b Overview
 *
 * Computes exp of input vector and stores results in output
 * vector. This uses a fast exp approximation with a maximum 7% error.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_expfast_32f(float* bVector, const float* aVector, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li aVector: Input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li bVector: The output vector.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       in[ii] = std::log((float)ii);
 *   }
 *
 *   volk_32f_expfast_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#define Mln2 0.6931471805f
#define A 8388608.0f
#define B 1065353216.0f
#define C 60801.0f


#ifndef INCLUDED_volk_32f_expfast_32f_a_H
#define INCLUDED_volk_32f_expfast_32f_a_H

#if LV_HAVE_AVX && LV_HAVE_FMA

#include <immintrin.h>

static inline void volk_32f_expfast_32f_a_avx_fma(float* bVector,
                                                  const float* aVector,
                                                  unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    __m256 aVal, bVal, a, b;
    __m256i exp;
    a = _mm256_set1_ps(A / Mln2);
    b = _mm256_set1_ps(B - C);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        exp = _mm256_cvtps_epi32(_mm256_fmadd_ps(a, aVal, b));
        bVal = _mm256_castsi256_ps(exp);

        _mm256_store_ps(bPtr, bVal);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA for aligned */

#ifdef LV_HAVE_AVX

#include <immintrin.h>

static inline void
volk_32f_expfast_32f_a_avx(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    __m256 aVal, bVal, a, b;
    __m256i exp;
    a = _mm256_set1_ps(A / Mln2);
    b = _mm256_set1_ps(B - C);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        exp = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(a, aVal), b));
        bVal = _mm256_castsi256_ps(exp);

        _mm256_store_ps(bPtr, bVal);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX for aligned */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_32f_expfast_32f_a_sse4_1(float* bVector,
                                                 const float* aVector,
                                                 unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m128 aVal, bVal, a, b;
    __m128i exp;
    a = _mm_set1_ps(A / Mln2);
    b = _mm_set1_ps(B - C);

    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);
        exp = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(a, aVal), b));
        bVal = _mm_castsi128_ps(exp);

        _mm_store_ps(bPtr, bVal);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for aligned */

#endif /* INCLUDED_volk_32f_expfast_32f_a_H */

#ifndef INCLUDED_volk_32f_expfast_32f_u_H
#define INCLUDED_volk_32f_expfast_32f_u_H

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_expfast_32f_u_avx_fma(float* bVector,
                                                  const float* aVector,
                                                  unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    __m256 aVal, bVal, a, b;
    __m256i exp;
    a = _mm256_set1_ps(A / Mln2);
    b = _mm256_set1_ps(B - C);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        exp = _mm256_cvtps_epi32(_mm256_fmadd_ps(a, aVal, b));
        bVal = _mm256_castsi256_ps(exp);

        _mm256_storeu_ps(bPtr, bVal);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA for unaligned */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_expfast_32f_u_avx(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    __m256 aVal, bVal, a, b;
    __m256i exp;
    a = _mm256_set1_ps(A / Mln2);
    b = _mm256_set1_ps(B - C);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        exp = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(a, aVal), b));
        bVal = _mm256_castsi256_ps(exp);

        _mm256_storeu_ps(bPtr, bVal);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX for unaligned */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_32f_expfast_32f_u_sse4_1(float* bVector,
                                                 const float* aVector,
                                                 unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m128 aVal, bVal, a, b;
    __m128i exp;
    a = _mm_set1_ps(A / Mln2);
    b = _mm_set1_ps(B - C);

    for (; number < quarterPoints; number++) {
        aVal = _mm_loadu_ps(aPtr);
        exp = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(a, aVal), b));
        bVal = _mm_castsi128_ps(exp);

        _mm_storeu_ps(bPtr, bVal);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for unaligned */


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_expfast_32f_generic(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32f_expfast_32f_u_H */
