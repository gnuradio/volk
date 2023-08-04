/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_64f_x2_dot_prod_64f
 *
 * \b Overview
 *
 * This block computes the dot product (or inner product) between two
 * vectors, the \p input and \p taps vectors. Given a set of \p
 * num_points taps, the result is the sum of products between the two
 * vectors. The result is a single value stored in the \p result
 * address and is returned as a double.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_64f_x2_dot_prod_64f(double* result, const double* input, const double* taps,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li input: vector of doubles.
 * \li taps:  double taps.
 * \li num_points: number of samples in both \p input and \p taps.
 *
 * \b Outputs
 * \li result: pointer to a double value to hold the dot product result.
 *
 * \b Example
 * Take the dot product of an increasing vector and a vector of ones. The result is the
 * sum of integers (0,9). \code int N = 10; unsigned int alignment = volk_get_alignment();
 *   double* increasing = (double*)volk_malloc(sizeof(double)*N, alignment);
 *   double* ones = (double*)volk_malloc(sizeof(double)*N, alignment);
 *   double* out = (double*)volk_malloc(sizeof(double)*1, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (double)ii;
 *       ones[ii] = 1.f;
 *   }
 *
 *   volk_64f_x2_dot_prod_64f(out, increasing, ones, N);
 *
 *   printf("out = %1.2lf\n", *out);
 *
 *   volk_free(increasing);
 *   volk_free(ones);
 *   volk_free(out);
 *
 *   return 0;
 * \endcode
 */

#ifndef INCLUDED_volk_64f_x2_dot_prod_64f_u_H
#define INCLUDED_volk_64f_x2_dot_prod_64f_u_H

#include <stdio.h>
#include <volk/volk_common.h>


#ifdef LV_HAVE_GENERIC


static inline void volk_64f_x2_dot_prod_64f_generic(double* result,
                                                    const double* input,
                                                    const double* taps,
                                                    unsigned int num_points)
{

    double dotProduct = 0;
    const double* aPtr = input;
    const double* bPtr = taps;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_SSE


static inline void volk_64f_x2_dot_prod_64f_u_sse(double* result,
                                                  const double* input,
                                                  const double* taps,
                                                  unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    double dotProduct = 0;
    const double* aPtr = input;
    const double* bPtr = taps;

    __m128d a0Val, a1Val, a2Val, a3Val;
    __m128d b0Val, b1Val, b2Val, b3Val;
    __m128d c0Val, c1Val, c2Val, c3Val;

    __m128d dotProdVal0 = _mm_setzero_pd();
    __m128d dotProdVal1 = _mm_setzero_pd();
    __m128d dotProdVal2 = _mm_setzero_pd();
    __m128d dotProdVal3 = _mm_setzero_pd();

    for (; number < eighthPoints; number++) {

        a0Val = _mm_loadu_pd(aPtr);
        a1Val = _mm_loadu_pd(aPtr + 2);
        a2Val = _mm_loadu_pd(aPtr + 4);
        a3Val = _mm_loadu_pd(aPtr + 6);
        b0Val = _mm_loadu_pd(bPtr);
        b1Val = _mm_loadu_pd(bPtr + 2);
        b2Val = _mm_loadu_pd(bPtr + 4);
        b3Val = _mm_loadu_pd(bPtr + 6);

        c0Val = _mm_mul_pd(a0Val, b0Val);
        c1Val = _mm_mul_pd(a1Val, b1Val);
        c2Val = _mm_mul_pd(a2Val, b2Val);
        c3Val = _mm_mul_pd(a3Val, b3Val);

        dotProdVal0 = _mm_add_pd(dotProdVal0, c0Val);
        dotProdVal1 = _mm_add_pd(dotProdVal1, c1Val);
        dotProdVal2 = _mm_add_pd(dotProdVal2, c2Val);
        dotProdVal3 = _mm_add_pd(dotProdVal3, c3Val);

        aPtr += 8;
        bPtr += 8;
    }

    dotProdVal0 = _mm_add_pd(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_pd(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_pd(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) double dotProductVector[2];
    _mm_store_pd(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_SSE*/

#ifdef LV_HAVE_SSE3

#include <pmmintrin.h>

static inline void volk_64f_x2_dot_prod_64f_u_sse3(double* result,
                                                   const double* input,
                                                   const double* taps,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    double dotProduct = 0;
    const double* aPtr = input;
    const double* bPtr = taps;

    __m128d a0Val, a1Val, a2Val, a3Val;
    __m128d b0Val, b1Val, b2Val, b3Val;
    __m128d c0Val, c1Val, c2Val, c3Val;

    __m128d dotProdVal0 = _mm_setzero_pd();
    __m128d dotProdVal1 = _mm_setzero_pd();
    __m128d dotProdVal2 = _mm_setzero_pd();
    __m128d dotProdVal3 = _mm_setzero_pd();

    for (; number < eighthPoints; number++) {

        a0Val = _mm_loadu_pd(aPtr);
        a1Val = _mm_loadu_pd(aPtr + 2);
        a2Val = _mm_loadu_pd(aPtr + 4);
        a3Val = _mm_loadu_pd(aPtr + 6);
        b0Val = _mm_loadu_pd(bPtr);
        b1Val = _mm_loadu_pd(bPtr + 2);
        b2Val = _mm_loadu_pd(bPtr + 4);
        b3Val = _mm_loadu_pd(bPtr + 6);

        c0Val = _mm_mul_pd(a0Val, b0Val);
        c1Val = _mm_mul_pd(a1Val, b1Val);
        c2Val = _mm_mul_pd(a2Val, b2Val);
        c3Val = _mm_mul_pd(a3Val, b3Val);

        dotProdVal0 = _mm_add_pd(dotProdVal0, c0Val);
        dotProdVal1 = _mm_add_pd(dotProdVal1, c1Val);
        dotProdVal2 = _mm_add_pd(dotProdVal2, c2Val);
        dotProdVal3 = _mm_add_pd(dotProdVal3, c3Val);

        aPtr += 8;
        bPtr += 8;
    }

    dotProdVal0 = _mm_add_pd(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_pd(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_pd(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) double dotProductVector[2];
    _mm_store_pd(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_SSE3*/

#ifdef LV_HAVE_SSE4_1

#include <smmintrin.h>

static inline void volk_64f_x2_dot_prod_64f_u_sse4_1(double* result,
                                                     const double* input,
                                                     const double* taps,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    double dotProduct = 0;
    const double* aPtr = input;
    const double* bPtr = taps;

    __m128d aVal1, bVal1, cVal1;
    __m128d aVal2, bVal2, cVal2;
    __m128d aVal3, bVal3, cVal3;
    __m128d aVal4, bVal4, cVal4;

    __m128d dotProdVal = _mm_setzero_pd();

    for (; number < eighthPoints; number++) {

        aVal1 = _mm_loadu_pd(aPtr);
        aPtr += 2;
        aVal2 = _mm_loadu_pd(aPtr);
        aPtr += 2;
        aVal3 = _mm_loadu_pd(aPtr);
        aPtr += 2;
        aVal4 = _mm_loadu_pd(aPtr);
        aPtr += 2;

        bVal1 = _mm_loadu_pd(bPtr);
        bPtr += 2;
        bVal2 = _mm_loadu_pd(bPtr);
        bPtr += 2;
        bVal3 = _mm_loadu_pd(bPtr);
        bPtr += 2;
        bVal4 = _mm_loadu_pd(bPtr);
        bPtr += 2;

        cVal1 = _mm_dp_pd(aVal1, bVal1, 0xF1); /* [(a[0-1] x b[0-1]), 0] */
        cVal2 = _mm_dp_pd(aVal2, bVal2, 0xF2); /* [0, (a[2-3] x b[2-3])] */
        cVal3 = _mm_dp_pd(aVal3, bVal3, 0xF1); /* [(a[4-5] x b[4-5]), 0] */
        cVal4 = _mm_dp_pd(aVal4, bVal4, 0xF2); /* [6, (a[6-7] x b[6-7])] */

        cVal1 = _mm_or_pd(cVal1, cVal2); /* [(a[0-1] x b[0-1]), (a[2-3] x b[2-3])] */
        cVal3 = _mm_or_pd(cVal3, cVal4); /* [(a[4-5] x b[4-5]), (a[6-7] x b[6-7])] */

        dotProdVal = _mm_add_pd(dotProdVal, cVal1);
        dotProdVal = _mm_add_pd(dotProdVal, cVal3);
    }

    __VOLK_ATTR_ALIGNED(16) double dotProductVector[2];
    _mm_store_pd(dotProductVector,
                 dotProdVal); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_SSE4_1*/

#ifdef LV_HAVE_AVX

#include <immintrin.h>

static inline void volk_64f_x2_dot_prod_64f_u_avx(double* result,
                                                  const double* input,
                                                  const double* taps,
                                                  unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    double dotProduct = 0;
    const double* aPtr = input;
    const double* bPtr = taps;

    __m256d a0Val, a1Val;
    __m256d b0Val, b1Val;
    __m256d c0Val, c1Val;

    __m256d dotProdVal0 = _mm256_setzero_pd();
    __m256d dotProdVal1 = _mm256_setzero_pd();

    for (; number < eighthPoints; number++) {

        a0Val = _mm256_loadu_pd(aPtr);
        a1Val = _mm256_loadu_pd(aPtr + 4);
        b0Val = _mm256_loadu_pd(bPtr);
        b1Val = _mm256_loadu_pd(bPtr + 4);

        c0Val = _mm256_mul_pd(a0Val, b0Val);
        c1Val = _mm256_mul_pd(a1Val, b1Val);

        dotProdVal0 = _mm256_add_pd(c0Val, dotProdVal0);
        dotProdVal1 = _mm256_add_pd(c1Val, dotProdVal1);

        aPtr += 8;
        bPtr += 8;
    }

    dotProdVal0 = _mm256_add_pd(dotProdVal0, dotProdVal1);

    __VOLK_ATTR_ALIGNED(32) double dotProductVector[4];

    _mm256_storeu_pd(dotProductVector,
                     dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];
    dotProduct += dotProductVector[2];
    dotProduct += dotProductVector[3];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_AVX*/

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
static inline void volk_64f_x2_dot_prod_64f_u_avx2_fma(double* result,
                                                       const double* input,
                                                       const double* taps,
                                                       unsigned int num_points)
{
    unsigned int number;
    const unsigned int fourthPoints = num_points / 4;

    const double* aPtr = input;
    const double* bPtr = taps;

    __m256d dotProdVal = _mm256_setzero_pd();
    __m256d aVal1, bVal1;

    for (number = 0; number < fourthPoints; number++) {

        aVal1 = _mm256_loadu_pd(aPtr);
        bVal1 = _mm256_loadu_pd(bPtr);
        aPtr += 4;
        bPtr += 4;

        dotProdVal = _mm256_fmadd_pd(aVal1, bVal1, dotProdVal);
    }

    __VOLK_ATTR_ALIGNED(32) double dotProductVector[4];
    _mm256_storeu_pd(dotProductVector,
                     dotProdVal); // Store the results back into the dot product vector

    double dotProduct = dotProductVector[0] + dotProductVector[1] + dotProductVector[2] +
                       dotProductVector[3];

    for (number = fourthPoints * 4; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#if LV_HAVE_AVX512F
#include <immintrin.h>
static inline void volk_64f_x2_dot_prod_64f_u_avx512f(double* result,
                                                      const double* input,
                                                      const double* taps,
                                                      unsigned int num_points)
{
    unsigned int number;
    const unsigned int eighthPoints = num_points / 8;

    const double* aPtr = input;
    const double* bPtr = taps;

    __m512d dotProdVal = _mm512_setzero_pd();
    __m512d aVal1, bVal1;

    for (number = 0; number < eighthPoints; number++) {

        aVal1 = _mm512_loadu_pd(aPtr);
        bVal1 = _mm512_loadu_pd(bPtr);
        aPtr += 8;
        bPtr += 8;

        dotProdVal = _mm512_fmadd_pd(aVal1, bVal1, dotProdVal);
    }

    __VOLK_ATTR_ALIGNED(64) double dotProductVector[8];
    _mm512_storeu_pd(dotProductVector,
                     dotProdVal); // Store the results back into the dot product vector

    double dotProduct = dotProductVector[0] + dotProductVector[1] + dotProductVector[2] +
                       dotProductVector[3] + dotProductVector[4] + dotProductVector[5] +
                       dotProductVector[6] + dotProductVector[7];

    for (number = eighthPoints * 8; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}
#endif /* LV_HAVE_AVX512F */

#endif /*INCLUDED_volk_64f_x2_dot_prod_64f_u_H*/

#ifndef INCLUDED_volk_64f_x2_dot_prod_64f_a_H
#define INCLUDED_volk_64f_x2_dot_prod_64f_a_H

#include <stdio.h>
#include <volk/volk_common.h>


#ifdef LV_HAVE_GENERIC


static inline void volk_64f_x2_dot_prod_64f_a_generic(double* result,
                                                      const double* input,
                                                      const double* taps,
                                                      unsigned int num_points)
{

    double dotProduct = 0;
    const double* aPtr = input;
    const double* bPtr = taps;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_SSE


static inline void volk_64f_x2_dot_prod_64f_a_sse(double* result,
                                                  const double* input,
                                                  const double* taps,
                                                  unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    double dotProduct = 0;
    const double* aPtr = input;
    const double* bPtr = taps;

    __m128d a0Val, a1Val, a2Val, a3Val;
    __m128d b0Val, b1Val, b2Val, b3Val;
    __m128d c0Val, c1Val, c2Val, c3Val;

    __m128d dotProdVal0 = _mm_setzero_pd();
    __m128d dotProdVal1 = _mm_setzero_pd();
    __m128d dotProdVal2 = _mm_setzero_pd();
    __m128d dotProdVal3 = _mm_setzero_pd();

    for (; number < eighthPoints; number++) {

        a0Val = _mm_load_pd(aPtr);
        a1Val = _mm_load_pd(aPtr + 2);
        a2Val = _mm_load_pd(aPtr + 4);
        a3Val = _mm_load_pd(aPtr + 6);
        b0Val = _mm_load_pd(bPtr);
        b1Val = _mm_load_pd(bPtr + 2);
        b2Val = _mm_load_pd(bPtr + 4);
        b3Val = _mm_load_pd(bPtr + 6);

        c0Val = _mm_mul_pd(a0Val, b0Val);
        c1Val = _mm_mul_pd(a1Val, b1Val);
        c2Val = _mm_mul_pd(a2Val, b2Val);
        c3Val = _mm_mul_pd(a3Val, b3Val);

        dotProdVal0 = _mm_add_pd(dotProdVal0, c0Val);
        dotProdVal1 = _mm_add_pd(dotProdVal1, c1Val);
        dotProdVal2 = _mm_add_pd(dotProdVal2, c2Val);
        dotProdVal3 = _mm_add_pd(dotProdVal3, c3Val);

        aPtr += 8;
        bPtr += 8;
    }

    dotProdVal0 = _mm_add_pd(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_pd(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_pd(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) double dotProductVector[2];
    _mm_store_pd(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_SSE*/

#ifdef LV_HAVE_SSE3

#include <pmmintrin.h>

static inline void volk_64f_x2_dot_prod_64f_a_sse3(double* result,
                                                   const double* input,
                                                   const double* taps,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    double dotProduct = 0;
    const double* aPtr = input;
    const double* bPtr = taps;

    __m128d a0Val, a1Val, a2Val, a3Val;
    __m128d b0Val, b1Val, b2Val, b3Val;
    __m128d c0Val, c1Val, c2Val, c3Val;

    __m128d dotProdVal0 = _mm_setzero_pd();
    __m128d dotProdVal1 = _mm_setzero_pd();
    __m128d dotProdVal2 = _mm_setzero_pd();
    __m128d dotProdVal3 = _mm_setzero_pd();

    for (; number < eighthPoints; number++) {

        a0Val = _mm_load_pd(aPtr);
        a1Val = _mm_load_pd(aPtr + 2);
        a2Val = _mm_load_pd(aPtr + 4);
        a3Val = _mm_load_pd(aPtr + 6);
        b0Val = _mm_load_pd(bPtr);
        b1Val = _mm_load_pd(bPtr + 2);
        b2Val = _mm_load_pd(bPtr + 4);
        b3Val = _mm_load_pd(bPtr + 6);

        c0Val = _mm_mul_pd(a0Val, b0Val);
        c1Val = _mm_mul_pd(a1Val, b1Val);
        c2Val = _mm_mul_pd(a2Val, b2Val);
        c3Val = _mm_mul_pd(a3Val, b3Val);

        dotProdVal0 = _mm_add_pd(dotProdVal0, c0Val);
        dotProdVal1 = _mm_add_pd(dotProdVal1, c1Val);
        dotProdVal2 = _mm_add_pd(dotProdVal2, c2Val);
        dotProdVal3 = _mm_add_pd(dotProdVal3, c3Val);

        aPtr += 8;
        bPtr += 8;
    }

    dotProdVal0 = _mm_add_pd(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_pd(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_pd(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) double dotProductVector[2];
    _mm_store_pd(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_SSE3*/

#ifdef LV_HAVE_SSE4_1

#include <smmintrin.h>

static inline void volk_64f_x2_dot_prod_64f_a_sse4_1(double* result,
                                                     const double* input,
                                                     const double* taps,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    double dotProduct = 0;
    const double* aPtr = input;
    const double* bPtr = taps;

    __m128d aVal1, bVal1, cVal1;
    __m128d aVal2, bVal2, cVal2;
    __m128d aVal3, bVal3, cVal3;
    __m128d aVal4, bVal4, cVal4;

    __m128d dotProdVal = _mm_setzero_pd();

    for (; number < eighthPoints; number++) {

        aVal1 = _mm_load_pd(aPtr);
        aPtr += 2;
        aVal2 = _mm_load_pd(aPtr);
        aPtr += 2;
        aVal3 = _mm_load_pd(aPtr);
        aPtr += 2;
        aVal4 = _mm_load_pd(aPtr);
        aPtr += 2;

        bVal1 = _mm_load_pd(bPtr);
        bPtr += 2;
        bVal2 = _mm_load_pd(bPtr);
        bPtr += 2;
        bVal3 = _mm_load_pd(bPtr);
        bPtr += 2;
        bVal4 = _mm_load_pd(bPtr);
        bPtr += 2;

        cVal1 = _mm_dp_pd(aVal1, bVal1, 0xF1); /* [(a[0-1] x b[0-1]), 0] */
        cVal2 = _mm_dp_pd(aVal2, bVal2, 0xF2); /* [0, (a[2-3] x b[2-3])] */
        cVal3 = _mm_dp_pd(aVal3, bVal3, 0xF1); /* [(a[4-5] x b[4-5]), 0] */
        cVal4 = _mm_dp_pd(aVal4, bVal4, 0xF2); /* [6, (a[6-7] x b[6-7])] */

        cVal1 = _mm_or_pd(cVal1, cVal2); /* [(a[0-1] x b[0-1]), (a[2-3] x b[2-3])] */
        cVal3 = _mm_or_pd(cVal3, cVal4); /* [(a[4-5] x b[4-5]), (a[6-7] x b[6-7])] */

        dotProdVal = _mm_add_pd(dotProdVal, cVal1);
        dotProdVal = _mm_add_pd(dotProdVal, cVal3);
    }

    __VOLK_ATTR_ALIGNED(16) double dotProductVector[2];
    _mm_store_pd(dotProductVector,
                 dotProdVal); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_SSE4_1*/

#ifdef LV_HAVE_AVX

#include <immintrin.h>

static inline void volk_64f_x2_dot_prod_64f_a_avx(double* result,
                                                  const double* input,
                                                  const double* taps,
                                                  unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    double dotProduct = 0;
    const double* aPtr = input;
    const double* bPtr = taps;

    __m256d a0Val, a1Val;
    __m256d b0Val, b1Val;
    __m256d c0Val, c1Val;

    __m256d dotProdVal0 = _mm256_setzero_pd();
    __m256d dotProdVal1 = _mm256_setzero_pd();

    for (; number < eighthPoints; number++) {

        a0Val = _mm256_load_pd(aPtr);
        a1Val = _mm256_load_pd(aPtr + 4);
        b0Val = _mm256_load_pd(bPtr);
        b1Val = _mm256_load_pd(bPtr + 4);

        c0Val = _mm256_mul_pd(a0Val, b0Val);
        c1Val = _mm256_mul_pd(a1Val, b1Val);

        dotProdVal0 = _mm256_add_pd(c0Val, dotProdVal0);
        dotProdVal1 = _mm256_add_pd(c1Val, dotProdVal1);

        aPtr += 8;
        bPtr += 8;
    }

    dotProdVal0 = _mm256_add_pd(dotProdVal0, dotProdVal1);

    __VOLK_ATTR_ALIGNED(32) double dotProductVector[4];

    _mm256_store_pd(dotProductVector,
                     dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];
    dotProduct += dotProductVector[2];
    dotProduct += dotProductVector[3];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}
#endif /*LV_HAVE_AVX*/


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
static inline void volk_64f_x2_dot_prod_64f_a_avx2_fma(double* result,
                                                       const double* input,
                                                       const double* taps,
                                                       unsigned int num_points)
{
    unsigned int number;
    const unsigned int fourthPoints = num_points / 4;

    const double* aPtr = input;
    const double* bPtr = taps;

    __m256d dotProdVal = _mm256_setzero_pd();
    __m256d aVal1, bVal1;

    for (number = 0; number < fourthPoints; number++) {

        aVal1 = _mm256_load_pd(aPtr);
        bVal1 = _mm256_load_pd(bPtr);
        aPtr += 4;
        bPtr += 4;

        dotProdVal = _mm256_fmadd_pd(aVal1, bVal1, dotProdVal);
    }

    __VOLK_ATTR_ALIGNED(32) double dotProductVector[4];
    _mm256_store_pd(dotProductVector,
                     dotProdVal); // Store the results back into the dot product vector

    double dotProduct = dotProductVector[0] + dotProductVector[1] + dotProductVector[2] +
                       dotProductVector[3];

    for (number = fourthPoints * 4; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#if LV_HAVE_AVX512F
#include <immintrin.h>
static inline void volk_64f_x2_dot_prod_64f_a_avx512f(double* result,
                                                      const double* input,
                                                      const double* taps,
                                                      unsigned int num_points)
{
    unsigned int number;
    const unsigned int eighthPoints = num_points / 8;

    const double* aPtr = input;
    const double* bPtr = taps;

    __m512d dotProdVal = _mm512_setzero_pd();
    __m512d aVal1, bVal1;

    for (number = 0; number < eighthPoints; number++) {

        aVal1 = _mm512_load_pd(aPtr);
        bVal1 = _mm512_load_pd(bPtr);
        aPtr += 8;
        bPtr += 8;

        dotProdVal = _mm512_fmadd_pd(aVal1, bVal1, dotProdVal);
    }

    __VOLK_ATTR_ALIGNED(64) double dotProductVector[8];
    _mm512_store_pd(dotProductVector,
                     dotProdVal); // Store the results back into the dot product vector

    double dotProduct = dotProductVector[0] + dotProductVector[1] + dotProductVector[2] +
                       dotProductVector[3] + dotProductVector[4] + dotProductVector[5] +
                       dotProductVector[6] + dotProductVector[7];

    for (number = eighthPoints * 8; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}
#endif /* LV_HAVE_AVX512F */

#endif /*INCLUDED_volk_64f_x2_dot_prod_64f_a_H*/
