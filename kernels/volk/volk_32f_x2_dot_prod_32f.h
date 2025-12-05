/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_x2_dot_prod_32f
 *
 * \b Overview
 *
 * This block computes the dot product (or inner product) between two
 * vectors, the \p input and \p taps vectors. Given a set of \p
 * num_points taps, the result is the sum of products between the two
 * vectors. The result is a single value stored in the \p result
 * address and is returned as a float.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_dot_prod_32f(float* result, const float* input, const float* taps,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li input: vector of floats.
 * \li taps:  float taps.
 * \li num_points: number of samples in both \p input and \p taps.
 *
 * \b Outputs
 * \li result: pointer to a float value to hold the dot product result.
 *
 * \b Example
 * Take the dot product of an increasing vector and a vector of ones. The result is the
 * sum of integers (0,9). \code int N = 10; unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* ones = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*1, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (float)ii;
 *       ones[ii] = 1.f;
 *   }
 *
 *   volk_32f_x2_dot_prod_32f(out, increasing, ones, N);
 *
 *   printf("out = %1.2f\n", *out);
 *
 *   volk_free(increasing);
 *   volk_free(ones);
 *   volk_free(out);
 *
 *   return 0;
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_dot_prod_32f_u_H
#define INCLUDED_volk_32f_x2_dot_prod_32f_u_H

#include <stdio.h>
#include <volk/volk_common.h>


#ifdef LV_HAVE_GENERIC


static inline void volk_32f_x2_dot_prod_32f_generic(float* result,
                                                    const float* input,
                                                    const float* taps,
                                                    unsigned int num_points)
{

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_SSE


static inline void volk_32f_x2_dot_prod_32f_u_sse(float* result,
                                                  const float* input,
                                                  const float* taps,
                                                  unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m128 a0Val, a1Val, a2Val, a3Val;
    __m128 b0Val, b1Val, b2Val, b3Val;
    __m128 c0Val, c1Val, c2Val, c3Val;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        a0Val = _mm_loadu_ps(aPtr);
        a1Val = _mm_loadu_ps(aPtr + 4);
        a2Val = _mm_loadu_ps(aPtr + 8);
        a3Val = _mm_loadu_ps(aPtr + 12);
        b0Val = _mm_loadu_ps(bPtr);
        b1Val = _mm_loadu_ps(bPtr + 4);
        b2Val = _mm_loadu_ps(bPtr + 8);
        b3Val = _mm_loadu_ps(bPtr + 12);

        c0Val = _mm_mul_ps(a0Val, b0Val);
        c1Val = _mm_mul_ps(a1Val, b1Val);
        c2Val = _mm_mul_ps(a2Val, b2Val);
        c3Val = _mm_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm_add_ps(c3Val, dotProdVal3);

        aPtr += 16;
        bPtr += 16;
    }

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];

    _mm_store_ps(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];
    dotProduct += dotProductVector[2];
    dotProduct += dotProductVector[3];

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_SSE*/

#ifdef LV_HAVE_SSE3

#include <pmmintrin.h>

static inline void volk_32f_x2_dot_prod_32f_u_sse3(float* result,
                                                   const float* input,
                                                   const float* taps,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m128 a0Val, a1Val, a2Val, a3Val;
    __m128 b0Val, b1Val, b2Val, b3Val;
    __m128 c0Val, c1Val, c2Val, c3Val;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        a0Val = _mm_loadu_ps(aPtr);
        a1Val = _mm_loadu_ps(aPtr + 4);
        a2Val = _mm_loadu_ps(aPtr + 8);
        a3Val = _mm_loadu_ps(aPtr + 12);
        b0Val = _mm_loadu_ps(bPtr);
        b1Val = _mm_loadu_ps(bPtr + 4);
        b2Val = _mm_loadu_ps(bPtr + 8);
        b3Val = _mm_loadu_ps(bPtr + 12);

        c0Val = _mm_mul_ps(a0Val, b0Val);
        c1Val = _mm_mul_ps(a1Val, b1Val);
        c2Val = _mm_mul_ps(a2Val, b2Val);
        c3Val = _mm_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm_add_ps(dotProdVal0, c0Val);
        dotProdVal1 = _mm_add_ps(dotProdVal1, c1Val);
        dotProdVal2 = _mm_add_ps(dotProdVal2, c2Val);
        dotProdVal3 = _mm_add_ps(dotProdVal3, c3Val);

        aPtr += 16;
        bPtr += 16;
    }

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];
    _mm_store_ps(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];
    dotProduct += dotProductVector[2];
    dotProduct += dotProductVector[3];

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_SSE3*/

#ifdef LV_HAVE_SSE4_1

#include <smmintrin.h>

static inline void volk_32f_x2_dot_prod_32f_u_sse4_1(float* result,
                                                     const float* input,
                                                     const float* taps,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m128 aVal1, bVal1, cVal1;
    __m128 aVal2, bVal2, cVal2;
    __m128 aVal3, bVal3, cVal3;
    __m128 aVal4, bVal4, cVal4;

    __m128 dotProdVal = _mm_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        aVal1 = _mm_loadu_ps(aPtr);
        aPtr += 4;
        aVal2 = _mm_loadu_ps(aPtr);
        aPtr += 4;
        aVal3 = _mm_loadu_ps(aPtr);
        aPtr += 4;
        aVal4 = _mm_loadu_ps(aPtr);
        aPtr += 4;

        bVal1 = _mm_loadu_ps(bPtr);
        bPtr += 4;
        bVal2 = _mm_loadu_ps(bPtr);
        bPtr += 4;
        bVal3 = _mm_loadu_ps(bPtr);
        bPtr += 4;
        bVal4 = _mm_loadu_ps(bPtr);
        bPtr += 4;

        cVal1 = _mm_dp_ps(aVal1, bVal1, 0xF1);
        cVal2 = _mm_dp_ps(aVal2, bVal2, 0xF2);
        cVal3 = _mm_dp_ps(aVal3, bVal3, 0xF4);
        cVal4 = _mm_dp_ps(aVal4, bVal4, 0xF8);

        cVal1 = _mm_or_ps(cVal1, cVal2);
        cVal3 = _mm_or_ps(cVal3, cVal4);
        cVal1 = _mm_or_ps(cVal1, cVal3);

        dotProdVal = _mm_add_ps(dotProdVal, cVal1);
    }

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];
    _mm_store_ps(dotProductVector,
                 dotProdVal); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];
    dotProduct += dotProductVector[2];
    dotProduct += dotProductVector[3];

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_SSE4_1*/

#ifdef LV_HAVE_AVX

#include <immintrin.h>

static inline void volk_32f_x2_dot_prod_32f_u_avx(float* result,
                                                  const float* input,
                                                  const float* taps,
                                                  unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m256 a0Val, a1Val;
    __m256 b0Val, b1Val;
    __m256 c0Val, c1Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        a0Val = _mm256_loadu_ps(aPtr);
        a1Val = _mm256_loadu_ps(aPtr + 8);
        b0Val = _mm256_loadu_ps(bPtr);
        b1Val = _mm256_loadu_ps(bPtr + 8);

        c0Val = _mm256_mul_ps(a0Val, b0Val);
        c1Val = _mm256_mul_ps(a1Val, b1Val);

        dotProdVal0 = _mm256_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm256_add_ps(c1Val, dotProdVal1);

        aPtr += 16;
        bPtr += 16;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_storeu_ps(dotProductVector,
                     dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];
    dotProduct += dotProductVector[2];
    dotProduct += dotProductVector[3];
    dotProduct += dotProductVector[4];
    dotProduct += dotProductVector[5];
    dotProduct += dotProductVector[6];
    dotProduct += dotProductVector[7];

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_AVX*/

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
static inline void volk_32f_x2_dot_prod_32f_u_avx2_fma(float* result,
                                                       const float* input,
                                                       const float* taps,
                                                       unsigned int num_points)
{
    unsigned int number;
    const unsigned int eighthPoints = num_points / 8;

    const float* aPtr = input;
    const float* bPtr = taps;

    __m256 dotProdVal = _mm256_setzero_ps();
    __m256 aVal1, bVal1;

    for (number = 0; number < eighthPoints; number++) {

        aVal1 = _mm256_loadu_ps(aPtr);
        bVal1 = _mm256_loadu_ps(bPtr);
        aPtr += 8;
        bPtr += 8;

        dotProdVal = _mm256_fmadd_ps(aVal1, bVal1, dotProdVal);
    }

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];
    _mm256_storeu_ps(dotProductVector,
                     dotProdVal); // Store the results back into the dot product vector

    float dotProduct = dotProductVector[0] + dotProductVector[1] + dotProductVector[2] +
                       dotProductVector[3] + dotProductVector[4] + dotProductVector[5] +
                       dotProductVector[6] + dotProductVector[7];

    for (number = eighthPoints * 8; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#if LV_HAVE_AVX512F
#include <immintrin.h>
static inline void volk_32f_x2_dot_prod_32f_u_avx512f(float* result,
                                                      const float* input,
                                                      const float* taps,
                                                      unsigned int num_points)
{
    unsigned int number;
    const unsigned int sixteenthPoints = num_points / 16;

    const float* aPtr = input;
    const float* bPtr = taps;

    __m512 dotProdVal = _mm512_setzero_ps();
    __m512 aVal1, bVal1;

    for (number = 0; number < sixteenthPoints; number++) {

        aVal1 = _mm512_loadu_ps(aPtr);
        bVal1 = _mm512_loadu_ps(bPtr);
        aPtr += 16;
        bPtr += 16;

        dotProdVal = _mm512_fmadd_ps(aVal1, bVal1, dotProdVal);
    }

    __VOLK_ATTR_ALIGNED(64) float dotProductVector[16];
    _mm512_storeu_ps(dotProductVector,
                     dotProdVal); // Store the results back into the dot product vector

    float dotProduct = dotProductVector[0] + dotProductVector[1] + dotProductVector[2] +
                       dotProductVector[3] + dotProductVector[4] + dotProductVector[5] +
                       dotProductVector[6] + dotProductVector[7] + dotProductVector[8] +
                       dotProductVector[9] + dotProductVector[10] + dotProductVector[11] +
                       dotProductVector[12] + dotProductVector[13] +
                       dotProductVector[14] + dotProductVector[15];

    for (number = sixteenthPoints * 16; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}
#endif /* LV_HAVE_AVX512F */

#endif /*INCLUDED_volk_32f_x2_dot_prod_32f_u_H*/

#ifndef INCLUDED_volk_32f_x2_dot_prod_32f_a_H
#define INCLUDED_volk_32f_x2_dot_prod_32f_a_H

#include <stdio.h>
#include <volk/volk_common.h>


#ifdef LV_HAVE_SSE


static inline void volk_32f_x2_dot_prod_32f_a_sse(float* result,
                                                  const float* input,
                                                  const float* taps,
                                                  unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m128 a0Val, a1Val, a2Val, a3Val;
    __m128 b0Val, b1Val, b2Val, b3Val;
    __m128 c0Val, c1Val, c2Val, c3Val;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        a0Val = _mm_load_ps(aPtr);
        a1Val = _mm_load_ps(aPtr + 4);
        a2Val = _mm_load_ps(aPtr + 8);
        a3Val = _mm_load_ps(aPtr + 12);
        b0Val = _mm_load_ps(bPtr);
        b1Val = _mm_load_ps(bPtr + 4);
        b2Val = _mm_load_ps(bPtr + 8);
        b3Val = _mm_load_ps(bPtr + 12);

        c0Val = _mm_mul_ps(a0Val, b0Val);
        c1Val = _mm_mul_ps(a1Val, b1Val);
        c2Val = _mm_mul_ps(a2Val, b2Val);
        c3Val = _mm_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm_add_ps(c3Val, dotProdVal3);

        aPtr += 16;
        bPtr += 16;
    }

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];

    _mm_store_ps(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];
    dotProduct += dotProductVector[2];
    dotProduct += dotProductVector[3];

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_SSE*/

#ifdef LV_HAVE_SSE3

#include <pmmintrin.h>

static inline void volk_32f_x2_dot_prod_32f_a_sse3(float* result,
                                                   const float* input,
                                                   const float* taps,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m128 a0Val, a1Val, a2Val, a3Val;
    __m128 b0Val, b1Val, b2Val, b3Val;
    __m128 c0Val, c1Val, c2Val, c3Val;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        a0Val = _mm_load_ps(aPtr);
        a1Val = _mm_load_ps(aPtr + 4);
        a2Val = _mm_load_ps(aPtr + 8);
        a3Val = _mm_load_ps(aPtr + 12);
        b0Val = _mm_load_ps(bPtr);
        b1Val = _mm_load_ps(bPtr + 4);
        b2Val = _mm_load_ps(bPtr + 8);
        b3Val = _mm_load_ps(bPtr + 12);

        c0Val = _mm_mul_ps(a0Val, b0Val);
        c1Val = _mm_mul_ps(a1Val, b1Val);
        c2Val = _mm_mul_ps(a2Val, b2Val);
        c3Val = _mm_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm_add_ps(dotProdVal0, c0Val);
        dotProdVal1 = _mm_add_ps(dotProdVal1, c1Val);
        dotProdVal2 = _mm_add_ps(dotProdVal2, c2Val);
        dotProdVal3 = _mm_add_ps(dotProdVal3, c3Val);

        aPtr += 16;
        bPtr += 16;
    }

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];
    _mm_store_ps(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];
    dotProduct += dotProductVector[2];
    dotProduct += dotProductVector[3];

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_SSE3*/

#ifdef LV_HAVE_SSE4_1

#include <smmintrin.h>

static inline void volk_32f_x2_dot_prod_32f_a_sse4_1(float* result,
                                                     const float* input,
                                                     const float* taps,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m128 aVal1, bVal1, cVal1;
    __m128 aVal2, bVal2, cVal2;
    __m128 aVal3, bVal3, cVal3;
    __m128 aVal4, bVal4, cVal4;

    __m128 dotProdVal = _mm_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        aVal1 = _mm_load_ps(aPtr);
        aPtr += 4;
        aVal2 = _mm_load_ps(aPtr);
        aPtr += 4;
        aVal3 = _mm_load_ps(aPtr);
        aPtr += 4;
        aVal4 = _mm_load_ps(aPtr);
        aPtr += 4;

        bVal1 = _mm_load_ps(bPtr);
        bPtr += 4;
        bVal2 = _mm_load_ps(bPtr);
        bPtr += 4;
        bVal3 = _mm_load_ps(bPtr);
        bPtr += 4;
        bVal4 = _mm_load_ps(bPtr);
        bPtr += 4;

        cVal1 = _mm_dp_ps(aVal1, bVal1, 0xF1);
        cVal2 = _mm_dp_ps(aVal2, bVal2, 0xF2);
        cVal3 = _mm_dp_ps(aVal3, bVal3, 0xF4);
        cVal4 = _mm_dp_ps(aVal4, bVal4, 0xF8);

        cVal1 = _mm_or_ps(cVal1, cVal2);
        cVal3 = _mm_or_ps(cVal3, cVal4);
        cVal1 = _mm_or_ps(cVal1, cVal3);

        dotProdVal = _mm_add_ps(dotProdVal, cVal1);
    }

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];
    _mm_store_ps(dotProductVector,
                 dotProdVal); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];
    dotProduct += dotProductVector[2];
    dotProduct += dotProductVector[3];

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#endif /*LV_HAVE_SSE4_1*/

#ifdef LV_HAVE_AVX

#include <immintrin.h>

static inline void volk_32f_x2_dot_prod_32f_a_avx(float* result,
                                                  const float* input,
                                                  const float* taps,
                                                  unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m256 a0Val, a1Val;
    __m256 b0Val, b1Val;
    __m256 c0Val, c1Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        a0Val = _mm256_load_ps(aPtr);
        a1Val = _mm256_load_ps(aPtr + 8);
        b0Val = _mm256_load_ps(bPtr);
        b1Val = _mm256_load_ps(bPtr + 8);

        c0Val = _mm256_mul_ps(a0Val, b0Val);
        c1Val = _mm256_mul_ps(a1Val, b1Val);

        dotProdVal0 = _mm256_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm256_add_ps(c1Val, dotProdVal1);

        aPtr += 16;
        bPtr += 16;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];
    dotProduct += dotProductVector[2];
    dotProduct += dotProductVector[3];
    dotProduct += dotProductVector[4];
    dotProduct += dotProductVector[5];
    dotProduct += dotProductVector[6];
    dotProduct += dotProductVector[7];

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}
#endif /*LV_HAVE_AVX*/


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
static inline void volk_32f_x2_dot_prod_32f_a_avx2_fma(float* result,
                                                       const float* input,
                                                       const float* taps,
                                                       unsigned int num_points)
{
    unsigned int number;
    const unsigned int eighthPoints = num_points / 8;

    const float* aPtr = input;
    const float* bPtr = taps;

    __m256 dotProdVal = _mm256_setzero_ps();
    __m256 aVal1, bVal1;

    for (number = 0; number < eighthPoints; number++) {

        aVal1 = _mm256_load_ps(aPtr);
        bVal1 = _mm256_load_ps(bPtr);
        aPtr += 8;
        bPtr += 8;

        dotProdVal = _mm256_fmadd_ps(aVal1, bVal1, dotProdVal);
    }

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];
    _mm256_store_ps(dotProductVector,
                    dotProdVal); // Store the results back into the dot product vector

    float dotProduct = dotProductVector[0] + dotProductVector[1] + dotProductVector[2] +
                       dotProductVector[3] + dotProductVector[4] + dotProductVector[5] +
                       dotProductVector[6] + dotProductVector[7];

    for (number = eighthPoints * 8; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#if LV_HAVE_AVX512F
#include <immintrin.h>
static inline void volk_32f_x2_dot_prod_32f_a_avx512f(float* result,
                                                      const float* input,
                                                      const float* taps,
                                                      unsigned int num_points)
{
    unsigned int number;
    const unsigned int sixteenthPoints = num_points / 16;

    const float* aPtr = input;
    const float* bPtr = taps;

    __m512 dotProdVal = _mm512_setzero_ps();
    __m512 aVal1, bVal1;

    for (number = 0; number < sixteenthPoints; number++) {

        aVal1 = _mm512_load_ps(aPtr);
        bVal1 = _mm512_load_ps(bPtr);
        aPtr += 16;
        bPtr += 16;

        dotProdVal = _mm512_fmadd_ps(aVal1, bVal1, dotProdVal);
    }

    __VOLK_ATTR_ALIGNED(64) float dotProductVector[16];
    _mm512_store_ps(dotProductVector,
                    dotProdVal); // Store the results back into the dot product vector

    float dotProduct = dotProductVector[0] + dotProductVector[1] + dotProductVector[2] +
                       dotProductVector[3] + dotProductVector[4] + dotProductVector[5] +
                       dotProductVector[6] + dotProductVector[7] + dotProductVector[8] +
                       dotProductVector[9] + dotProductVector[10] + dotProductVector[11] +
                       dotProductVector[12] + dotProductVector[13] +
                       dotProductVector[14] + dotProductVector[15];

    for (number = sixteenthPoints * 16; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_x2_dot_prod_32f_neon(float* result,
                                                 const float* input,
                                                 const float* taps,
                                                 unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    const float* aPtr = input;
    const float* bPtr = taps;

    // Use 4 independent accumulators for better pipelining
    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    float32x4_t acc2 = vdupq_n_f32(0);
    float32x4_t acc3 = vdupq_n_f32(0);

    for (unsigned int number = 0; number < sixteenthPoints; number++) {
        float32x4_t a0 = vld1q_f32(aPtr);
        float32x4_t a1 = vld1q_f32(aPtr + 4);
        float32x4_t a2 = vld1q_f32(aPtr + 8);
        float32x4_t a3 = vld1q_f32(aPtr + 12);

        float32x4_t b0 = vld1q_f32(bPtr);
        float32x4_t b1 = vld1q_f32(bPtr + 4);
        float32x4_t b2 = vld1q_f32(bPtr + 8);
        float32x4_t b3 = vld1q_f32(bPtr + 12);

        acc0 = vmlaq_f32(acc0, a0, b0);
        acc1 = vmlaq_f32(acc1, a1, b1);
        acc2 = vmlaq_f32(acc2, a2, b2);
        acc3 = vmlaq_f32(acc3, a3, b3);

        aPtr += 16;
        bPtr += 16;
    }

    // Combine accumulators
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    // Horizontal sum (ARMv7 compatible)
    float32x2_t sum = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
    sum = vpadd_f32(sum, sum);
    float dotProduct = vget_lane_f32(sum, 0);

    // Handle remainder
    for (unsigned int number = sixteenthPoints * 16; number < num_points; number++) {
        dotProduct += (*aPtr++) * (*bPtr++);
    }

    *result = dotProduct;
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_x2_dot_prod_32f_neonv8(float* result,
                                                   const float* input,
                                                   const float* taps,
                                                   unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    const float* aPtr = input;
    const float* bPtr = taps;

    /* Use 4 independent accumulators for better pipelining with FMA */
    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    float32x4_t acc2 = vdupq_n_f32(0);
    float32x4_t acc3 = vdupq_n_f32(0);

    for (unsigned int number = 0; number < sixteenthPoints; number++) {
        float32x4_t a0 = vld1q_f32(aPtr);
        float32x4_t a1 = vld1q_f32(aPtr + 4);
        float32x4_t a2 = vld1q_f32(aPtr + 8);
        float32x4_t a3 = vld1q_f32(aPtr + 12);

        float32x4_t b0 = vld1q_f32(bPtr);
        float32x4_t b1 = vld1q_f32(bPtr + 4);
        float32x4_t b2 = vld1q_f32(bPtr + 8);
        float32x4_t b3 = vld1q_f32(bPtr + 12);
        __VOLK_PREFETCH(aPtr + 32);
        __VOLK_PREFETCH(bPtr + 32);

        /* Use FMA for accumulate */
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        acc2 = vfmaq_f32(acc2, a2, b2);
        acc3 = vfmaq_f32(acc3, a3, b3);

        aPtr += 16;
        bPtr += 16;
    }

    /* Combine accumulators */
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    /* Horizontal sum */
    float dotProduct = vaddvq_f32(acc0);

    /* Handle remainder */
    for (unsigned int number = sixteenthPoints * 16; number < num_points; number++) {
        dotProduct += (*aPtr++) * (*bPtr++);
    }

    *result = dotProduct;
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_NEONV7
extern void volk_32f_x2_dot_prod_32f_a_neonasm(float* cVector,
                                               const float* aVector,
                                               const float* bVector,
                                               unsigned int num_points);
#endif /* LV_HAVE_NEONV7 */

#ifdef LV_HAVE_NEONV7
extern void volk_32f_x2_dot_prod_32f_a_neonasm_opts(float* cVector,
                                                    const float* aVector,
                                                    const float* bVector,
                                                    unsigned int num_points);
#endif /* LV_HAVE_NEONV7 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_32f_x2_dot_prod_32f_rvv(float* result,
                                                const float* input,
                                                const float* taps,
                                                unsigned int num_points)
{
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0, __riscv_vsetvlmax_e32m8());
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, input += vl, taps += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(input, vl);
        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(taps, vl);
        vsum = __riscv_vfmacc_tu(vsum, v0, v1, vl);
    }
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t v = RISCV_SHRINK8(vfadd, f, 32, vsum);
    v = __riscv_vfredusum(v, __riscv_vfmv_s_f_f32m1(0, vl), vl);
    *result = __riscv_vfmv_f(v);
}
#endif /*LV_HAVE_RVV*/

#endif /*INCLUDED_volk_32f_x2_dot_prod_32f_a_H*/
