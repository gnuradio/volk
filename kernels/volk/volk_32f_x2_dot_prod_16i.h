/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_x2_dot_prod_16i
 *
 * \b Overview
 *
 * This block computes the dot product (or inner product) between two
 * vectors, the \p input and \p taps vectors. Given a set of \p
 * num_points taps, the result is the sum of products between the two
 * vectors. The result is a single value stored in the \p result
 * address and is conerted to a fixed-point short.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_dot_prod_16i(int16_t* result, const float* input, const float* taps,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li input: vector of floats.
 * \li taps:  float taps.
 * \li num_points: number of samples in both \p input and \p taps.
 *
 * \b Outputs
 * \li result: pointer to a short value to hold the dot product result.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * <FIXME>
 *
 * volk_32f_x2_dot_prod_16i();
 *
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_dot_prod_16i_H
#define INCLUDED_volk_32f_x2_dot_prod_16i_H

#include <stdio.h>
#include <volk/volk_common.h>


#ifdef LV_HAVE_GENERIC


static inline void volk_32f_x2_dot_prod_16i_generic(int16_t* result,
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

    *result = (int16_t)rintf(dotProduct);
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_SSE

static inline void volk_32f_x2_dot_prod_16i_a_sse(int16_t* result,
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

    *result = (short)rintf(dotProduct);
}

#endif /*LV_HAVE_SSE*/


#if LV_HAVE_AVX2 && LV_HAVE_FMA

static inline void volk_32f_x2_dot_prod_16i_a_avx2_fma(int16_t* result,
                                                       const float* input,
                                                       const float* taps,
                                                       unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < thirtysecondPoints; number++) {

        a0Val = _mm256_load_ps(aPtr);
        a1Val = _mm256_load_ps(aPtr + 8);
        a2Val = _mm256_load_ps(aPtr + 16);
        a3Val = _mm256_load_ps(aPtr + 24);
        b0Val = _mm256_load_ps(bPtr);
        b1Val = _mm256_load_ps(bPtr + 8);
        b2Val = _mm256_load_ps(bPtr + 16);
        b3Val = _mm256_load_ps(bPtr + 24);

        dotProdVal0 = _mm256_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm256_fmadd_ps(a1Val, b1Val, dotProdVal1);
        dotProdVal2 = _mm256_fmadd_ps(a2Val, b2Val, dotProdVal2);
        dotProdVal3 = _mm256_fmadd_ps(a3Val, b3Val, dotProdVal3);

        aPtr += 32;
        bPtr += 32;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

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

    number = thirtysecondPoints * 32;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = (short)rintf(dotProduct);
}

#endif /*LV_HAVE_AVX2 && LV_HAVE_FMA*/


#ifdef LV_HAVE_AVX

static inline void volk_32f_x2_dot_prod_16i_a_avx(int16_t* result,
                                                  const float* input,
                                                  const float* taps,
                                                  unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;
    __m256 c0Val, c1Val, c2Val, c3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < thirtysecondPoints; number++) {

        a0Val = _mm256_load_ps(aPtr);
        a1Val = _mm256_load_ps(aPtr + 8);
        a2Val = _mm256_load_ps(aPtr + 16);
        a3Val = _mm256_load_ps(aPtr + 24);
        b0Val = _mm256_load_ps(bPtr);
        b1Val = _mm256_load_ps(bPtr + 8);
        b2Val = _mm256_load_ps(bPtr + 16);
        b3Val = _mm256_load_ps(bPtr + 24);

        c0Val = _mm256_mul_ps(a0Val, b0Val);
        c1Val = _mm256_mul_ps(a1Val, b1Val);
        c2Val = _mm256_mul_ps(a2Val, b2Val);
        c3Val = _mm256_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm256_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm256_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm256_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm256_add_ps(c3Val, dotProdVal3);

        aPtr += 32;
        bPtr += 32;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

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

    number = thirtysecondPoints * 32;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = (short)rintf(dotProduct);
}

#endif /*LV_HAVE_AVX*/

#ifdef LV_HAVE_AVX512F

static inline void volk_32f_x2_dot_prod_16i_a_avx512f(int16_t* result,
                                                      const float* input,
                                                      const float* taps,
                                                      unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixtyfourthPoints = num_points / 64;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m512 a0Val, a1Val, a2Val, a3Val;
    __m512 b0Val, b1Val, b2Val, b3Val;

    __m512 dotProdVal0 = _mm512_setzero_ps();
    __m512 dotProdVal1 = _mm512_setzero_ps();
    __m512 dotProdVal2 = _mm512_setzero_ps();
    __m512 dotProdVal3 = _mm512_setzero_ps();

    for (; number < sixtyfourthPoints; number++) {

        a0Val = _mm512_load_ps(aPtr);
        a1Val = _mm512_load_ps(aPtr + 16);
        a2Val = _mm512_load_ps(aPtr + 32);
        a3Val = _mm512_load_ps(aPtr + 48);
        b0Val = _mm512_load_ps(bPtr);
        b1Val = _mm512_load_ps(bPtr + 16);
        b2Val = _mm512_load_ps(bPtr + 32);
        b3Val = _mm512_load_ps(bPtr + 48);

        dotProdVal0 = _mm512_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm512_fmadd_ps(a1Val, b1Val, dotProdVal1);
        dotProdVal2 = _mm512_fmadd_ps(a2Val, b2Val, dotProdVal2);
        dotProdVal3 = _mm512_fmadd_ps(a3Val, b3Val, dotProdVal3);

        aPtr += 64;
        bPtr += 64;
    }

    dotProdVal0 = _mm512_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm512_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm512_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(64) float dotProductVector[16];

    _mm512_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];
    dotProduct += dotProductVector[2];
    dotProduct += dotProductVector[3];
    dotProduct += dotProductVector[4];
    dotProduct += dotProductVector[5];
    dotProduct += dotProductVector[6];
    dotProduct += dotProductVector[7];
    dotProduct += dotProductVector[8];
    dotProduct += dotProductVector[9];
    dotProduct += dotProductVector[10];
    dotProduct += dotProductVector[11];
    dotProduct += dotProductVector[12];
    dotProduct += dotProductVector[13];
    dotProduct += dotProductVector[14];
    dotProduct += dotProductVector[15];

    number = sixtyfourthPoints * 64;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = (short)rintf(dotProduct);
}

#endif /*LV_HAVE_AVX512F*/


#ifdef LV_HAVE_SSE

static inline void volk_32f_x2_dot_prod_16i_u_sse(int16_t* result,
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

    *result = (short)rintf(dotProduct);
}

#endif /*LV_HAVE_SSE*/


#if LV_HAVE_AVX2 && LV_HAVE_FMA

static inline void volk_32f_x2_dot_prod_16i_u_avx2_fma(int16_t* result,
                                                       const float* input,
                                                       const float* taps,
                                                       unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < thirtysecondPoints; number++) {

        a0Val = _mm256_loadu_ps(aPtr);
        a1Val = _mm256_loadu_ps(aPtr + 8);
        a2Val = _mm256_loadu_ps(aPtr + 16);
        a3Val = _mm256_loadu_ps(aPtr + 24);
        b0Val = _mm256_loadu_ps(bPtr);
        b1Val = _mm256_loadu_ps(bPtr + 8);
        b2Val = _mm256_loadu_ps(bPtr + 16);
        b3Val = _mm256_loadu_ps(bPtr + 24);

        dotProdVal0 = _mm256_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm256_fmadd_ps(a1Val, b1Val, dotProdVal1);
        dotProdVal2 = _mm256_fmadd_ps(a2Val, b2Val, dotProdVal2);
        dotProdVal3 = _mm256_fmadd_ps(a3Val, b3Val, dotProdVal3);

        aPtr += 32;
        bPtr += 32;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

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

    number = thirtysecondPoints * 32;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = (short)rintf(dotProduct);
}

#endif /*LV_HAVE_AVX2 && lV_HAVE_FMA*/


#ifdef LV_HAVE_AVX

static inline void volk_32f_x2_dot_prod_16i_u_avx(int16_t* result,
                                                  const float* input,
                                                  const float* taps,
                                                  unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;
    __m256 c0Val, c1Val, c2Val, c3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < thirtysecondPoints; number++) {

        a0Val = _mm256_loadu_ps(aPtr);
        a1Val = _mm256_loadu_ps(aPtr + 8);
        a2Val = _mm256_loadu_ps(aPtr + 16);
        a3Val = _mm256_loadu_ps(aPtr + 24);
        b0Val = _mm256_loadu_ps(bPtr);
        b1Val = _mm256_loadu_ps(bPtr + 8);
        b2Val = _mm256_loadu_ps(bPtr + 16);
        b3Val = _mm256_loadu_ps(bPtr + 24);

        c0Val = _mm256_mul_ps(a0Val, b0Val);
        c1Val = _mm256_mul_ps(a1Val, b1Val);
        c2Val = _mm256_mul_ps(a2Val, b2Val);
        c3Val = _mm256_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm256_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm256_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm256_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm256_add_ps(c3Val, dotProdVal3);

        aPtr += 32;
        bPtr += 32;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

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

    number = thirtysecondPoints * 32;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = (short)rintf(dotProduct);
}

#endif /*LV_HAVE_AVX*/

#ifdef LV_HAVE_AVX512F

static inline void volk_32f_x2_dot_prod_16i_u_avx512f(int16_t* result,
                                                      const float* input,
                                                      const float* taps,
                                                      unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixtyfourthPoints = num_points / 64;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    __m512 a0Val, a1Val, a2Val, a3Val;
    __m512 b0Val, b1Val, b2Val, b3Val;

    __m512 dotProdVal0 = _mm512_setzero_ps();
    __m512 dotProdVal1 = _mm512_setzero_ps();
    __m512 dotProdVal2 = _mm512_setzero_ps();
    __m512 dotProdVal3 = _mm512_setzero_ps();

    for (; number < sixtyfourthPoints; number++) {

        a0Val = _mm512_loadu_ps(aPtr);
        a1Val = _mm512_loadu_ps(aPtr + 16);
        a2Val = _mm512_loadu_ps(aPtr + 32);
        a3Val = _mm512_loadu_ps(aPtr + 48);
        b0Val = _mm512_loadu_ps(bPtr);
        b1Val = _mm512_loadu_ps(bPtr + 16);
        b2Val = _mm512_loadu_ps(bPtr + 32);
        b3Val = _mm512_loadu_ps(bPtr + 48);

        dotProdVal0 = _mm512_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm512_fmadd_ps(a1Val, b1Val, dotProdVal1);
        dotProdVal2 = _mm512_fmadd_ps(a2Val, b2Val, dotProdVal2);
        dotProdVal3 = _mm512_fmadd_ps(a3Val, b3Val, dotProdVal3);

        aPtr += 64;
        bPtr += 64;
    }

    dotProdVal0 = _mm512_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm512_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm512_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(64) float dotProductVector[16];

    _mm512_storeu_ps(dotProductVector,
                     dotProdVal0); // Store the results back into the dot product vector

    dotProduct = dotProductVector[0];
    dotProduct += dotProductVector[1];
    dotProduct += dotProductVector[2];
    dotProduct += dotProductVector[3];
    dotProduct += dotProductVector[4];
    dotProduct += dotProductVector[5];
    dotProduct += dotProductVector[6];
    dotProduct += dotProductVector[7];
    dotProduct += dotProductVector[8];
    dotProduct += dotProductVector[9];
    dotProduct += dotProductVector[10];
    dotProduct += dotProductVector[11];
    dotProduct += dotProductVector[12];
    dotProduct += dotProductVector[13];
    dotProduct += dotProductVector[14];
    dotProduct += dotProductVector[15];

    number = sixtyfourthPoints * 64;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = (short)rintf(dotProduct);
}

#endif /*LV_HAVE_AVX512F*/

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_x2_dot_prod_16i_neon(int16_t* result,
                                                 const float* input,
                                                 const float* taps,
                                                 unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    float32x4_t dotProdVal0 = vdupq_n_f32(0.0f);
    float32x4_t dotProdVal1 = vdupq_n_f32(0.0f);
    float32x4_t dotProdVal2 = vdupq_n_f32(0.0f);
    float32x4_t dotProdVal3 = vdupq_n_f32(0.0f);

    for (; number < sixteenthPoints; number++) {
        float32x4_t a0Val = vld1q_f32(aPtr);
        float32x4_t a1Val = vld1q_f32(aPtr + 4);
        float32x4_t a2Val = vld1q_f32(aPtr + 8);
        float32x4_t a3Val = vld1q_f32(aPtr + 12);

        float32x4_t b0Val = vld1q_f32(bPtr);
        float32x4_t b1Val = vld1q_f32(bPtr + 4);
        float32x4_t b2Val = vld1q_f32(bPtr + 8);
        float32x4_t b3Val = vld1q_f32(bPtr + 12);

        dotProdVal0 = vmlaq_f32(dotProdVal0, a0Val, b0Val);
        dotProdVal1 = vmlaq_f32(dotProdVal1, a1Val, b1Val);
        dotProdVal2 = vmlaq_f32(dotProdVal2, a2Val, b2Val);
        dotProdVal3 = vmlaq_f32(dotProdVal3, a3Val, b3Val);

        aPtr += 16;
        bPtr += 16;
    }

    dotProdVal0 = vaddq_f32(dotProdVal0, dotProdVal1);
    dotProdVal0 = vaddq_f32(dotProdVal0, dotProdVal2);
    dotProdVal0 = vaddq_f32(dotProdVal0, dotProdVal3);

    float32x2_t sum = vadd_f32(vget_low_f32(dotProdVal0), vget_high_f32(dotProdVal0));
    sum = vpadd_f32(sum, sum);
    dotProduct = vget_lane_f32(sum, 0);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = (int16_t)rintf(dotProduct);
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_x2_dot_prod_16i_neonv8(int16_t* result,
                                                   const float* input,
                                                   const float* taps,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float dotProduct = 0;
    const float* aPtr = input;
    const float* bPtr = taps;

    float32x4_t dotProdVal0 = vdupq_n_f32(0.0f);
    float32x4_t dotProdVal1 = vdupq_n_f32(0.0f);
    float32x4_t dotProdVal2 = vdupq_n_f32(0.0f);
    float32x4_t dotProdVal3 = vdupq_n_f32(0.0f);

    for (; number < sixteenthPoints; number++) {
        float32x4_t a0Val = vld1q_f32(aPtr);
        float32x4_t a1Val = vld1q_f32(aPtr + 4);
        float32x4_t a2Val = vld1q_f32(aPtr + 8);
        float32x4_t a3Val = vld1q_f32(aPtr + 12);

        float32x4_t b0Val = vld1q_f32(bPtr);
        float32x4_t b1Val = vld1q_f32(bPtr + 4);
        float32x4_t b2Val = vld1q_f32(bPtr + 8);
        float32x4_t b3Val = vld1q_f32(bPtr + 12);
        __VOLK_PREFETCH(aPtr + 16);
        __VOLK_PREFETCH(bPtr + 16);

        dotProdVal0 = vfmaq_f32(dotProdVal0, a0Val, b0Val);
        dotProdVal1 = vfmaq_f32(dotProdVal1, a1Val, b1Val);
        dotProdVal2 = vfmaq_f32(dotProdVal2, a2Val, b2Val);
        dotProdVal3 = vfmaq_f32(dotProdVal3, a3Val, b3Val);

        aPtr += 16;
        bPtr += 16;
    }

    dotProdVal0 = vaddq_f32(dotProdVal0, dotProdVal1);
    dotProdVal0 = vaddq_f32(dotProdVal0, dotProdVal2);
    dotProdVal0 = vaddq_f32(dotProdVal0, dotProdVal3);

    dotProduct = vaddvq_f32(dotProdVal0);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = (int16_t)rintf(dotProduct);
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

#include "volk_32f_x2_dot_prod_32f.h"

static inline void volk_32f_x2_dot_prod_16i_rvv(int16_t* result,
                                                const float* input,
                                                const float* taps,
                                                unsigned int num_points)
{
    float fresult = 0;
    volk_32f_x2_dot_prod_32f_rvv(&fresult, input, taps, num_points);
    *result = (int16_t)rintf(fresult);
}
#endif /*LV_HAVE_RVV*/

#endif /*INCLUDED_volk_32f_x2_dot_prod_16i_H*/
