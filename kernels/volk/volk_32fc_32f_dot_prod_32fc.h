/* -*- c++ -*- */
/*
 * Copyright 2012, 2013, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_32f_dot_prod_32fc
 *
 * \b Overview
 *
 * This block computes the dot product (or inner product) between two
 * vectors, the \p input and \p taps vectors. Given a set of \p
 * num_points taps, the result is the sum of products between the two
 * vectors. The result is a single value stored in the \p result
 * address and will be complex.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_32f_dot_prod_32fc(lv_32fc_t* result, const lv_32fc_t* input, const float
 * * taps, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li input: vector of complex samples
 * \li taps:  floating point taps
 * \li num_points: number of samples in both \p input and \p taps
 *
 * \b Outputs
 * \li result: pointer to a complex value to hold the dot product result.
 *
 * \b Example
 * \code
 * int N = 10000;
 * lv_32fc_t y;
 * lv_32fc_t *x = (lv_32fc_t*)volk_malloc(N*sizeof(lv_32fc_t), volk_get_alignment());
 * float *t = (float*)volk_malloc(N*sizeof(float), volk_get_alignment());
 *
 * <populate x and t with some values>
 *
 * volk_32fc_dot_prod_32fc(&y, x, t, N);
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_32f_dot_prod_32fc_a_H
#define INCLUDED_volk_32fc_32f_dot_prod_32fc_a_H

#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_32f_dot_prod_32fc_generic(lv_32fc_t* result,
                                                       const lv_32fc_t* input,
                                                       const float* taps,
                                                       unsigned int num_points)
{

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const float* aPtr = (float*)input;
    const float* bPtr = taps;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[1] * bPtr[0]);
        aPtr += 2;
        bPtr += 1;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_GENERIC*/

#ifdef LV_HAVE_AVX512F

#include <immintrin.h>

static inline void volk_32fc_32f_dot_prod_32fc_a_avx512f(lv_32fc_t* result,
                                                         const lv_32fc_t* input,
                                                         const float* taps,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const float* aPtr = (float*)input;
    const float* bPtr = taps;

    __m512 a0Val, a1Val;
    __m512 b0Val, b1Val;
    __m512 xVal;

    __m512 dotProdVal0 = _mm512_setzero_ps();
    __m512 dotProdVal1 = _mm512_setzero_ps();

    // Create index patterns for duplication: 0,0,1,1,2,2,3,3,...,15,15
    const __m512i idx = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const __m512i idx2 =
        _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);

    for (; number < sixteenthPoints; number++) {
        // Load 16 complex numbers (32 floats)
        a0Val = _mm512_load_ps(aPtr);      // 8 complex (I0,Q0,I1,Q1,...)
        a1Val = _mm512_load_ps(aPtr + 16); // 8 complex (I8,Q8,I9,Q9,...)

        // Load 16 real taps
        xVal = _mm512_load_ps(bPtr); // t0|t1|t2|...|t15

        // Duplicate each tap value to match complex format using permutexvar
        b0Val = _mm512_permutexvar_ps(idx, xVal);
        b1Val = _mm512_permutexvar_ps(idx2, xVal);

        dotProdVal0 = _mm512_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm512_fmadd_ps(a1Val, b1Val, dotProdVal1);

        aPtr += 32;
        bPtr += 16;
    }

    dotProdVal0 = _mm512_add_ps(dotProdVal0, dotProdVal1);

    __VOLK_ATTR_ALIGNED(64) float dotProductVector[16];
    _mm512_store_ps(dotProductVector, dotProdVal0);

    for (unsigned int i = 0; i < 16; i += 2) {
        returnValue += lv_cmake(dotProductVector[i], dotProductVector[i + 1]);
    }

    number = sixteenthPoints * 16;
    lv_32fc_t returnTail = lv_cmake(0.0f, 0.0f);
    volk_32fc_32f_dot_prod_32fc_generic(
        &returnTail, input + number, bPtr, num_points - number);
    returnValue += returnTail;

    *result = returnValue;
}

#endif /*LV_HAVE_AVX512F*/

#if LV_HAVE_AVX2 && LV_HAVE_FMA

#include <immintrin.h>

static inline void volk_32fc_32f_dot_prod_32fc_a_avx2_fma(lv_32fc_t* result,
                                                          const lv_32fc_t* input,
                                                          const float* taps,
                                                          unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const float* aPtr = (float*)input;
    const float* bPtr = taps;

    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;
    __m256 x0Val, x1Val, x0loVal, x0hiVal, x1loVal, x1hiVal;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        a0Val = _mm256_load_ps(aPtr);
        a1Val = _mm256_load_ps(aPtr + 8);
        a2Val = _mm256_load_ps(aPtr + 16);
        a3Val = _mm256_load_ps(aPtr + 24);

        x0Val = _mm256_load_ps(bPtr); // t0|t1|t2|t3|t4|t5|t6|t7
        x1Val = _mm256_load_ps(bPtr + 8);
        x0loVal = _mm256_unpacklo_ps(x0Val, x0Val); // t0|t0|t1|t1|t4|t4|t5|t5
        x0hiVal = _mm256_unpackhi_ps(x0Val, x0Val); // t2|t2|t3|t3|t6|t6|t7|t7
        x1loVal = _mm256_unpacklo_ps(x1Val, x1Val);
        x1hiVal = _mm256_unpackhi_ps(x1Val, x1Val);

        // TODO: it may be possible to rearrange swizzling to better pipeline data
        b0Val = _mm256_permute2f128_ps(x0loVal, x0hiVal, 0x20); // t0|t0|t1|t1|t2|t2|t3|t3
        b1Val = _mm256_permute2f128_ps(x0loVal, x0hiVal, 0x31); // t4|t4|t5|t5|t6|t6|t7|t7
        b2Val = _mm256_permute2f128_ps(x1loVal, x1hiVal, 0x20);
        b3Val = _mm256_permute2f128_ps(x1loVal, x1hiVal, 0x31);

        dotProdVal0 = _mm256_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm256_fmadd_ps(a1Val, b1Val, dotProdVal1);
        dotProdVal2 = _mm256_fmadd_ps(a2Val, b2Val, dotProdVal2);
        dotProdVal3 = _mm256_fmadd_ps(a3Val, b3Val, dotProdVal3);

        aPtr += 32;
        bPtr += 16;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);
    returnValue += lv_cmake(dotProductVector[4], dotProductVector[5]);
    returnValue += lv_cmake(dotProductVector[6], dotProductVector[7]);

    number = sixteenthPoints * 16;
    lv_32fc_t returnTail = lv_cmake(0.0f, 0.0f);
    volk_32fc_32f_dot_prod_32fc_generic(
        &returnTail, input + number, bPtr, num_points - number);
    returnValue += returnTail;

    *result = returnValue;
}

#endif /*LV_HAVE_AVX2 && LV_HAVE_FMA*/

#ifdef LV_HAVE_AVX

#include <immintrin.h>

static inline void volk_32fc_32f_dot_prod_32fc_a_avx(lv_32fc_t* result,
                                                     const lv_32fc_t* input,
                                                     const float* taps,
                                                     unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const float* aPtr = (float*)input;
    const float* bPtr = taps;

    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;
    __m256 x0Val, x1Val, x0loVal, x0hiVal, x1loVal, x1hiVal;
    __m256 c0Val, c1Val, c2Val, c3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        a0Val = _mm256_load_ps(aPtr);
        a1Val = _mm256_load_ps(aPtr + 8);
        a2Val = _mm256_load_ps(aPtr + 16);
        a3Val = _mm256_load_ps(aPtr + 24);

        x0Val = _mm256_load_ps(bPtr); // t0|t1|t2|t3|t4|t5|t6|t7
        x1Val = _mm256_load_ps(bPtr + 8);
        x0loVal = _mm256_unpacklo_ps(x0Val, x0Val); // t0|t0|t1|t1|t4|t4|t5|t5
        x0hiVal = _mm256_unpackhi_ps(x0Val, x0Val); // t2|t2|t3|t3|t6|t6|t7|t7
        x1loVal = _mm256_unpacklo_ps(x1Val, x1Val);
        x1hiVal = _mm256_unpackhi_ps(x1Val, x1Val);

        // TODO: it may be possible to rearrange swizzling to better pipeline data
        b0Val = _mm256_permute2f128_ps(x0loVal, x0hiVal, 0x20); // t0|t0|t1|t1|t2|t2|t3|t3
        b1Val = _mm256_permute2f128_ps(x0loVal, x0hiVal, 0x31); // t4|t4|t5|t5|t6|t6|t7|t7
        b2Val = _mm256_permute2f128_ps(x1loVal, x1hiVal, 0x20);
        b3Val = _mm256_permute2f128_ps(x1loVal, x1hiVal, 0x31);

        c0Val = _mm256_mul_ps(a0Val, b0Val);
        c1Val = _mm256_mul_ps(a1Val, b1Val);
        c2Val = _mm256_mul_ps(a2Val, b2Val);
        c3Val = _mm256_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm256_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm256_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm256_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm256_add_ps(c3Val, dotProdVal3);

        aPtr += 32;
        bPtr += 16;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);
    returnValue += lv_cmake(dotProductVector[4], dotProductVector[5]);
    returnValue += lv_cmake(dotProductVector[6], dotProductVector[7]);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[1] * bPtr[0]);
        aPtr += 2;
        bPtr += 1;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_AVX*/


#ifdef LV_HAVE_SSE


static inline void volk_32fc_32f_dot_prod_32fc_a_sse(lv_32fc_t* result,
                                                     const lv_32fc_t* input,
                                                     const float* taps,
                                                     unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const float* aPtr = (float*)input;
    const float* bPtr = taps;

    __m128 a0Val, a1Val, a2Val, a3Val;
    __m128 b0Val, b1Val, b2Val, b3Val;
    __m128 x0Val, x1Val, x2Val, x3Val;
    __m128 c0Val, c1Val, c2Val, c3Val;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < eighthPoints; number++) {

        a0Val = _mm_load_ps(aPtr);
        a1Val = _mm_load_ps(aPtr + 4);
        a2Val = _mm_load_ps(aPtr + 8);
        a3Val = _mm_load_ps(aPtr + 12);

        x0Val = _mm_load_ps(bPtr);
        x1Val = _mm_load_ps(bPtr);
        x2Val = _mm_load_ps(bPtr + 4);
        x3Val = _mm_load_ps(bPtr + 4);
        b0Val = _mm_unpacklo_ps(x0Val, x1Val);
        b1Val = _mm_unpackhi_ps(x0Val, x1Val);
        b2Val = _mm_unpacklo_ps(x2Val, x3Val);
        b3Val = _mm_unpackhi_ps(x2Val, x3Val);

        c0Val = _mm_mul_ps(a0Val, b0Val);
        c1Val = _mm_mul_ps(a1Val, b1Val);
        c2Val = _mm_mul_ps(a2Val, b2Val);
        c3Val = _mm_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm_add_ps(c3Val, dotProdVal3);

        aPtr += 16;
        bPtr += 8;
    }

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];

    _mm_store_ps(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[1] * bPtr[0]);
        aPtr += 2;
        bPtr += 1;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_SSE*/

#ifdef LV_HAVE_AVX512F

#include <immintrin.h>

static inline void volk_32fc_32f_dot_prod_32fc_u_avx512f(lv_32fc_t* result,
                                                         const lv_32fc_t* input,
                                                         const float* taps,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const float* aPtr = (float*)input;
    const float* bPtr = taps;

    __m512 a0Val, a1Val;
    __m512 b0Val, b1Val;
    __m512 xVal;

    __m512 dotProdVal0 = _mm512_setzero_ps();
    __m512 dotProdVal1 = _mm512_setzero_ps();

    // Create index patterns for duplication: 0,0,1,1,2,2,3,3,...,15,15
    const __m512i idx = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const __m512i idx2 =
        _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);

    for (; number < sixteenthPoints; number++) {
        // Load 16 complex numbers (32 floats) - unaligned
        a0Val = _mm512_loadu_ps(aPtr);      // 8 complex (I0,Q0,I1,Q1,...)
        a1Val = _mm512_loadu_ps(aPtr + 16); // 8 complex (I8,Q8,I9,Q9,...)

        // Load 16 real taps - unaligned
        xVal = _mm512_loadu_ps(bPtr); // t0|t1|t2|...|t15

        // Duplicate each tap value to match complex format using permutexvar
        b0Val = _mm512_permutexvar_ps(idx, xVal);
        b1Val = _mm512_permutexvar_ps(idx2, xVal);

        dotProdVal0 = _mm512_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm512_fmadd_ps(a1Val, b1Val, dotProdVal1);

        aPtr += 32;
        bPtr += 16;
    }

    dotProdVal0 = _mm512_add_ps(dotProdVal0, dotProdVal1);

    __VOLK_ATTR_ALIGNED(64) float dotProductVector[16];
    _mm512_store_ps(dotProductVector, dotProdVal0);

    for (unsigned int i = 0; i < 16; i += 2) {
        returnValue += lv_cmake(dotProductVector[i], dotProductVector[i + 1]);
    }

    number = sixteenthPoints * 16;
    lv_32fc_t returnTail = lv_cmake(0.0f, 0.0f);
    volk_32fc_32f_dot_prod_32fc_generic(
        &returnTail, input + number, bPtr, num_points - number);
    returnValue += returnTail;

    *result = returnValue;
}

#endif /*LV_HAVE_AVX512F*/

#if LV_HAVE_AVX2 && LV_HAVE_FMA

#include <immintrin.h>

static inline void volk_32fc_32f_dot_prod_32fc_u_avx2_fma(lv_32fc_t* result,
                                                          const lv_32fc_t* input,
                                                          const float* taps,
                                                          unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const float* aPtr = (float*)input;
    const float* bPtr = taps;

    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;
    __m256 x0Val, x1Val, x0loVal, x0hiVal, x1loVal, x1hiVal;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        a0Val = _mm256_loadu_ps(aPtr);
        a1Val = _mm256_loadu_ps(aPtr + 8);
        a2Val = _mm256_loadu_ps(aPtr + 16);
        a3Val = _mm256_loadu_ps(aPtr + 24);

        x0Val = _mm256_loadu_ps(bPtr); // t0|t1|t2|t3|t4|t5|t6|t7
        x1Val = _mm256_loadu_ps(bPtr + 8);
        x0loVal = _mm256_unpacklo_ps(x0Val, x0Val); // t0|t0|t1|t1|t4|t4|t5|t5
        x0hiVal = _mm256_unpackhi_ps(x0Val, x0Val); // t2|t2|t3|t3|t6|t6|t7|t7
        x1loVal = _mm256_unpacklo_ps(x1Val, x1Val);
        x1hiVal = _mm256_unpackhi_ps(x1Val, x1Val);

        // TODO: it may be possible to rearrange swizzling to better pipeline data
        b0Val = _mm256_permute2f128_ps(x0loVal, x0hiVal, 0x20); // t0|t0|t1|t1|t2|t2|t3|t3
        b1Val = _mm256_permute2f128_ps(x0loVal, x0hiVal, 0x31); // t4|t4|t5|t5|t6|t6|t7|t7
        b2Val = _mm256_permute2f128_ps(x1loVal, x1hiVal, 0x20);
        b3Val = _mm256_permute2f128_ps(x1loVal, x1hiVal, 0x31);

        dotProdVal0 = _mm256_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm256_fmadd_ps(a1Val, b1Val, dotProdVal1);
        dotProdVal2 = _mm256_fmadd_ps(a2Val, b2Val, dotProdVal2);
        dotProdVal3 = _mm256_fmadd_ps(a3Val, b3Val, dotProdVal3);

        aPtr += 32;
        bPtr += 16;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);
    returnValue += lv_cmake(dotProductVector[4], dotProductVector[5]);
    returnValue += lv_cmake(dotProductVector[6], dotProductVector[7]);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[1] * bPtr[0]);
        aPtr += 2;
        bPtr += 1;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_AVX2 && LV_HAVE_FMA*/

#ifdef LV_HAVE_AVX

#include <immintrin.h>

static inline void volk_32fc_32f_dot_prod_32fc_u_avx(lv_32fc_t* result,
                                                     const lv_32fc_t* input,
                                                     const float* taps,
                                                     unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const float* aPtr = (float*)input;
    const float* bPtr = taps;

    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;
    __m256 x0Val, x1Val, x0loVal, x0hiVal, x1loVal, x1hiVal;
    __m256 c0Val, c1Val, c2Val, c3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        a0Val = _mm256_loadu_ps(aPtr);
        a1Val = _mm256_loadu_ps(aPtr + 8);
        a2Val = _mm256_loadu_ps(aPtr + 16);
        a3Val = _mm256_loadu_ps(aPtr + 24);

        x0Val = _mm256_loadu_ps(bPtr); // t0|t1|t2|t3|t4|t5|t6|t7
        x1Val = _mm256_loadu_ps(bPtr + 8);
        x0loVal = _mm256_unpacklo_ps(x0Val, x0Val); // t0|t0|t1|t1|t4|t4|t5|t5
        x0hiVal = _mm256_unpackhi_ps(x0Val, x0Val); // t2|t2|t3|t3|t6|t6|t7|t7
        x1loVal = _mm256_unpacklo_ps(x1Val, x1Val);
        x1hiVal = _mm256_unpackhi_ps(x1Val, x1Val);

        // TODO: it may be possible to rearrange swizzling to better pipeline data
        b0Val = _mm256_permute2f128_ps(x0loVal, x0hiVal, 0x20); // t0|t0|t1|t1|t2|t2|t3|t3
        b1Val = _mm256_permute2f128_ps(x0loVal, x0hiVal, 0x31); // t4|t4|t5|t5|t6|t6|t7|t7
        b2Val = _mm256_permute2f128_ps(x1loVal, x1hiVal, 0x20);
        b3Val = _mm256_permute2f128_ps(x1loVal, x1hiVal, 0x31);

        c0Val = _mm256_mul_ps(a0Val, b0Val);
        c1Val = _mm256_mul_ps(a1Val, b1Val);
        c2Val = _mm256_mul_ps(a2Val, b2Val);
        c3Val = _mm256_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm256_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm256_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm256_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm256_add_ps(c3Val, dotProdVal3);

        aPtr += 32;
        bPtr += 16;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);
    returnValue += lv_cmake(dotProductVector[4], dotProductVector[5]);
    returnValue += lv_cmake(dotProductVector[6], dotProductVector[7]);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[1] * bPtr[0]);
        aPtr += 2;
        bPtr += 1;
    }

    *result = returnValue;
}
#endif /*LV_HAVE_AVX*/

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32fc_32f_dot_prod_32fc_neon_unroll(lv_32fc_t* __restrict result,
                                        const lv_32fc_t* __restrict input,
                                        const float* __restrict taps,
                                        unsigned int num_points)
{

    unsigned int number;
    const unsigned int quarterPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const float* inputPtr = (float*)input;
    const float* tapsPtr = taps;
    float zero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float accVector_real[4];
    float accVector_imag[4];

    float32x4x2_t inputVector0, inputVector1;
    float32x4_t tapsVector0, tapsVector1;
    float32x4_t tmp_real0, tmp_imag0;
    float32x4_t tmp_real1, tmp_imag1;
    float32x4_t real_accumulator0, imag_accumulator0;
    float32x4_t real_accumulator1, imag_accumulator1;

    // zero out accumulators
    // take a *float, return float32x4_t
    real_accumulator0 = vld1q_f32(zero);
    imag_accumulator0 = vld1q_f32(zero);
    real_accumulator1 = vld1q_f32(zero);
    imag_accumulator1 = vld1q_f32(zero);

    for (number = 0; number < quarterPoints; number++) {
        // load doublewords and duplicate in to second lane
        tapsVector0 = vld1q_f32(tapsPtr);
        tapsVector1 = vld1q_f32(tapsPtr + 4);

        // load quadword of complex numbers in to 2 lanes. 1st lane is real, 2dn imag
        inputVector0 = vld2q_f32(inputPtr);
        inputVector1 = vld2q_f32(inputPtr + 8);
        // inputVector is now a struct of two vectors, 0th is real, 1st is imag

        tmp_real0 = vmulq_f32(tapsVector0, inputVector0.val[0]);
        tmp_imag0 = vmulq_f32(tapsVector0, inputVector0.val[1]);

        tmp_real1 = vmulq_f32(tapsVector1, inputVector1.val[0]);
        tmp_imag1 = vmulq_f32(tapsVector1, inputVector1.val[1]);

        real_accumulator0 = vaddq_f32(real_accumulator0, tmp_real0);
        imag_accumulator0 = vaddq_f32(imag_accumulator0, tmp_imag0);

        real_accumulator1 = vaddq_f32(real_accumulator1, tmp_real1);
        imag_accumulator1 = vaddq_f32(imag_accumulator1, tmp_imag1);

        tapsPtr += 8;
        inputPtr += 16;
    }

    real_accumulator0 = vaddq_f32(real_accumulator0, real_accumulator1);
    imag_accumulator0 = vaddq_f32(imag_accumulator0, imag_accumulator1);
    // void vst1q_f32( float32_t * ptr, float32x4_t val);
    // store results back to a complex (array of 2 floats)
    vst1q_f32(accVector_real, real_accumulator0);
    vst1q_f32(accVector_imag, imag_accumulator0);
    returnValue += lv_cmake(
        accVector_real[0] + accVector_real[1] + accVector_real[2] + accVector_real[3],
        accVector_imag[0] + accVector_imag[1] + accVector_imag[2] + accVector_imag[3]);

    // clean up the remainder
    for (number = quarterPoints * 8; number < num_points; number++) {
        returnValue += lv_cmake(inputPtr[0] * tapsPtr[0], inputPtr[1] * tapsPtr[0]);
        inputPtr += 2;
        tapsPtr += 1;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_NEON*/

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_32f_dot_prod_32fc_a_neon(lv_32fc_t* __restrict result,
                                                      const lv_32fc_t* __restrict input,
                                                      const float* __restrict taps,
                                                      unsigned int num_points)
{

    unsigned int number;
    const unsigned int quarterPoints = num_points / 4;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const float* inputPtr = (float*)input;
    const float* tapsPtr = taps;
    float zero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float accVector_real[4];
    float accVector_imag[4];

    float32x4x2_t inputVector;
    float32x4_t tapsVector;
    float32x4_t tmp_real, tmp_imag;
    float32x4_t real_accumulator, imag_accumulator;


    // zero out accumulators
    // take a *float, return float32x4_t
    real_accumulator = vld1q_f32(zero);
    imag_accumulator = vld1q_f32(zero);

    for (number = 0; number < quarterPoints; number++) {
        // load taps ( float32x2x2_t = vld1q_f32( float32_t const * ptr) )
        // load doublewords and duplicate in to second lane
        tapsVector = vld1q_f32(tapsPtr);

        // load quadword of complex numbers in to 2 lanes. 1st lane is real, 2dn imag
        inputVector = vld2q_f32(inputPtr);

        tmp_real = vmulq_f32(tapsVector, inputVector.val[0]);
        tmp_imag = vmulq_f32(tapsVector, inputVector.val[1]);

        real_accumulator = vaddq_f32(real_accumulator, tmp_real);
        imag_accumulator = vaddq_f32(imag_accumulator, tmp_imag);


        tapsPtr += 4;
        inputPtr += 8;
    }

    // store results back to a complex (array of 2 floats)
    vst1q_f32(accVector_real, real_accumulator);
    vst1q_f32(accVector_imag, imag_accumulator);
    returnValue += lv_cmake(
        accVector_real[0] + accVector_real[1] + accVector_real[2] + accVector_real[3],
        accVector_imag[0] + accVector_imag[1] + accVector_imag[2] + accVector_imag[3]);

    // clean up the remainder
    for (number = quarterPoints * 4; number < num_points; number++) {
        returnValue += lv_cmake(inputPtr[0] * tapsPtr[0], inputPtr[1] * tapsPtr[0]);
        inputPtr += 2;
        tapsPtr += 1;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_NEON*/

#ifdef LV_HAVE_NEONV7
extern void volk_32fc_32f_dot_prod_32fc_a_neonasm(lv_32fc_t* result,
                                                  const lv_32fc_t* input,
                                                  const float* taps,
                                                  unsigned int num_points);
#endif /*LV_HAVE_NEONV7*/

#ifdef LV_HAVE_NEONV7
extern void volk_32fc_32f_dot_prod_32fc_a_neonasmvmla(lv_32fc_t* result,
                                                      const lv_32fc_t* input,
                                                      const float* taps,
                                                      unsigned int num_points);
#endif /*LV_HAVE_NEONV7*/

#ifdef LV_HAVE_NEONV7
extern void volk_32fc_32f_dot_prod_32fc_a_neonpipeline(lv_32fc_t* result,
                                                       const lv_32fc_t* input,
                                                       const float* taps,
                                                       unsigned int num_points);
#endif /*LV_HAVE_NEONV7*/

#ifdef LV_HAVE_SSE

static inline void volk_32fc_32f_dot_prod_32fc_u_sse(lv_32fc_t* result,
                                                     const lv_32fc_t* input,
                                                     const float* taps,
                                                     unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const float* aPtr = (float*)input;
    const float* bPtr = taps;

    __m128 a0Val, a1Val, a2Val, a3Val;
    __m128 b0Val, b1Val, b2Val, b3Val;
    __m128 x0Val, x1Val, x2Val, x3Val;
    __m128 c0Val, c1Val, c2Val, c3Val;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < eighthPoints; number++) {

        a0Val = _mm_loadu_ps(aPtr);
        a1Val = _mm_loadu_ps(aPtr + 4);
        a2Val = _mm_loadu_ps(aPtr + 8);
        a3Val = _mm_loadu_ps(aPtr + 12);

        x0Val = _mm_loadu_ps(bPtr);
        x1Val = _mm_loadu_ps(bPtr);
        x2Val = _mm_loadu_ps(bPtr + 4);
        x3Val = _mm_loadu_ps(bPtr + 4);
        b0Val = _mm_unpacklo_ps(x0Val, x1Val);
        b1Val = _mm_unpackhi_ps(x0Val, x1Val);
        b2Val = _mm_unpacklo_ps(x2Val, x3Val);
        b3Val = _mm_unpackhi_ps(x2Val, x3Val);

        c0Val = _mm_mul_ps(a0Val, b0Val);
        c1Val = _mm_mul_ps(a1Val, b1Val);
        c2Val = _mm_mul_ps(a2Val, b2Val);
        c3Val = _mm_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm_add_ps(c3Val, dotProdVal3);

        aPtr += 16;
        bPtr += 8;
    }

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];

    _mm_store_ps(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[1] * bPtr[0]);
        aPtr += 2;
        bPtr += 1;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_SSE*/

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_32fc_32f_dot_prod_32fc_rvv(lv_32fc_t* result,
                                                   const lv_32fc_t* input,
                                                   const float* taps,
                                                   unsigned int num_points)
{
    vfloat32m4_t vsumr = __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
    vfloat32m4_t vsumi = vsumr;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, input += vl, taps += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint64m8_t va = __riscv_vle64_v_u64m8((const uint64_t*)input, vl);
        vfloat32m4_t vbr = __riscv_vle32_v_f32m4(taps, vl), vbi = vbr;
        vfloat32m4_t var = __riscv_vreinterpret_f32m4(__riscv_vnsrl(va, 0, vl));
        vfloat32m4_t vai = __riscv_vreinterpret_f32m4(__riscv_vnsrl(va, 32, vl));
        vsumr = __riscv_vfmacc_tu(vsumr, var, vbr, vl);
        vsumi = __riscv_vfmacc_tu(vsumi, vai, vbi, vl);
    }
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vr = RISCV_SHRINK4(vfadd, f, 32, vsumr);
    vfloat32m1_t vi = RISCV_SHRINK4(vfadd, f, 32, vsumi);
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0, vl);
    *result = lv_cmake(__riscv_vfmv_f(__riscv_vfredusum(vr, z, vl)),
                       __riscv_vfmv_f(__riscv_vfredusum(vi, z, vl)));
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_32fc_32f_dot_prod_32fc_rvvseg(lv_32fc_t* result,
                                                      const lv_32fc_t* input,
                                                      const float* taps,
                                                      unsigned int num_points)
{
    vfloat32m4_t vsumr = __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
    vfloat32m4_t vsumi = vsumr;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, input += vl, taps += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t va = __riscv_vlseg2e32_v_f32m4x2((const float*)input, vl);
        vfloat32m4_t var = __riscv_vget_f32m4(va, 0), vai = __riscv_vget_f32m4(va, 1);
        vfloat32m4_t vbr = __riscv_vle32_v_f32m4(taps, vl), vbi = vbr;
        vsumr = __riscv_vfmacc_tu(vsumr, var, vbr, vl);
        vsumi = __riscv_vfmacc_tu(vsumi, vai, vbi, vl);
    }
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vr = RISCV_SHRINK4(vfadd, f, 32, vsumr);
    vfloat32m1_t vi = RISCV_SHRINK4(vfadd, f, 32, vsumi);
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0, vl);
    *result = lv_cmake(__riscv_vfmv_f(__riscv_vfredusum(vr, z, vl)),
                       __riscv_vfmv_f(__riscv_vfredusum(vi, z, vl)));
}
#endif /*LV_HAVE_RVVSEG*/

#endif /*INCLUDED_volk_32fc_32f_dot_prod_32fc_H*/
