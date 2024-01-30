/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16u_byteswap
 *
 * \b Overview
 *
 * Byteswaps (in-place) an aligned vector of int16_t's.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16u_byteswap(uint16_t* intsToSwap, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li intsToSwap: The vector of data to byte swap.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li intsToSwap: returns as an in-place calculation.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * <FIXME>
 *
 * volk_16u_byteswap(x, N);
 *
 * \endcode
 */

#ifndef INCLUDED_volk_16u_byteswap_u_H
#define INCLUDED_volk_16u_byteswap_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16u_byteswap_generic(uint16_t* intsToSwap,
                                             unsigned int num_points)
{
    uint16_t* inputPtr = intsToSwap;
    for (unsigned int point = 0; point < num_points; point++) {
        uint16_t output = *inputPtr;
        output = (((output >> 8) & 0xff) | ((output << 8) & 0xff00));
        *inputPtr = output;
        inputPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */


#if LV_HAVE_AVX2
#include <immintrin.h>
static inline void volk_16u_byteswap_a_avx2(uint16_t* intsToSwap, unsigned int num_points)
{
    unsigned int number;

    const unsigned int nPerSet = 16;
    const uint64_t nSets = num_points / nPerSet;

    uint16_t* inputPtr = (uint16_t*)intsToSwap;

    const uint8_t shuffleVector[32] = { 1,  0,  3,  2,  5,  4,  7,  6,  9,  8,  11,
                                        10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20,
                                        23, 22, 25, 24, 27, 26, 29, 28, 31, 30 };

    const __m256i myShuffle = _mm256_loadu_si256((__m256i*)&shuffleVector[0]);

    for (number = 0; number < nSets; number++) {
        // Load the 32t values, increment inputPtr later since we're doing it in-place.
        const __m256i input = _mm256_load_si256((__m256i*)inputPtr);
        const __m256i output = _mm256_shuffle_epi8(input, myShuffle);

        // Store the results
        _mm256_store_si256((__m256i*)inputPtr, output);
        inputPtr += nPerSet;
    }

    // Byteswap any remaining points:
    for (number = nPerSet * nSets; number < num_points; number++) {
        uint16_t outputVal = *inputPtr;
        outputVal = (((outputVal >> 8) & 0xff) | ((outputVal << 8) & 0xff00));
        *inputPtr = outputVal;
        inputPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */


#if LV_HAVE_AVX2
#include <immintrin.h>
static inline void volk_16u_byteswap_u_avx2(uint16_t* intsToSwap, unsigned int num_points)
{
    unsigned int number;

    const unsigned int nPerSet = 16;
    const uint64_t nSets = num_points / nPerSet;

    uint16_t* inputPtr = (uint16_t*)intsToSwap;

    const uint8_t shuffleVector[32] = { 1,  0,  3,  2,  5,  4,  7,  6,  9,  8,  11,
                                        10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20,
                                        23, 22, 25, 24, 27, 26, 29, 28, 31, 30 };

    const __m256i myShuffle = _mm256_loadu_si256((__m256i*)&shuffleVector[0]);

    for (number = 0; number < nSets; number++) {
        // Load the 32t values, increment inputPtr later since we're doing it in-place.
        const __m256i input = _mm256_loadu_si256((__m256i*)inputPtr);
        const __m256i output = _mm256_shuffle_epi8(input, myShuffle);

        // Store the results
        _mm256_storeu_si256((__m256i*)inputPtr, output);
        inputPtr += nPerSet;
    }

    // Byteswap any remaining points:
    for (number = nPerSet * nSets; number < num_points; number++) {
        uint16_t outputVal = *inputPtr;
        outputVal = (((outputVal >> 8) & 0xff) | ((outputVal << 8) & 0xff00));
        *inputPtr = outputVal;
        inputPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16u_byteswap_u_sse2(uint16_t* intsToSwap, unsigned int num_points)
{
    unsigned int number = 0;
    uint16_t* inputPtr = intsToSwap;
    __m128i input, left, right, output;

    const unsigned int eighthPoints = num_points / 8;
    for (; number < eighthPoints; number++) {
        // Load the 16t values, increment inputPtr later since we're doing it in-place.
        input = _mm_loadu_si128((__m128i*)inputPtr);
        // Do the two shifts
        left = _mm_slli_epi16(input, 8);
        right = _mm_srli_epi16(input, 8);
        // Or the left and right halves together
        output = _mm_or_si128(left, right);
        // Store the results
        _mm_storeu_si128((__m128i*)inputPtr, output);
        inputPtr += 8;
    }

    // Byteswap any remaining points:
    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        uint16_t outputVal = *inputPtr;
        outputVal = (((outputVal >> 8) & 0xff) | ((outputVal << 8) & 0xff00));
        *inputPtr = outputVal;
        inputPtr++;
    }
}
#endif /* LV_HAVE_SSE2 */


#endif /* INCLUDED_volk_16u_byteswap_u_H */
#ifndef INCLUDED_volk_16u_byteswap_a_H
#define INCLUDED_volk_16u_byteswap_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16u_byteswap_a_sse2(uint16_t* intsToSwap, unsigned int num_points)
{
    uint16_t* inputPtr = intsToSwap;
    __m128i input, left, right, output;

    const unsigned int eighthPoints = num_points / 8;
    for (unsigned int number = 0; number < eighthPoints; number++) {
        // Load the 16t values, increment inputPtr later since we're doing it in-place.
        input = _mm_load_si128((__m128i*)inputPtr);
        // Do the two shifts
        left = _mm_slli_epi16(input, 8);
        right = _mm_srli_epi16(input, 8);
        // Or the left and right halves together
        output = _mm_or_si128(left, right);
        // Store the results
        _mm_store_si128((__m128i*)inputPtr, output);
        inputPtr += 8;
    }

    // Byteswap any remaining points:
    volk_16u_byteswap_generic(inputPtr, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16u_byteswap_neon(uint16_t* intsToSwap, unsigned int num_points)
{
    unsigned int number;
    unsigned int eighth_points = num_points / 8;
    uint16x8_t input, output;
    uint16_t* inputPtr = intsToSwap;

    for (number = 0; number < eighth_points; number++) {
        input = vld1q_u16(inputPtr);
        output = vsriq_n_u16(output, input, 8);
        output = vsliq_n_u16(output, input, 8);
        vst1q_u16(inputPtr, output);
        inputPtr += 8;
    }

    volk_16u_byteswap_generic(inputPtr, num_points - eighth_points * 8);
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16u_byteswap_neon_table(uint16_t* intsToSwap,
                                                unsigned int num_points)
{
    uint16_t* inputPtr = intsToSwap;
    unsigned int number = 0;
    unsigned int n16points = num_points / 16;

    uint8x8x4_t input_table;
    uint8x8_t int_lookup01, int_lookup23, int_lookup45, int_lookup67;
    uint8x8_t swapped_int01, swapped_int23, swapped_int45, swapped_int67;

    /* these magic numbers are used as byte-indices in the LUT.
       they are pre-computed to save time. A simple C program
       can calculate them; for example for lookup01:
      uint8_t chars[8] = {24, 16, 8, 0, 25, 17, 9, 1};
      for(ii=0; ii < 8; ++ii) {
          index += ((uint64_t)(*(chars+ii))) << (ii*8);
      }
    */
    int_lookup01 = vcreate_u8(1232017111498883080);
    int_lookup23 = vcreate_u8(1376697457175036426);
    int_lookup45 = vcreate_u8(1521377802851189772);
    int_lookup67 = vcreate_u8(1666058148527343118);

    for (number = 0; number < n16points; ++number) {
        input_table = vld4_u8((uint8_t*)inputPtr);
        swapped_int01 = vtbl4_u8(input_table, int_lookup01);
        swapped_int23 = vtbl4_u8(input_table, int_lookup23);
        swapped_int45 = vtbl4_u8(input_table, int_lookup45);
        swapped_int67 = vtbl4_u8(input_table, int_lookup67);
        vst1_u8((uint8_t*)inputPtr, swapped_int01);
        vst1_u8((uint8_t*)(inputPtr + 4), swapped_int23);
        vst1_u8((uint8_t*)(inputPtr + 8), swapped_int45);
        vst1_u8((uint8_t*)(inputPtr + 12), swapped_int67);

        inputPtr += 16;
    }

    volk_16u_byteswap_generic(inputPtr, num_points - n16points * 16);
}
#endif /* LV_HAVE_NEON */


#endif /* INCLUDED_volk_16u_byteswap_a_H */
