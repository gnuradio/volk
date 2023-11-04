/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_64u_byteswap
 *
 * \b Overview
 *
 * Byteswaps (in-place) an aligned vector of int64_t's.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_64u_byteswap(uint64_t* intsToSwap, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li intsToSwap: The vector of data to byte swap
 * \li num_points: The number of data points
 *
 * \b Outputs
 * \li intsToSwap: returns as an in-place calculation.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *
 *   uint64_t bitstring[] = {0x0, 0x1, 0xf, 0xffffffffffffffff,
 *       0x5a5a5a5a5a5a5a5a, 0xa5a5a5a5a5a5a5a5, 0x2a2a2a2a2a2a2a2a,
 *       0xffffffff, 0x32, 0x64};
 *   uint64_t hamming_distance = 0;
 *
 *   printf("byteswap vector =\n");
 *   for(unsigned int ii=0; ii<N; ++ii){
 *       printf("    %.16lx\n", bitstring[ii]);
 *   }
 *
 *   volk_64u_byteswap(bitstring, N);
 *
 *   printf("byteswapped vector =\n");
 *   for(unsigned int ii=0; ii<N; ++ii){
 *       printf("    %.16lx\n", bitstring[ii]);
 *   }
 * \endcode
 */

#ifndef INCLUDED_volk_64u_byteswap_u_H
#define INCLUDED_volk_64u_byteswap_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_64u_byteswap_u_sse2(uint64_t* intsToSwap, unsigned int num_points)
{
    uint32_t* inputPtr = (uint32_t*)intsToSwap;
    __m128i input, byte1, byte2, byte3, byte4, output;
    __m128i byte2mask = _mm_set1_epi32(0x00FF0000);
    __m128i byte3mask = _mm_set1_epi32(0x0000FF00);
    uint64_t number = 0;
    const unsigned int halfPoints = num_points / 2;
    for (; number < halfPoints; number++) {
        // Load the 32t values, increment inputPtr later since we're doing it in-place.
        input = _mm_loadu_si128((__m128i*)inputPtr);

        // Do the four shifts
        byte1 = _mm_slli_epi32(input, 24);
        byte2 = _mm_slli_epi32(input, 8);
        byte3 = _mm_srli_epi32(input, 8);
        byte4 = _mm_srli_epi32(input, 24);
        // Or bytes together
        output = _mm_or_si128(byte1, byte4);
        byte2 = _mm_and_si128(byte2, byte2mask);
        output = _mm_or_si128(output, byte2);
        byte3 = _mm_and_si128(byte3, byte3mask);
        output = _mm_or_si128(output, byte3);

        // Reorder the two words
        output = _mm_shuffle_epi32(output, _MM_SHUFFLE(2, 3, 0, 1));

        // Store the results
        _mm_storeu_si128((__m128i*)inputPtr, output);
        inputPtr += 4;
    }

    // Byteswap any remaining points:
    number = halfPoints * 2;
    for (; number < num_points; number++) {
        uint32_t output1 = *inputPtr;
        uint32_t output2 = inputPtr[1];

        output1 = (((output1 >> 24) & 0xff) | ((output1 >> 8) & 0x0000ff00) |
                   ((output1 << 8) & 0x00ff0000) | ((output1 << 24) & 0xff000000));

        output2 = (((output2 >> 24) & 0xff) | ((output2 >> 8) & 0x0000ff00) |
                   ((output2 << 8) & 0x00ff0000) | ((output2 << 24) & 0xff000000));

        *inputPtr++ = output2;
        *inputPtr++ = output1;
    }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void volk_64u_byteswap_generic(uint64_t* intsToSwap,
                                             unsigned int num_points)
{
    uint32_t* inputPtr = (uint32_t*)intsToSwap;
    unsigned int point;
    for (point = 0; point < num_points; point++) {
        uint32_t output1 = *inputPtr;
        uint32_t output2 = inputPtr[1];

        output1 = (((output1 >> 24) & 0xff) | ((output1 >> 8) & 0x0000ff00) |
                   ((output1 << 8) & 0x00ff0000) | ((output1 << 24) & 0xff000000));

        output2 = (((output2 >> 24) & 0xff) | ((output2 >> 8) & 0x0000ff00) |
                   ((output2 << 8) & 0x00ff0000) | ((output2 << 24) & 0xff000000));

        *inputPtr++ = output2;
        *inputPtr++ = output1;
    }
}
#endif /* LV_HAVE_GENERIC */

#if LV_HAVE_AVX2
#include <immintrin.h>
static inline void volk_64u_byteswap_a_avx2(uint64_t* intsToSwap, unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int nPerSet = 4;
    const uint64_t nSets = num_points / nPerSet;

    uint32_t* inputPtr = (uint32_t*)intsToSwap;

    const uint8_t shuffleVector[32] = { 7,  6,  5,  4,  3,  2,  1,  0,  15, 14, 13,
                                        12, 11, 10, 9,  8,  23, 22, 21, 20, 19, 18,
                                        17, 16, 31, 30, 29, 28, 27, 26, 25, 24 };

    const __m256i myShuffle = _mm256_loadu_si256((__m256i*)&shuffleVector[0]);

    for (; number < nSets; number++) {

        // Load the 32t values, increment inputPtr later since we're doing it in-place.
        const __m256i input = _mm256_load_si256((__m256i*)inputPtr);
        const __m256i output = _mm256_shuffle_epi8(input, myShuffle);

        // Store the results
        _mm256_store_si256((__m256i*)inputPtr, output);

        /*  inputPtr is 32bit so increment twice  */
        inputPtr += 2 * nPerSet;
    }

    // Byteswap any remaining points:
    for (number = nSets * nPerSet; number < num_points; ++number) {
        uint32_t output1 = *inputPtr;
        uint32_t output2 = inputPtr[1];
        uint32_t out1 =
            ((((output1) >> 24) & 0x000000ff) | (((output1) >> 8) & 0x0000ff00) |
             (((output1) << 8) & 0x00ff0000) | (((output1) << 24) & 0xff000000));

        uint32_t out2 =
            ((((output2) >> 24) & 0x000000ff) | (((output2) >> 8) & 0x0000ff00) |
             (((output2) << 8) & 0x00ff0000) | (((output2) << 24) & 0xff000000));
        *inputPtr++ = out2;
        *inputPtr++ = out1;
    }
}

#endif /* LV_HAVE_AVX2 */


#if LV_HAVE_SSSE3
#include <tmmintrin.h>
static inline void volk_64u_byteswap_a_ssse3(uint64_t* intsToSwap,
                                             unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int nPerSet = 2;
    const uint64_t nSets = num_points / nPerSet;

    uint32_t* inputPtr = (uint32_t*)intsToSwap;

    uint8_t shuffleVector[16] = { 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8 };

    const __m128i myShuffle = _mm_loadu_si128((__m128i*)&shuffleVector);

    for (; number < nSets; number++) {

        // Load the 32t values, increment inputPtr later since we're doing it in-place.
        const __m128i input = _mm_load_si128((__m128i*)inputPtr);
        const __m128i output = _mm_shuffle_epi8(input, myShuffle);

        // Store the results
        _mm_store_si128((__m128i*)inputPtr, output);

        /*  inputPtr is 32bit so increment twice  */
        inputPtr += 2 * nPerSet;
    }

    // Byteswap any remaining points:
    for (number = nSets * nPerSet; number < num_points; ++number) {
        uint32_t output1 = *inputPtr;
        uint32_t output2 = inputPtr[1];
        uint32_t out1 =
            ((((output1) >> 24) & 0x000000ff) | (((output1) >> 8) & 0x0000ff00) |
             (((output1) << 8) & 0x00ff0000) | (((output1) << 24) & 0xff000000));

        uint32_t out2 =
            ((((output2) >> 24) & 0x000000ff) | (((output2) >> 8) & 0x0000ff00) |
             (((output2) << 8) & 0x00ff0000) | (((output2) << 24) & 0xff000000));
        *inputPtr++ = out2;
        *inputPtr++ = out1;
    }
}
#endif /* LV_HAVE_SSSE3 */
#endif /* INCLUDED_volk_64u_byteswap_u_H */


#ifndef INCLUDED_volk_64u_byteswap_a_H
#define INCLUDED_volk_64u_byteswap_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_64u_byteswap_a_sse2(uint64_t* intsToSwap, unsigned int num_points)
{
    uint32_t* inputPtr = (uint32_t*)intsToSwap;
    __m128i input, byte1, byte2, byte3, byte4, output;
    __m128i byte2mask = _mm_set1_epi32(0x00FF0000);
    __m128i byte3mask = _mm_set1_epi32(0x0000FF00);
    uint64_t number = 0;
    const unsigned int halfPoints = num_points / 2;
    for (; number < halfPoints; number++) {
        // Load the 32t values, increment inputPtr later since we're doing it in-place.
        input = _mm_load_si128((__m128i*)inputPtr);

        // Do the four shifts
        byte1 = _mm_slli_epi32(input, 24);
        byte2 = _mm_slli_epi32(input, 8);
        byte3 = _mm_srli_epi32(input, 8);
        byte4 = _mm_srli_epi32(input, 24);
        // Or bytes together
        output = _mm_or_si128(byte1, byte4);
        byte2 = _mm_and_si128(byte2, byte2mask);
        output = _mm_or_si128(output, byte2);
        byte3 = _mm_and_si128(byte3, byte3mask);
        output = _mm_or_si128(output, byte3);

        // Reorder the two words
        output = _mm_shuffle_epi32(output, _MM_SHUFFLE(2, 3, 0, 1));

        // Store the results
        _mm_store_si128((__m128i*)inputPtr, output);
        inputPtr += 4;
    }

    // Byteswap any remaining points:
    number = halfPoints * 2;
    for (; number < num_points; number++) {
        uint32_t output1 = *inputPtr;
        uint32_t output2 = inputPtr[1];

        output1 = (((output1 >> 24) & 0xff) | ((output1 >> 8) & 0x0000ff00) |
                   ((output1 << 8) & 0x00ff0000) | ((output1 << 24) & 0xff000000));

        output2 = (((output2 >> 24) & 0xff) | ((output2 >> 8) & 0x0000ff00) |
                   ((output2 << 8) & 0x00ff0000) | ((output2 << 24) & 0xff000000));

        *inputPtr++ = output2;
        *inputPtr++ = output1;
    }
}
#endif /* LV_HAVE_SSE2 */

#if LV_HAVE_AVX2
#include <immintrin.h>
static inline void volk_64u_byteswap_u_avx2(uint64_t* intsToSwap, unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int nPerSet = 4;
    const uint64_t nSets = num_points / nPerSet;

    uint32_t* inputPtr = (uint32_t*)intsToSwap;

    const uint8_t shuffleVector[32] = { 7,  6,  5,  4,  3,  2,  1,  0,  15, 14, 13,
                                        12, 11, 10, 9,  8,  23, 22, 21, 20, 19, 18,
                                        17, 16, 31, 30, 29, 28, 27, 26, 25, 24 };

    const __m256i myShuffle = _mm256_loadu_si256((__m256i*)&shuffleVector[0]);

    for (; number < nSets; number++) {
        // Load the 32t values, increment inputPtr later since we're doing it in-place.
        const __m256i input = _mm256_loadu_si256((__m256i*)inputPtr);
        const __m256i output = _mm256_shuffle_epi8(input, myShuffle);

        // Store the results
        _mm256_storeu_si256((__m256i*)inputPtr, output);

        /*  inputPtr is 32bit so increment twice  */
        inputPtr += 2 * nPerSet;
    }

    // Byteswap any remaining points:
    for (number = nSets * nPerSet; number < num_points; ++number) {
        uint32_t output1 = *inputPtr;
        uint32_t output2 = inputPtr[1];
        uint32_t out1 =
            ((((output1) >> 24) & 0x000000ff) | (((output1) >> 8) & 0x0000ff00) |
             (((output1) << 8) & 0x00ff0000) | (((output1) << 24) & 0xff000000));

        uint32_t out2 =
            ((((output2) >> 24) & 0x000000ff) | (((output2) >> 8) & 0x0000ff00) |
             (((output2) << 8) & 0x00ff0000) | (((output2) << 24) & 0xff000000));
        *inputPtr++ = out2;
        *inputPtr++ = out1;
    }
}

#endif /* LV_HAVE_AVX2 */


#if LV_HAVE_SSSE3
#include <tmmintrin.h>
static inline void volk_64u_byteswap_u_ssse3(uint64_t* intsToSwap,
                                             unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int nPerSet = 2;
    const uint64_t nSets = num_points / nPerSet;

    uint32_t* inputPtr = (uint32_t*)intsToSwap;

    uint8_t shuffleVector[16] = { 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8 };

    const __m128i myShuffle = _mm_loadu_si128((__m128i*)&shuffleVector);

    for (; number < nSets; number++) {
        // Load the 32t values, increment inputPtr later since we're doing it in-place.
        const __m128i input = _mm_loadu_si128((__m128i*)inputPtr);
        const __m128i output = _mm_shuffle_epi8(input, myShuffle);

        // Store the results
        _mm_storeu_si128((__m128i*)inputPtr, output);

        /*  inputPtr is 32bit so increment twice  */
        inputPtr += 2 * nPerSet;
    }

    // Byteswap any remaining points:
    for (number = nSets * nPerSet; number < num_points; ++number) {
        uint32_t output1 = *inputPtr;
        uint32_t output2 = inputPtr[1];
        uint32_t out1 =
            ((((output1) >> 24) & 0x000000ff) | (((output1) >> 8) & 0x0000ff00) |
             (((output1) << 8) & 0x00ff0000) | (((output1) << 24) & 0xff000000));

        uint32_t out2 =
            ((((output2) >> 24) & 0x000000ff) | (((output2) >> 8) & 0x0000ff00) |
             (((output2) << 8) & 0x00ff0000) | (((output2) << 24) & 0xff000000));
        *inputPtr++ = out2;
        *inputPtr++ = out1;
    }
}
#endif /* LV_HAVE_SSSE3 */


#endif /* INCLUDED_volk_64u_byteswap_a_H */
