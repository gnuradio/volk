/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_convert_8i
 *
 * \b Overview
 *
 * Converts 16-bit signed integers to 8-bit signed integers by arithmetic
 * right-shifting each input value by 8 bits (keeping the high byte).
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_convert_8i(int8_t* outputVector, const int16_t* inputVector, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector of 16-bit signed integers (int16_t).
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: The output vector of 8-bit signed integers (int8_t).
 *
 * \b Example
 * Convert 16-bit samples to 8-bit by extracting the high byte.
 * \code
 *   #include <volk/volk.h>
 *   #include <stdio.h>
 *
 *   int main() {
 *     unsigned int num_points = 8;
 *     unsigned int alignment = volk_get_alignment();
 *
 *     // Allocate aligned memory
 *     int16_t* input =
 *         (int16_t*)volk_malloc(sizeof(int16_t) * num_points, alignment);
 *     int8_t* output =
 *         (int8_t*)volk_malloc(sizeof(int8_t) * num_points, alignment);
 *
 *     // Initialize with values whose high byte is meaningful
 *     // Right-shifting by 8 keeps the upper byte: 0x0100 >> 8 = 1, etc.
 *     input[0] = 0x0100;  // 256  -> 1
 *     input[1] = 0x0200;  // 512  -> 2
 *     input[2] = 0x0A00;  // 2560 -> 10
 *     input[3] = 0x7F00;  // 32512 -> 127
 *     input[4] = -256;    // 0xFF00 -> -1 (sign-preserving)
 *     input[5] = -512;    // 0xFE00 -> -2
 *     input[6] = 0x0050;  // 80   -> 0 (low byte discarded)
 *     input[7] = 0x03C0;  // 960  -> 3
 *
 *     // Convert 16-bit to 8-bit: output[i] = (int8_t)(input[i] >> 8)
 *     volk_16i_convert_8i(output, input, num_points);
 *
 *     for (unsigned int i = 0; i < num_points; i++) {
 *       printf("input[%u] = %6d  ->  output[%u] = %4d\n",
 *              i, input[i], i, output[i]);
 *     }
 *
 *     volk_free(input);
 *     volk_free(output);
 *     return 0;
 *   }
 * \endcode
 */

#ifndef INCLUDED_volk_16i_convert_8i_u_H
#define INCLUDED_volk_16i_convert_8i_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_convert_8i_u_avx2(int8_t* outputVector,
                                              const int16_t* inputVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    int8_t* outputVectorPtr = outputVector;
    int16_t* inputPtr = (int16_t*)inputVector;
    __m256i inputVal1;
    __m256i inputVal2;
    __m256i ret;

    for (; number < thirtysecondPoints; number++) {

        // Load the 16 values
        inputVal1 = _mm256_loadu_si256((__m256i*)inputPtr);
        inputPtr += 16;
        inputVal2 = _mm256_loadu_si256((__m256i*)inputPtr);
        inputPtr += 16;

        inputVal1 = _mm256_srai_epi16(inputVal1, 8);
        inputVal2 = _mm256_srai_epi16(inputVal2, 8);

        ret = _mm256_packs_epi16(inputVal1, inputVal2);
        ret = _mm256_permute4x64_epi64(ret, 0b11011000);

        _mm256_storeu_si256((__m256i*)outputVectorPtr, ret);

        outputVectorPtr += 32;
    }

    number = thirtysecondPoints * 32;
    for (; number < num_points; number++) {
        outputVector[number] = (int8_t)(inputVector[number] >> 8);
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16i_convert_8i_u_avx512bw(int8_t* outputVector,
                                                  const int16_t* inputVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixtyfourthPoints = num_points / 64;

    int8_t* outputVectorPtr = outputVector;
    int16_t* inputPtr = (int16_t*)inputVector;
    __m512i inputVal1;
    __m512i inputVal2;
    __m512i shifted1, shifted2;
    __m256i ret1, ret2;

    for (; number < sixtyfourthPoints; number++) {

        // Load 64 int16 values
        inputVal1 = _mm512_loadu_si512((__m512i*)inputPtr);
        inputPtr += 32;
        inputVal2 = _mm512_loadu_si512((__m512i*)inputPtr);
        inputPtr += 32;

        shifted1 = _mm512_srai_epi16(inputVal1, 8);
        shifted2 = _mm512_srai_epi16(inputVal2, 8);

        ret1 = _mm512_cvtsepi16_epi8(shifted1);
        ret2 = _mm512_cvtsepi16_epi8(shifted2);

        _mm256_storeu_si256((__m256i*)outputVectorPtr, ret1);
        outputVectorPtr += 32;
        _mm256_storeu_si256((__m256i*)outputVectorPtr, ret2);
        outputVectorPtr += 32;
    }

    number = sixtyfourthPoints * 64;
    for (; number < num_points; number++) {
        outputVector[number] = (int8_t)(inputVector[number] >> 8);
    }
}
#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16i_convert_8i_u_sse2(int8_t* outputVector,
                                              const int16_t* inputVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    int8_t* outputVectorPtr = outputVector;
    int16_t* inputPtr = (int16_t*)inputVector;
    __m128i inputVal1;
    __m128i inputVal2;
    __m128i ret;

    for (; number < sixteenthPoints; number++) {

        // Load the 16 values
        inputVal1 = _mm_loadu_si128((__m128i*)inputPtr);
        inputPtr += 8;
        inputVal2 = _mm_loadu_si128((__m128i*)inputPtr);
        inputPtr += 8;

        inputVal1 = _mm_srai_epi16(inputVal1, 8);
        inputVal2 = _mm_srai_epi16(inputVal2, 8);

        ret = _mm_packs_epi16(inputVal1, inputVal2);

        _mm_storeu_si128((__m128i*)outputVectorPtr, ret);

        outputVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (int8_t)(inputVector[number] >> 8);
    }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void volk_16i_convert_8i_generic(int8_t* outputVector,
                                               const int16_t* inputVector,
                                               unsigned int num_points)
{
    int8_t* outputVectorPtr = outputVector;
    const int16_t* inputVectorPtr = inputVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *outputVectorPtr++ = ((int8_t)(*inputVectorPtr++ >> 8));
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_16i_convert_8i_u_H */
#ifndef INCLUDED_volk_16i_convert_8i_a_H
#define INCLUDED_volk_16i_convert_8i_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_convert_8i_a_avx2(int8_t* outputVector,
                                              const int16_t* inputVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    int8_t* outputVectorPtr = outputVector;
    int16_t* inputPtr = (int16_t*)inputVector;
    __m256i inputVal1;
    __m256i inputVal2;
    __m256i ret;

    for (; number < thirtysecondPoints; number++) {

        // Load the 16 values
        inputVal1 = _mm256_load_si256((__m256i*)inputPtr);
        inputPtr += 16;
        inputVal2 = _mm256_load_si256((__m256i*)inputPtr);
        inputPtr += 16;

        inputVal1 = _mm256_srai_epi16(inputVal1, 8);
        inputVal2 = _mm256_srai_epi16(inputVal2, 8);

        ret = _mm256_packs_epi16(inputVal1, inputVal2);
        ret = _mm256_permute4x64_epi64(ret, 0b11011000);

        _mm256_store_si256((__m256i*)outputVectorPtr, ret);

        outputVectorPtr += 32;
    }

    number = thirtysecondPoints * 32;
    for (; number < num_points; number++) {
        outputVector[number] = (int8_t)(inputVector[number] >> 8);
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16i_convert_8i_a_avx512bw(int8_t* outputVector,
                                                  const int16_t* inputVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixtyfourthPoints = num_points / 64;

    int8_t* outputVectorPtr = outputVector;
    int16_t* inputPtr = (int16_t*)inputVector;
    __m512i inputVal1;
    __m512i inputVal2;
    __m512i shifted1, shifted2;
    __m256i ret1, ret2;

    for (; number < sixtyfourthPoints; number++) {

        // Load 64 int16 values
        inputVal1 = _mm512_load_si512((__m512i*)inputPtr);
        inputPtr += 32;
        inputVal2 = _mm512_load_si512((__m512i*)inputPtr);
        inputPtr += 32;

        shifted1 = _mm512_srai_epi16(inputVal1, 8);
        shifted2 = _mm512_srai_epi16(inputVal2, 8);

        ret1 = _mm512_cvtsepi16_epi8(shifted1);
        ret2 = _mm512_cvtsepi16_epi8(shifted2);

        _mm256_store_si256((__m256i*)outputVectorPtr, ret1);
        outputVectorPtr += 32;
        _mm256_store_si256((__m256i*)outputVectorPtr, ret2);
        outputVectorPtr += 32;
    }

    number = sixtyfourthPoints * 64;
    for (; number < num_points; number++) {
        outputVector[number] = (int8_t)(inputVector[number] >> 8);
    }
}
#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16i_convert_8i_a_sse2(int8_t* outputVector,
                                              const int16_t* inputVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    int8_t* outputVectorPtr = outputVector;
    int16_t* inputPtr = (int16_t*)inputVector;
    __m128i inputVal1;
    __m128i inputVal2;
    __m128i ret;

    for (; number < sixteenthPoints; number++) {

        // Load the 16 values
        inputVal1 = _mm_load_si128((__m128i*)inputPtr);
        inputPtr += 8;
        inputVal2 = _mm_load_si128((__m128i*)inputPtr);
        inputPtr += 8;

        inputVal1 = _mm_srai_epi16(inputVal1, 8);
        inputVal2 = _mm_srai_epi16(inputVal2, 8);

        ret = _mm_packs_epi16(inputVal1, inputVal2);

        _mm_store_si128((__m128i*)outputVectorPtr, ret);

        outputVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (int8_t)(inputVector[number] >> 8);
    }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16i_convert_8i_neon(int8_t* outputVector,
                                            const int16_t* inputVector,
                                            unsigned int num_points)
{
    int8_t* outputVectorPtr = outputVector;
    const int16_t* inputVectorPtr = inputVector;
    unsigned int number = 0;
    unsigned int sixteenth_points = num_points / 16;

    int16x8_t inputVal0;
    int16x8_t inputVal1;
    int8x8_t outputVal0;
    int8x8_t outputVal1;
    int8x16_t outputVal;

    for (number = 0; number < sixteenth_points; number++) {
        // load two input vectors
        inputVal0 = vld1q_s16(inputVectorPtr);
        inputVal1 = vld1q_s16(inputVectorPtr + 8);
        // shift right
        outputVal0 = vshrn_n_s16(inputVal0, 8);
        outputVal1 = vshrn_n_s16(inputVal1, 8);
        // squash two vectors and write output
        outputVal = vcombine_s8(outputVal0, outputVal1);
        vst1q_s8(outputVectorPtr, outputVal);
        inputVectorPtr += 16;
        outputVectorPtr += 16;
    }

    for (number = sixteenth_points * 16; number < num_points; number++) {
        *outputVectorPtr++ = ((int8_t)(*inputVectorPtr++ >> 8));
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16i_convert_8i_neonv8(int8_t* outputVector,
                                              const int16_t* inputVector,
                                              unsigned int num_points)
{
    int8_t* outputVectorPtr = outputVector;
    const int16_t* inputVectorPtr = inputVector;
    const unsigned int thirtysecondPoints = num_points / 32;

    for (unsigned int number = 0; number < thirtysecondPoints; number++) {
        int16x8_t in0 = vld1q_s16(inputVectorPtr);
        int16x8_t in1 = vld1q_s16(inputVectorPtr + 8);
        int16x8_t in2 = vld1q_s16(inputVectorPtr + 16);
        int16x8_t in3 = vld1q_s16(inputVectorPtr + 24);
        __VOLK_PREFETCH(inputVectorPtr + 64);

        int8x8_t out0 = vshrn_n_s16(in0, 8);
        int8x8_t out1 = vshrn_n_s16(in1, 8);
        int8x8_t out2 = vshrn_n_s16(in2, 8);
        int8x8_t out3 = vshrn_n_s16(in3, 8);

        vst1q_s8(outputVectorPtr, vcombine_s8(out0, out1));
        vst1q_s8(outputVectorPtr + 16, vcombine_s8(out2, out3));

        inputVectorPtr += 32;
        outputVectorPtr += 32;
    }

    for (unsigned int number = thirtysecondPoints * 32; number < num_points; number++) {
        *outputVectorPtr++ = ((int8_t)(*inputVectorPtr++ >> 8));
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16i_convert_8i_rvv(int8_t* outputVector,
                                           const int16_t* inputVector,
                                           unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e16m8(n);
        vint16m8_t v = __riscv_vle16_v_i16m8(inputVector, vl);
        __riscv_vse8(outputVector, __riscv_vnsra(v, 8, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_16i_convert_8i_a_H */
