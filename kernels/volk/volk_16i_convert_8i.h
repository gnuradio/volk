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
 * Converts 16-bit shorts to 8-bit chars.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_convert_8i(int8_t* outputVector, const int16_t* inputVector, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector of 16-bit shorts.
 * \li num_points: The number of complex data points.
 *
 * \b Outputs
 * \li outputVector: The output vector of 8-bit chars.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_16i_convert_8i();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_16i_convert_8i_u_H
#define INCLUDED_volk_16i_convert_8i_u_H

#include <inttypes.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16i_convert_8i_generic(int8_t* outputVector,
                                               const int16_t* inputVector,
                                               unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; ++number) {
        *outputVector++ = (int8_t)(*inputVector++ >> 8);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_convert_8i_u_avx2(int8_t* outputVector,
                                              const int16_t* inputVector,
                                              unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {

        // Load the 16 values
        const __m256i inputVal1 = _mm256_loadu_si256((__m256i*)inputVector);
        inputVector += 16;
        const __m256i inputVal2 = _mm256_loadu_si256((__m256i*)inputVector);
        inputVector += 16;

        const __m256i shifted1 = _mm256_srai_epi16(inputVal1, 8);
        const __m256i shifted2 = _mm256_srai_epi16(inputVal2, 8);

        const __m256i packed = _mm256_packs_epi16(shifted1, shifted2);
        const __m256i output = _mm256_permute4x64_epi64(packed, 0b11011000);

        _mm256_storeu_si256((__m256i*)outputVector, output);
        outputVector += 32;
    }

    volk_16i_convert_8i_generic(
        outputVector, inputVector, num_points - thirtysecondPoints * 32);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16i_convert_8i_u_avx512bw(int8_t* outputVector,
                                                  const int16_t* inputVector,
                                                  unsigned int num_points)
{
    const unsigned int sixtyfourthPoints = num_points / 64;

    for (unsigned int number = 0; number < sixtyfourthPoints; ++number) {

        // Load 64 int16 values
        const __m512i inputVal1 = _mm512_loadu_si512((__m512i*)inputVector);
        inputVector += 32;
        const __m512i inputVal2 = _mm512_loadu_si512((__m512i*)inputVector);
        inputVector += 32;

        const __m512i shifted1 = _mm512_srai_epi16(inputVal1, 8);
        const __m512i shifted2 = _mm512_srai_epi16(inputVal2, 8);

        const __m256i output1 = _mm512_cvtsepi16_epi8(shifted1);
        const __m256i output2 = _mm512_cvtsepi16_epi8(shifted2);

        _mm256_storeu_si256((__m256i*)outputVector, output1);
        outputVector += 32;
        _mm256_storeu_si256((__m256i*)outputVector, output2);
        outputVector += 32;
    }

    volk_16i_convert_8i_generic(
        outputVector, inputVector, num_points - sixtyfourthPoints * 64);
}
#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16i_convert_8i_u_sse2(int8_t* outputVector,
                                              const int16_t* inputVector,
                                              unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {

        // Load the 16 values
        const __m128i inputVal1 = _mm_loadu_si128((__m128i*)inputVector);
        inputVector += 8;
        const __m128i inputVal2 = _mm_loadu_si128((__m128i*)inputVector);
        inputVector += 8;

        const __m128i shifted1 = _mm_srai_epi16(inputVal1, 8);
        const __m128i shifted2 = _mm_srai_epi16(inputVal2, 8);

        const __m128i output = _mm_packs_epi16(shifted1, shifted2);

        _mm_storeu_si128((__m128i*)outputVector, output);
        outputVector += 16;
    }

    volk_16i_convert_8i_generic(
        outputVector, inputVector, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_SSE2 */


#endif /* INCLUDED_volk_16i_convert_8i_u_H */
#ifndef INCLUDED_volk_16i_convert_8i_a_H
#define INCLUDED_volk_16i_convert_8i_a_H

#include <inttypes.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_convert_8i_a_avx2(int8_t* outputVector,
                                              const int16_t* inputVector,
                                              unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {

        // Load the 16 values
        const __m256i inputVal1 = _mm256_load_si256((__m256i*)inputVector);
        inputVector += 16;
        const __m256i inputVal2 = _mm256_load_si256((__m256i*)inputVector);
        inputVector += 16;

        const __m256i shifted1 = _mm256_srai_epi16(inputVal1, 8);
        const __m256i shifted2 = _mm256_srai_epi16(inputVal2, 8);

        const __m256i packed = _mm256_packs_epi16(shifted1, shifted2);
        const __m256i output = _mm256_permute4x64_epi64(packed, 0b11011000);

        _mm256_store_si256((__m256i*)outputVector, output);
        outputVector += 32;
    }

    volk_16i_convert_8i_generic(
        outputVector, inputVector, num_points - thirtysecondPoints * 32);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16i_convert_8i_a_avx512bw(int8_t* outputVector,
                                                  const int16_t* inputVector,
                                                  unsigned int num_points)
{
    const unsigned int sixtyfourthPoints = num_points / 64;

    for (unsigned int number = 0; number < sixtyfourthPoints; ++number) {

        // Load 64 int16 values
        const __m512i inputVal1 = _mm512_load_si512((__m512i*)inputVector);
        inputVector += 32;
        const __m512i inputVal2 = _mm512_load_si512((__m512i*)inputVector);
        inputVector += 32;

        const __m512i shifted1 = _mm512_srai_epi16(inputVal1, 8);
        const __m512i shifted2 = _mm512_srai_epi16(inputVal2, 8);

        const __m256i output1 = _mm512_cvtsepi16_epi8(shifted1);
        const __m256i output2 = _mm512_cvtsepi16_epi8(shifted2);

        _mm256_store_si256((__m256i*)outputVector, output1);
        outputVector += 32;
        _mm256_store_si256((__m256i*)outputVector, output2);
        outputVector += 32;
    }

    volk_16i_convert_8i_generic(
        outputVector, inputVector, num_points - sixtyfourthPoints * 64);
}
#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16i_convert_8i_a_sse2(int8_t* outputVector,
                                              const int16_t* inputVector,
                                              unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    for (unsigned int number = 0; number < sixteenthPoints; ++number) {

        // Load the 16 values
        const __m128i inputVal1 = _mm_load_si128((__m128i*)inputVector);
        inputVector += 8;
        const __m128i inputVal2 = _mm_load_si128((__m128i*)inputVector);
        inputVector += 8;

        const __m128i shifted1 = _mm_srai_epi16(inputVal1, 8);
        const __m128i shifted2 = _mm_srai_epi16(inputVal2, 8);

        const __m128i output = _mm_packs_epi16(shifted1, shifted2);

        _mm_store_si128((__m128i*)outputVector, output);
        outputVector += 16;
    }

    volk_16i_convert_8i_generic(
        outputVector, inputVector, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16i_convert_8i_neon(int8_t* outputVector,
                                            const int16_t* inputVector,
                                            unsigned int num_points)
{
    const unsigned int sixteenth_points = num_points / 16;

    for (unsigned int number = 0; number < sixteenth_points; ++number) {
        // load two input vectors
        const int16x8_t inputVal0 = vld1q_s16(inputVector);
        const int16x8_t inputVal1 = vld1q_s16(inputVector + 8);
        // shift right
        const int8x8_t outputVal0 = vshrn_n_s16(inputVal0, 8);
        const int8x8_t outputVal1 = vshrn_n_s16(inputVal1, 8);
        // squash two vectors and write output
        const int8x16_t outputVal = vcombine_s8(outputVal0, outputVal1);
        vst1q_s8(outputVector, outputVal);
        inputVector += 16;
        outputVector += 16;
    }

    volk_16i_convert_8i_generic(
        outputVector, inputVector, num_points - sixteenth_points * 16);
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16i_convert_8i_neonv8(int8_t* outputVector,
                                              const int16_t* inputVector,
                                              unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;

    for (unsigned int number = 0; number < thirtysecondPoints; ++number) {
        const int16x8_t in0 = vld1q_s16(inputVector);
        const int16x8_t in1 = vld1q_s16(inputVector + 8);
        const int16x8_t in2 = vld1q_s16(inputVector + 16);
        const int16x8_t in3 = vld1q_s16(inputVector + 24);
        __VOLK_PREFETCH(inputVector + 64);

        const int8x8_t out0 = vshrn_n_s16(in0, 8);
        const int8x8_t out1 = vshrn_n_s16(in1, 8);
        const int8x8_t out2 = vshrn_n_s16(in2, 8);
        const int8x8_t out3 = vshrn_n_s16(in3, 8);

        vst1q_s8(outputVector, vcombine_s8(out0, out1));
        vst1q_s8(outputVector + 16, vcombine_s8(out2, out3));

        inputVector += 32;
        outputVector += 32;
    }

    volk_16i_convert_8i_generic(
        outputVector, inputVector, num_points - thirtysecondPoints * 32);
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16i_convert_8i_rvv(int8_t* outputVector,
                                           const int16_t* inputVector,
                                           unsigned int num_points)
{
    size_t remaining = num_points;
    for (size_t vl; remaining > 0;
         remaining -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e16m8(remaining);
        vint16m8_t v = __riscv_vle16_v_i16m8(inputVector, vl);
        __riscv_vse8(outputVector, __riscv_vnsra(v, 8, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_16i_convert_8i_a_H */
